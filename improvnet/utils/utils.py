import random
import copy
import os
import json
import torch
import math
import torch.distributed as dist
from improvnet.tokenizer.midi import MidiDict
from improvnet.tokenizer.absolute import AbsTokenizer

# Constants for Diffusion
MAX_DIFFUSION_STEPS = 1000 

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        rank = 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = 1
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

def read_jsonl_files(data_dirs, split="train"):
    files = []
    for file in data_dirs:
        if os.path.exists(file):
            with open(file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data["split"] == split:
                        files.append(data)
        else:
            print(f"Warning: {file} does not exist. Skipping.")
    return files

class ProcessData:
    def __init__(self):
        self.tokenizer = AbsTokenizer()
        # Precompute suffix products for efficient masking (Linear schedule)
        steps = torch.arange(MAX_DIFFUSION_STEPS + 1, dtype=torch.float32)
        mask_probs = steps / MAX_DIFFUSION_STEPS
        # suffix_prod[t] = Product(alpha_t ... alpha_T)
        self.suffix_prod = torch.cumprod(mask_probs.flip(0), dim=0).flip(0)

    def read_midi(self, file_path: str) -> MidiDict:
        try:
            midi_dict = MidiDict.from_midi(file_path)
        except Exception as e:
            print(f"Error reading MIDI file {file_path}: {e}")
            raise e
        return midi_dict
    
    def save_midi(self, midi_dict: MidiDict, file_path: str):
        try:
            midi_dict.save(file_path)
        except Exception as e:
            print(f"Error saving MIDI file {file_path}: {e}")
            raise e
    
    def midi_to_tokens(self, midi_dict: MidiDict) -> list:
        return self.tokenizer.tokenize(midi_dict)
    
    def tokens_to_midi(self, tokens: list) -> MidiDict:
        midi_dict = self.tokenizer.detokenize(tokens)
        return midi_dict.to_midi()
    
    def tokens_to_tensor(self, tokens: list) -> dict[str, torch.Tensor]:
        token_types = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
        token_type_lists = {t: [] for t in token_types}

        for compound_event in tokens:
            if not isinstance(compound_event, tuple) or len(compound_event) != len(token_types):
                # If it's a raw mask string, we might skip or handle it, 
                # but typically we expect tuples here.
                raise ValueError(f"Invalid compound token format: {compound_event}")

            for i, token_type in enumerate(token_types):
                field = compound_event[i]
                tok_to_id = getattr(self.tokenizer, f"tok_to_id_{token_type}")

                if isinstance(field, tuple):
                    if (len(field) == 2 or len(field) == 3) and field in tok_to_id:
                        token_type_lists[token_type].append(tok_to_id[field])
                    else:
                        raise KeyError(f"Token {field} not found in vocab for type '{token_type}'")
                elif isinstance(field, str):
                    if field in tok_to_id:
                        token_type_lists[token_type].append(tok_to_id[field])
                    else:
                        raise KeyError(f"Special token '{field}' not found in vocab for type '{token_type}'")
                else:
                    raise TypeError(f"Unexpected token field type ({type(field)}): {field}")

        return {t: torch.tensor(ids, dtype=torch.long) for t, ids in token_type_lists.items()}
    
    def tensor_to_tokens(self, token_tensors: dict[str, torch.Tensor]) -> list:
        token_types = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
        lengths = [len(token_tensors[t]) for t in token_types]
        num_events = lengths[0]
        id_to_tok_maps = {t: getattr(self.tokenizer, f"id_to_tok_{t}") for t in token_types}
        compound_tokens = []

        for i in range(num_events):
            event = []
            for token_type in token_types:
                tok_id = token_tensors[token_type][i].item()
                id_to_tok = id_to_tok_maps[token_type]
                if tok_id not in id_to_tok:
                    raise KeyError(f"Invalid token ID {tok_id} for type '{token_type}'.")
                event.append(id_to_tok[tok_id])
            compound_tokens.append(tuple(event))
        return compound_tokens

    def genre_form_to_tensor(self, genre: str | None, form: str | None) -> dict[str, torch.Tensor]:
        genre_token = genre if genre is not None else 'unknown'
        form_token = form if form is not None else 'unknown'
        genre_id = self.tokenizer.tok_to_id_genre.get(genre_token, self.tokenizer.tok_to_id_genre['unknown'])
        form_id = self.tokenizer.tok_to_id_form.get(form_token, self.tokenizer.tok_to_id_form['unknown'])
        return {
            'genre': torch.tensor([genre_id], dtype=torch.long),
            'form': torch.tensor([form_id], dtype=torch.long)
        }

    def split_instrument_tokens(self, tokens: list) -> tuple[list, list | None, str | None]:
        """
        Splits a compound token sequence into two groups based on a randomly chosen instrument.
        
        Routes:
        1. Prefix tokens -> Only to the channel of that specific instrument.
        2. Note tokens -> Only to the channel of that specific instrument.
        3. Structural tokens (<S>, <E>) -> To BOTH channels (to maintain valid sequence structure).
        """
        
        # 1. Identify instruments present in the NOTES
        # We scan note events, not prefixes, to ensure we split based on actual content
        unique_instruments = {
            tok[0][1] for tok in tokens
            if isinstance(tok, tuple) and len(tok) > 0 
            and isinstance(tok[0], tuple) and len(tok[0]) == 2 
            and tok[0][0] == "instrument"
        }

        # If 0 or 1 instrument, return as Main, with None for Accompaniment
        if len(unique_instruments) <= 1:
            return tokens, None, None

        selected_instrument = random.choice(list(unique_instruments))
        accompaniment_tokens = []
        original_tokens = []

        for tok in tokens:
            if not isinstance(tok, tuple) or len(tok) == 0:
                continue 
            
            first_field = tok[0]

            # --- Case A: Prefix Token ---
            # Format: (('prefix', 'instrument', 'Name'), ...)
            if isinstance(first_field, tuple) and len(first_field) == 3 and first_field[0] == "prefix":
                inst_name = first_field[2]
                if inst_name == selected_instrument:
                    accompaniment_tokens.append(tok)
                else:
                    original_tokens.append(tok)

            # --- Case B: Note Token ---
            # Format: (('instrument', 'Name'), ...)
            elif isinstance(first_field, tuple) and len(first_field) == 2 and first_field[0] == "instrument":
                inst_name = first_field[1]
                if inst_name == selected_instrument:
                    accompaniment_tokens.append(tok)
                else:
                    original_tokens.append(tok)

            # --- Case C: Structural Tokens (<S>, <E>, etc.) ---
            # Format: ('<S>', '<S>', ...)
            else:
                # Send structural tokens to BOTH streams so they remain valid sequences
                accompaniment_tokens.append(tok)
                original_tokens.append(tok)

        return original_tokens, accompaniment_tokens, selected_instrument

    def apply_diffusion_mask(self, tokens: list, timestep: int, mask_token: str = "<MASK>") -> list:
        """
        Applies Trajectory Re-composition Masking.
        - Masks each attribute INDEPENDENTLY based on the effective probability.
        - SKIPS masking for:
            1. Start Token (<S>)
            2. Prefix Tokens (('prefix', ...))
        - Masks ALL other tokens (Notes, <E>, <T>, etc.)
        """
        t_idx = min(timestep, MAX_DIFFUSION_STEPS)
        effective_prob = self.suffix_prod[t_idx].item()
        
        masked_tokens = []
        for tok in tokens:
            # --- Safety Check ---
            if not isinstance(tok, tuple) or len(tok) == 0:
                masked_tokens.append(tok)
                continue

            first_field = tok[0]

            # --- 1. Check for Exclusion (Start Token) ---
            # Checks if the tuple is ('<S>', '<S>', ...)
            is_start_token = (first_field == '<S>')

            # --- 2. Check for Exclusion (Prefix Token) ---
            # Checks if the first field is (('prefix', ...), ...)
            is_prefix_token = (isinstance(first_field, tuple) and len(first_field) > 0 and first_field[0] == 'prefix')

            if is_start_token or is_prefix_token:
                masked_tokens.append(tok)
                continue

            # --- 3. Apply Independent Attribute Masking ---
            # This applies to Notes, <E>, and any other structural tokens not excluded above
            new_tok = []
            for attr in tok:
                if random.random() < effective_prob:
                    new_tok.append(mask_token)
                else:
                    new_tok.append(attr)
            masked_tokens.append(tuple(new_tok))
        
        return masked_tokens
    
    def pitch_augmentation(self, tokens: list) -> list:
        semitone_shift = random.randint(-7, 7)
        augmented_tokens = copy.deepcopy(tokens)
        for i, event in enumerate(augmented_tokens):
            if not isinstance(event, tuple): continue
            new_event = list(event)
            for j, field in enumerate(event):
                if isinstance(field, tuple) and field[0] == 'pitch':
                    new_pitch = max(0, min(127, field[1] + semitone_shift))
                    new_event[j] = ('pitch', new_pitch)
            augmented_tokens[i] = tuple(new_event)
        return augmented_tokens

    def get_random_segment_from_data(self, tokens: list, segment_length: int) -> list:
        if len(tokens) <= segment_length:
            return tokens
        max_start = len(tokens) - segment_length
        start = random.randint(0, max_start)
        return tokens[start : start + segment_length]

    def pretraining_pipeline(self, file_path: str, genre: str | None, form: str | None,
                         segment_length: int, **kwargs) -> tuple:
        """
        New Pipeline for Amortized ImprovNet.
        - Swaps channels 50% of the time (even for Solo).
        - Pads empty channels with '2' (PAD) to match the length of the active channel.
        """
        # 1. Read and Preprocess
        midi_dict = self.read_midi(file_path)
        tokens = self.midi_to_tokens(midi_dict)
        segment_tokens = self.get_random_segment_from_data(tokens, segment_length)

        if kwargs.get('apply_pitch_augmentation', True):
            segment_tokens = self.pitch_augmentation(segment_tokens)

        # 2. Split Main vs Accompaniment
        main_tokens, accomp_tokens, _ = self.split_instrument_tokens(segment_tokens)
        
        # Normalize None to []
        if main_tokens is None: main_tokens = []
        if accomp_tokens is None: accomp_tokens = []

        # 3. Channel Swapping (Data Augmentation)
        # Always swap 50% of the time, regardless of content
        if random.random() < 0.5:
            main_tokens, accomp_tokens = accomp_tokens, main_tokens

        # 4. Sample Diffusion Timestep t ~ U[1, T]
        timestep = random.randint(1, MAX_DIFFUSION_STEPS)

        # 5. Create Encoder Input (Noisy)
        # Apply mask to whatever is in the channel (could be empty)
        encoder_main = self.apply_diffusion_mask(main_tokens, timestep)
        encoder_accom = self.apply_diffusion_mask(accomp_tokens, timestep)

        # 6. Convert to Tensors (with smart padding)
        # If a channel is empty, we create a PAD tensor of the same length as the OTHER channel.
        # This ensures the model sees [Active, PAD] or [PAD, Active].
        
        def to_tensors_or_pad(target_tokens, ref_tokens):
            if target_tokens:
                return self.tokens_to_tensor(target_tokens)
            elif ref_tokens:
                # Target is empty, but Ref has content -> Create PADs of len(Ref)
                ref_tensors = self.tokens_to_tensor(ref_tokens)
                # Fill with 2 (PAD ID)
                return {k: torch.full_like(v, fill_value=2) for k, v in ref_tensors.items()}
            else:
                # Both empty? (Shouldn't happen with valid files, but safe fallback)
                # Return single step PAD
                dummy = torch.tensor([2], dtype=torch.long)
                keys = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
                return {k: dummy for k in keys}

        # Decoder Tensors (Clean Targets)
        dec_main_tensors = to_tensors_or_pad(main_tokens, accomp_tokens)
        dec_accom_tensors = to_tensors_or_pad(accomp_tokens, main_tokens)

        # Encoder Tensors (Noisy Sources)
        enc_main_tensors = to_tensors_or_pad(encoder_main, encoder_accom)
        enc_accom_tensors = to_tensors_or_pad(encoder_accom, encoder_main)

        # Conditions
        genre_form_dict = self.genre_form_to_tensor(genre, form)
        timestep_tensor = torch.tensor([timestep], dtype=torch.long)

        return (
            enc_main_tensors,
            enc_accom_tensors,
            dec_main_tensors,
            dec_accom_tensors,
            genre_form_dict['genre'],
            genre_form_dict['form'],
            timestep_tensor
        )
    
    def inference_pipeline(self, file_path: str, genre: str | None, form: str | None):
        """
        Reads a MIDI file and prepares the FULL sequence without cropping.
        Returns lists of tokens (not tensors) to be sliced by the generator.
        """
        # 1. Read
        midi_dict = self.read_midi(file_path)
        tokens = self.midi_to_tokens(midi_dict)
        
        # 2. Split (No Augmentation or Cropping)
        main_tokens, accomp_tokens, _ = self.split_instrument_tokens(tokens)
        
        if main_tokens is None: main_tokens = []
        if accomp_tokens is None: accomp_tokens = []

        # 3. Prepare Conditions
        genre_form = self.genre_form_to_tensor(genre, form)
        
        return main_tokens, accomp_tokens, genre_form['genre'], genre_form['form']