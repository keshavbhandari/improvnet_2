import random
import copy
import os
import json
import torch
import torch.distributed as dist
from improvnet.tokenizer.midi import MidiDict
from improvnet.tokenizer.absolute import AbsTokenizer


def setup_distributed():
    """
    Expects torchrun / torch.distributed to set RANK, WORLD_SIZE, LOCAL_RANK, etc.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        # Fallback single-process
        rank = 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = 1

    # Set device for this process
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

def read_jsonl_files(data_dirs, split="train"):
    """
    Reads all jsonl files in the specified directories and returns a list of dictionaries.
    """
    files = []
    for file in data_dirs:
        if os.path.exists(file):
            with open(file, 'r') as f:
                for line in f:
                    # Parse the JSON string into a dictionary
                    data = json.loads(line.strip())
                    # Now you can access the dictionary keys
                    if data["split"] == split:
                        files.append(data)
        else:
            print(f"Warning: {file} does not exist. Skipping.")
    return files

# Example compound tokens:
# [(('prefix', 'instrument', 'Acoustic Bass'), '<BLANK>', '<BLANK>', '<BLANK>', '<BLANK>'), (('prefix', 'instrument', 'drum'), '<BLANK>', '<BLANK>', '<BLANK>', '<BLANK>'), ('<S>', '<S>', '<S>', '<S>', '<S>'), (('instrument', 'Acoustic Piano'), ('pitch', 71), ('velocity', 70), ('onset', 0), ('dur', 1360)), (('instrument', 'Acoustic Piano'), ('pitch', 68), ('velocity', 50), ('onset', 20), ('dur', 1330)), (('instrument', 'Acoustic Piano'), ('pitch', 65), ('velocity', 50), ('onset', 40), ('dur', 1300)), (('instrument', 'Acoustic Piano'), ('pitch', 56), ('velocity', 40), ('onset', 50), ('dur', 1330)), (('instrument', 'Acoustic Piano'), ('pitch', 62), ('velocity', 40), ('onset', 50), ('dur', 1290)), (('instrument', 'Acoustic Piano'), ('pitch', 72), ('velocity', 50), ('onset', 1330), ('dur', 50)), (('instrument', 'Acoustic Piano'), ('pitch', 69), ('velocity', 60), ('onset', 1350), ('dur', 30)), ...  ('<E>', '<E>', '<E>', '<E>', '<E>')]
class ProcessData:
    def __init__(self):
        self.tokenizer = AbsTokenizer()

    def read_midi(self, file_path: str) -> MidiDict:
        """
        Reads a MIDI file and converts it to a MidiDict.
        """
        try:
            midi_dict = MidiDict.from_midi(file_path)
        except Exception as e:
            print(f"Error reading MIDI file {file_path}: {e}")
            raise e
        return midi_dict
    
    def save_midi(self, midi_dict: MidiDict, file_path: str):
        """
        Saves a MidiDict as a MIDI file.
        """
        try:
            midi_dict.save(file_path)
        except Exception as e:
            print(f"Error saving MIDI file {file_path}: {e}")
            raise e
    
    def midi_to_tokens(self, midi_dict: MidiDict) -> list:
        """
        Converts a MidiDict to a list of compound tokens using the Aria tokenizer.
        """
        tokens = self.tokenizer.tokenize(midi_dict)
        return tokens
    
    def tokens_to_midi(self, tokens: list) -> MidiDict:
        """
        Converts a list of compound tokens back to a MidiDict using the tokenizer.
        """
        midi_dict = self.tokenizer.detokenize(tokens)
        midi_out = midi_dict.to_midi()
        return midi_out
    
    def tokens_to_tensor(self, tokens: list) -> dict[str, torch.Tensor]:
        """
        Converts a list of compound tokens into separate tensors per token type.
        
        Assumes each token tuple follows a fixed positional order:
            (instrument, pitch, velocity, onset, duration)
        
        Handles special tuples like ('<S>', '<S>', '<S>', '<S>', '<S>')
        or prefixed instrument tokens like (('prefix', 'instrument', 'Acoustic Bass'), ...)

        Returns:
            dict[str, torch.Tensor]: one tensor per token type
        """

        # Fixed order of attributes
        token_types = ['instrument', 'pitch', 'velocity', 'onset', 'duration']

        # Verify that all expected vocab mappings exist
        for token_type in token_types:
            if not hasattr(self.tokenizer, f"tok_to_id_{token_type}"):
                raise AttributeError(
                    f"Missing tokenizer mapping: self.tok_to_id_{token_type} not found."
                )

        # Initialize lists to hold IDs
        token_type_lists = {t: [] for t in token_types}

        for compound_event in tokens:
            # Must be a tuple of length 5
            if not isinstance(compound_event, tuple) or len(compound_event) != len(token_types):
                raise ValueError(
                    f"Invalid compound token format: {compound_event}"
                )

            for i, token_type in enumerate(token_types):
                field = compound_event[i]
                tok_to_id = getattr(self.tokenizer, f"tok_to_id_{token_type}")  # guaranteed to exist

                # --- Case 1: field is a (token_type, value) tuple ---
                if isinstance(field, tuple):
                    # e.g. ('instrument', 'Acoustic Piano')
                    if len(field) == 2 and field in tok_to_id:
                        token_type_lists[token_type].append(tok_to_id[field])
                    # e.g. ('prefix', 'instrument', 'Acoustic Bass')
                    elif len(field) == 3 and field in tok_to_id:
                        token_type_lists[token_type].append(tok_to_id[field])
                    else:
                        raise KeyError(
                            f"Token {field} not found in vocab for type '{token_type}'"
                        )

                # --- Case 2: field is a special string like '<S>', '<E>', '<BLANK>' ---
                elif isinstance(field, str):
                    if field in tok_to_id:
                        token_type_lists[token_type].append(tok_to_id[field])
                    else:
                        raise KeyError(
                            f"Special token '{field}' not found in vocab for type '{token_type}'"
                        )

                # --- Case 3: unknown format ---
                else:
                    raise TypeError(
                        f"Unexpected token field type ({type(field)}): {field}"
                    )

        # Convert lists to tensors
        token_type_tensors = {
            t: torch.tensor(ids, dtype=torch.long)
            for t, ids in token_type_lists.items()
        }

        return token_type_tensors
    
    def tensor_to_tokens(self, token_tensors: dict[str, torch.Tensor]) -> list:
        """
        Reconstructs the original compound token sequence from per-token-type tensors.
        
        Args:
            token_tensors (dict[str, torch.Tensor]): dictionary where keys are token types
                ('instrument', 'pitch', 'velocity', 'onset', 'dur') and values are 1D tensors
                of token IDs for that type.

        Returns:
            list[tuple]: list of compound token tuples in this order:
                (instrument, pitch, velocity, onset, dur)

        Example output:
            [
                (('instrument', 'Acoustic Piano'), ('pitch', 71), ('velocity', 70), ('onset', 0), ('dur', 1360)),
                ...
            ]
        """
        
        token_types = ['instrument', 'pitch', 'velocity', 'onset', 'duration']

        # --- Sanity checks ---
        for token_type in token_types:
            if token_type not in token_tensors:
                raise KeyError(f"Missing tensor for token type '{token_type}'.")
            if not hasattr(self.tokenizer, f"id_to_tok_{token_type}"):
                raise AttributeError(f"Missing mapping: self.id_to_tok_{token_type} not found.")

        # Ensure all tensors have equal length
        lengths = [len(token_tensors[t]) for t in token_types]
        if len(set(lengths)) != 1:
            raise ValueError(f"All token tensors must have equal length, got {lengths}.")

        num_events = lengths[0]

        # --- Convert each token id back to its symbolic token ---
        id_to_tok_maps = {
            t: getattr(self.tokenizer, f"id_to_tok_{t}") for t in token_types
        }

        compound_tokens = []

        for i in range(num_events):
            event = []
            for token_type in token_types:
                tok_id = token_tensors[token_type][i].item()
                id_to_tok = id_to_tok_maps[token_type]

                if tok_id not in id_to_tok:
                    raise KeyError(
                        f"Invalid token ID {tok_id} for type '{token_type}'."
                    )

                tok = id_to_tok[tok_id]
                event.append(tok)
            
            compound_tokens.append(tuple(event))

        return compound_tokens

    def genre_form_to_tensor(self, genre: str | None, form: str | None) -> dict[str, torch.Tensor]:
        """
        Converts genre and form strings to tensors.
        If genre or form is None, uses 'unknown' token.
        """
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
        Works even if the sequence does not start with prefix instrument tokens (e.g., cropped data).

        Args:
            tokens (list): A list of compound tokens, where each token is a tuple of fields like:
                (('instrument', 'Acoustic Piano'), ('pitch', 71), ('velocity', 70), ('onset', 0), ('dur', 1360))

        Returns:
            tuple:
                original_tokens (list): Tokens excluding the chosen instrument (main mix).
                accompaniment_tokens (list | None): Tokens belonging to the randomly chosen instrument, or None if only one instrument exists.
                selected_instrument (str | None): The instrument name that was split out, or None if no split occurred.
        """

        # --- Find all instruments from note tokens ---
        unique_instruments = {
            tok[0][1]
            for tok in tokens
            if isinstance(tok, tuple)
            and len(tok) > 0
            and isinstance(tok[0], tuple)
            and len(tok[0]) == 2
            and tok[0][0] == "instrument"
        }

        # --- Handle cases with no or only one instrument ---
        if not unique_instruments:
            # No identifiable instruments — return unchanged
            return tokens, None, None

        if len(unique_instruments) == 1:
            # Only one instrument present — skip accompaniment split
            return tokens, None, None

        # --- Randomly pick one instrument for accompaniment ---
        selected_instrument = random.choice(list(unique_instruments))

        accompaniment_tokens = []
        original_tokens = []

        # --- Split tokens based on instrument association ---
        for tok in tokens:
            if not isinstance(tok, tuple) or len(tok) == 0:
                continue  # skip invalid tokens

            first_field = tok[0]

            # Prefix tokens are retained in both streams
            if isinstance(first_field, tuple) and len(first_field) == 3 and first_field[0] == "prefix":
                accompaniment_tokens.append(tok)
                original_tokens.append(tok)

            # Instrument note tokens (actual content)
            elif isinstance(first_field, tuple) and len(first_field) == 2 and first_field[0] == "instrument":
                if first_field[1] == selected_instrument:
                    accompaniment_tokens.append(tok)
                else:
                    original_tokens.append(tok)

            # Other tokens (shared/global elements)
            else:
                accompaniment_tokens.append(tok)
                original_tokens.append(tok)

        return original_tokens, accompaniment_tokens, selected_instrument
    
    def pitch_augmentation(self, tokens: list) -> list:
        """
        Shifts pitch values in compound tokens by a random number of semitones.
        Each event is assumed to be of the form:
            (('instrument', ...), ('pitch', val), ('velocity', val), ('onset', val), ('dur', val))
        """
        semitone_shift = random.randint(-7, 7)
        augmented_tokens = copy.deepcopy(tokens)

        for i, event in enumerate(augmented_tokens):
            if not isinstance(event, tuple):
                continue
            new_event = list(event)
            for j, field in enumerate(event):
                if isinstance(field, tuple) and field[0] == 'pitch':
                    new_pitch = field[1] + semitone_shift
                    new_pitch = max(0, min(127, new_pitch))  # clamp MIDI pitch
                    new_event[j] = ('pitch', new_pitch)
            augmented_tokens[i] = tuple(new_event)

        return augmented_tokens
    
    def whole_mask(self, tokens: list, mask_token: str, ratio: float) -> list:
        """
        Masks a percentage of compound tokens completely (all attributes),
        including string-only tuples like ('<S>', '<S>', ...).
        """
        num_tokens = len(tokens)
        num_mask = int(num_tokens * ratio)
        mask_indices = random.sample(range(num_tokens), num_mask)
        
        masked_tokens = copy.deepcopy(tokens)
        for idx in mask_indices:
            tok = masked_tokens[idx]
            if isinstance(tok, tuple):
                masked_tokens[idx] = tuple([mask_token] * len(tok))
        return masked_tokens
    
    def random_mask(self, tokens: list, mask_token: str, ratio: float) -> list:
        """
        Randomly masks a subset of attributes within selected compound tokens.
        
        Simplified logic:
        1. Select exactly 'ratio' percent of tokens.
        2. For each selected token, mask a random number of attributes (1 to all).
        """
        num_tokens = len(tokens)
        num_mask = int(num_tokens * ratio)
        mask_indices = random.sample(range(num_tokens), num_mask)
        
        masked_tokens = copy.deepcopy(tokens)
        
        for idx in mask_indices:
            tok = masked_tokens[idx]
            
            # 1. Handle non-tuple tokens (like <S>, <E>)
            if not isinstance(tok, tuple):
                masked_tokens[idx] = mask_token
                continue

            # 2. Handle Compound Tokens
            tok_list = list(tok)
            num_attrs = len(tok_list)
            
            # Determine how many attributes to mask (ensure at least 1)
            # This makes the corruption random but guaranteed
            count_to_mask = random.randint(1, num_attrs)
            
            # Pick specific indices to mask
            indices_to_mask = random.sample(range(num_attrs), count_to_mask)
            
            new_tok = []
            for i in range(num_attrs):
                if i in indices_to_mask:
                    new_tok.append(mask_token)
                else:
                    new_tok.append(tok_list[i])
            
            masked_tokens[idx] = tuple(new_tok)

        return masked_tokens
    
    def instrument_mask(self, tokens: list, mask_token: str, ratio: float) -> list:
        """
        Masks the instrument attribute in a percentage of compound tokens.
        """
        instrument_indices = [
            i for i, tok in enumerate(tokens)
            if isinstance(tok, tuple)
            and len(tok) > 0
            and isinstance(tok[0], tuple)
            and tok[0][0] == "instrument"
        ]
        num_to_mask = int(len(instrument_indices) * ratio)
        mask_indices = random.sample(instrument_indices, num_to_mask)
        
        masked_tokens = copy.deepcopy(tokens)
        for idx in mask_indices:
            tok = masked_tokens[idx]
            new_tok = [mask_token if j == 0 else tok[j] for j in range(len(tok))]
            masked_tokens[idx] = tuple(new_tok)
        return masked_tokens

    def left_mask(self, tokens: list, mask_token: str, ratio: float) -> list:
        """
        Masks a percentage of compound tokens from the left side of the sequence.
        Each masked token becomes ('<MASK>', '<MASK>', '<MASK>', '<MASK>', '<MASK>').
        """
        num_tokens = len(tokens)
        num_mask = int(num_tokens * ratio)
        masked_tokens = copy.deepcopy(tokens)

        for idx in range(num_mask):
            tok = masked_tokens[idx]
            if isinstance(tok, tuple):
                masked_tokens[idx] = tuple([mask_token] * len(tok))
        return masked_tokens


    def right_mask(self, tokens: list, mask_token: str, ratio: float) -> list:
        """
        Masks a percentage of compound tokens from the right side of the sequence.
        Each masked token becomes ('<MASK>', '<MASK>', '<MASK>', '<MASK>', '<MASK>').
        """
        num_tokens = len(tokens)
        num_mask = int(num_tokens * ratio)
        masked_tokens = copy.deepcopy(tokens)

        for idx in range(num_tokens - num_mask, num_tokens):
            tok = masked_tokens[idx]
            if isinstance(tok, tuple):
                masked_tokens[idx] = tuple([mask_token] * len(tok))
        return masked_tokens


    def center_mask(self, tokens: list, mask_token: str, ratio: float) -> list:
        """
        Masks a percentage of compound tokens from the center of the sequence.
        Each masked token becomes ('<MASK>', '<MASK>', '<MASK>', '<MASK>', '<MASK>').
        """
        num_tokens = len(tokens)
        num_mask = int(num_tokens * ratio)
        start_idx = (num_tokens - num_mask) // 2
        end_idx = start_idx + num_mask

        masked_tokens = copy.deepcopy(tokens)

        for idx in range(start_idx, end_idx):
            tok = masked_tokens[idx]
            if isinstance(tok, tuple):
                masked_tokens[idx] = tuple([mask_token] * len(tok))
        return masked_tokens
    
    def pitch_velocity_mask(self, tokens: list, mask_token: str, ratio: float) -> list:
        """
        Masks both pitch and velocity fields in selected notes with the mask_token.
        """
        augmented_tokens = copy.deepcopy(tokens)

        # Find indices of notes that contain pitch/velocity fields
        note_indices = [
            i for i, event in enumerate(augmented_tokens)
            if isinstance(event, tuple) and any(isinstance(f, tuple) and f[0] in ('pitch','velocity') for f in event)
        ]

        # Select notes to mask
        num_to_mask = int(len(note_indices) * ratio)
        mask_notes = random.sample(note_indices, num_to_mask)

        for i in mask_notes:
            event_list = list(augmented_tokens[i])
            for j, field in enumerate(event_list):
                if isinstance(field, tuple) and field[0] in ('pitch', 'velocity'):
                    event_list[j] = mask_token
            augmented_tokens[i] = tuple(event_list)

        return augmented_tokens

    
    def onset_duration_mask(self, tokens: list, mask_token: str, ratio: float) -> list:
        """
        Masks both onset and duration fields in selected notes with the mask_token.
        """
        augmented_tokens = copy.deepcopy(tokens)

        # Find indices of notes that contain onset/duration fields
        note_indices = [
            i for i, event in enumerate(augmented_tokens)
            if isinstance(event, tuple) and any(isinstance(f, tuple) and f[0] in ('onset','dur') for f in event)
        ]

        # Select notes to mask
        num_to_mask = int(len(note_indices) * ratio)
        mask_notes = random.sample(note_indices, num_to_mask)

        for i in mask_notes:
            event_list = list(augmented_tokens[i])
            for j, field in enumerate(event_list):
                if isinstance(field, tuple) and field[0] in ('onset', 'dur'):
                    event_list[j] = mask_token
            augmented_tokens[i] = tuple(event_list)

        return augmented_tokens
    
    def accompaniment_mask(self, tokens: list, mask_token: str) -> list:
        """
        Keeps only the prefix instrument tokens at the beginning of the sequence.
        All other tokens are replaced with the mask token (full tuple of masks).
        """
        masked_tokens = copy.deepcopy(tokens)

        for i, tok in enumerate(masked_tokens):
            # Check if the token is a prefix instrument token
            if isinstance(tok, tuple) and len(tok) > 0 and isinstance(tok[0], tuple):
                if tok[0][0] == "prefix" and tok[0][1] == "instrument":
                    continue  # keep this token
            # Otherwise, mask out the entire token
            if isinstance(tok, tuple):
                masked_tokens[i] = tuple([mask_token] * len(tok))
            else:
                masked_tokens[i] = mask_token  # fallback for string-only tokens

        return masked_tokens
    
    def skyline_groundline(self, tokens: list, mask_token: str, algorithm: str) -> list:
        """
        Implements the skyline and groundline algorithm.
        Keeps only the highest pitch note from each chord (notes with onset within 50ms).
        """
        notes = []
        for i, t in enumerate(tokens):
            if isinstance(t, tuple) and len(t[0]) == 2 and t[0][0] == 'instrument':
                inst = t[0][1]
                pitch = t[1][1]
                vel = t[2][1]
                onset = t[3][1]
                dur = t[4][1]
                # Look ahead for onset/dur within a small window
                notes.append({
                    "pitch": pitch,
                    "vel": vel,
                    "inst": inst,
                    "onset": onset if onset is not None else 0,
                    "dur": dur if dur is not None else 0,
                    "pitch_idx": i,
                    "orig_idx": i,  # Added this field to track original index
                    "onset_idx": i + 1,  # Approximate position of onset token
                    "dur_idx": i + 2     # Approximate position of duration token
                })

        if not notes:
            return list(tokens)

        # Group notes into chords (notes with onset within 50ms)
        chords = []
        i = 0
        WINDOW_MS = 50
        k = len(notes)
        while i < k:
            start = notes[i]["onset"]
            chord = [notes[i]]
            j = i + 1
            while (j < k) and abs((notes[j]["onset"] - start)) <= WINDOW_MS:
                chord.append(notes[j])
                j = j + 1
                i = i + 1
            chords.append(chord)
            i = i + 1
        # Choose melody per chord (highest pitch)
        melody_pitch_indices = set()
        for chord in chords:
            max_pitch = max(n["pitch"] for n in chord)
            candidates = [n for n in chord if n["pitch"] == max_pitch]
            # Choose the first one encountered (earliest in original sequence)
            chosen = min(candidates, key=lambda n: n["pitch_idx"])
            melody_pitch_indices.add(chosen["pitch_idx"])

        if algorithm == "skyline":
            # Create output by masking non-melody notes
            skyline_out = list(tokens)
            for note in notes:
                if note["pitch_idx"] not in melody_pitch_indices:
                    # Mask the pitch token (maintain tuple structure)
                    if isinstance(skyline_out[note["pitch_idx"]], tuple) and len(skyline_out[note["pitch_idx"]]) == 5:
                      skyline_out[note["pitch_idx"]] = (mask_token, mask_token, mask_token, mask_token, mask_token)

            return skyline_out

        else:
            # Create output by masking melody notes
            groundline_out = list(tokens)
            for note in notes:
                if note["pitch_idx"] in melody_pitch_indices:
                    # Mask the pitch token (maintain tuple structure)
                    if isinstance(groundline_out[note["pitch_idx"]], tuple) and len(groundline_out[note["pitch_idx"]]) == 5:
                      groundline_out[note["pitch_idx"]] = (mask_token, mask_token, mask_token, mask_token, mask_token)

            return groundline_out

    def split_data(self, tokens: list, segment_length: int) -> list:
        """
        Splits the token sequence into segments of specified length.
        """
        segments = []
        for i in range(0, len(tokens), segment_length):
            segment = tokens[i:i + segment_length]
            segments.append(segment)
        return segments
    
    def get_random_segment_from_data(self, tokens: list, segment_length: int, segment_idx=None) -> list:
        """
        Retrieves a random segment of specified length from the token sequence.
        """
        if len(tokens) <= segment_length:
            return tokens
        # Call split_data to get all segments
        segments = self.split_data(tokens, segment_length)
        if segment_idx is not None and 0 <= segment_idx < len(segments):
            return segments[segment_idx]
        # Randomly select one segment
        random_segment = random.choice(segments)
        return random_segment

    # def apply_corruption(self, tokens: list, corruption_type: str, **kwargs) -> tuple[list, list]:
    #     """
    #     Applies the specified corruption technique to a compound token sequence.
    #     Returns:
    #         corrupted_tokens: list of corrupted compound tokens
    #         changed_mask: list of 5-element lists (one per token)
    #     """
    #     NUM_ATTRIBUTES = 5 # Your model has 5 attributes
    #     random_ratio = round(random.uniform(0.0, 1.0), 2)

    #     # Map corruption_type to method
    #     method_map = {
    #         'whole_mask': self.whole_mask,
    #         'random_mask': self.random_mask,
    #         'instrument_mask': self.instrument_mask,
    #         'left_mask': self.left_mask,
    #         'right_mask': self.right_mask,
    #         'center_mask': self.center_mask,
    #         'pitch_velocity_mask': self.pitch_velocity_mask,
    #         'onset_duration_mask': self.onset_duration_mask,
    #         'skyline': self.skyline_groundline,
    #         'groundline': self.skyline_groundline,
    #         'accompaniment_mask': self.accompaniment_mask
    #     }

    #     if corruption_type == 'random':
    #         # Use customizable numerical weights (1-10) ---
            
    #         # 1. Define weights for each corruption type.
    #         # You can customize these values (e.g., from 1 to 10).
    #         corruption_weights = {
    #             'whole_mask': 10,
    #             'left_mask': 7,
    #             'center_mask': 10,
    #             'right_mask': 10,
    #             'random_mask': 5,
    #             'pitch_velocity_mask': 5,
    #             'onset_duration_mask': 8,
    #             'skyline': 5,
    #             'groundline': 5,
    #             'instrument_mask': 2
    #         }

    #         # 2. Create the population and weights lists
    #         # This automatically excludes 'random' and 'accompaniment_mask'
    #         valid_population = list(corruption_weights.keys())
    #         valid_weights = [corruption_weights[k] for k in valid_population]

    #         # 3. Use random.choices() to pick one method based on weights
    #         selected_method = random.choices(valid_population, weights=valid_weights, k=1)[0]

    #         return self.apply_corruption(tokens, selected_method, ratio=random_ratio, **kwargs)

    #     if corruption_type not in method_map:
    #         raise ValueError(f"Unknown corruption type: {corruption_type}")

    #     method = method_map[corruption_type]

    #     # Handle different function signatures
    #     if corruption_type == 'accompaniment_mask':
    #         corrupted_tokens = method(tokens, mask_token="<MASK>")
    #     elif corruption_type in ['skyline', 'groundline']:
    #         # Call as a method of self
    #         corrupted_tokens = method(tokens, mask_token="<MASK>", algorithm=corruption_type)
    #     else:
    #         # All other methods use ratio
    #         ratio = kwargs.get('ratio', random_ratio)
    #         corrupted_tokens = method(tokens, mask_token="<MASK>", ratio=ratio)

    #     # Build binary mask of changed attributes per token
    #     changed_mask = []
    #     for orig_event, corrupt_event in zip(tokens, corrupted_tokens):
    #         # Default mask is all zeros
    #         mask_row = [0] * NUM_ATTRIBUTES 

    #         if orig_event == corrupt_event:
    #             # If they are identical, mask is [0, 0, 0, 0, 0]
    #             pass
    #         elif (isinstance(orig_event, tuple) and isinstance(corrupt_event, tuple) and
    #               len(orig_event) == NUM_ATTRIBUTES and len(corrupt_event) == NUM_ATTRIBUTES):
    #             # This is a standard 5-tuple note that has changed
    #             for i in range(NUM_ATTRIBUTES):
    #                 if orig_event[i] != corrupt_event[i]:
    #                     mask_row[i] = 1
    #         else:
    #             # This is a non-standard token (like <S>) or a token
    #             # that was fully masked. Mark all attributes as changed.
    #             mask_row = [1] * NUM_ATTRIBUTES
            
    #         changed_mask.append(mask_row)

    #     return corrupted_tokens, changed_mask

    def apply_corruption(self, tokens: list, corruption_type: str, **kwargs) -> tuple[list, list]:
        """
        Applies the specified corruption technique to a compound token sequence.
        Returns:
            corrupted_tokens: list of corrupted compound tokens
            changed_mask: list of 5-element lists (one per token)
        """
        NUM_ATTRIBUTES = 5 # Your model has 5 attributes
        random_ratio = round(random.uniform(0.0, 1.0), 2)

        # Map corruption_type to method
        method_map = {
            'whole_mask': self.whole_mask,
            'random_mask': self.random_mask,
            'instrument_mask': self.instrument_mask,
            'left_mask': self.left_mask,
            'right_mask': self.right_mask,
            'center_mask': self.center_mask,
            'pitch_velocity_mask': self.pitch_velocity_mask,
            'onset_duration_mask': self.onset_duration_mask,
            'skyline': self.skyline_groundline,
            'groundline': self.skyline_groundline,
            'accompaniment_mask': self.accompaniment_mask
        }

        if corruption_type == 'random':
            # --- MODIFICATION ---
            # As requested, 'random' corruption now *only* selects 'random_mask'.
            selected_method = 'random_mask'
            # --- END MODIFICATION ---

            return self.apply_corruption(tokens, selected_method, ratio=random_ratio, **kwargs)

        if corruption_type not in method_map:
            raise ValueError(f"Unknown corruption type: {corruption_type}")

        method = method_map[corruption_type]

        # Handle different function signatures
        if corruption_type == 'accompaniment_mask':
            corrupted_tokens = method(tokens, mask_token="<MASK>")
        elif corruption_type in ['skyline', 'groundline']:
            # Call as a method of self
            corrupted_tokens = method(tokens, mask_token="<MASK>", algorithm=corruption_type)
        else:
            # All other methods use ratio
            ratio = kwargs.get('ratio', random_ratio)
            corrupted_tokens = method(tokens, mask_token="<MASK>", ratio=ratio)

        # Build binary mask of changed attributes per token
        changed_mask = []
        for orig_event, corrupt_event in zip(tokens, corrupted_tokens):
            # Default mask is all zeros
            mask_row = [0] * NUM_ATTRIBUTES 

            if orig_event == corrupt_event:
                # If they are identical, mask is [0, 0, 0, 0, 0]
                pass
            elif (isinstance(orig_event, tuple) and isinstance(corrupt_event, tuple) and
                  len(orig_event) == NUM_ATTRIBUTES and len(corrupt_event) == NUM_ATTRIBUTES):
                # This is a standard 5-tuple note that has changed
                for i in range(NUM_ATTRIBUTES):
                    if orig_event[i] != corrupt_event[i]:
                        mask_row[i] = 1
            else:
                # This is a non-standard token (like <S>) or a token
                # that was fully masked. Mark all attributes as changed.
                mask_row = [1] * NUM_ATTRIBUTES
            
            changed_mask.append(mask_row)

        return corrupted_tokens, changed_mask


    def pretraining_pipeline(self, file_path: str, genre: str | None, form: str | None,
                         corruption_type: str, segment_length: int, **kwargs) -> tuple:
        """
        Full pretraining data processing pipeline from MIDI file to corrupted tokens and change indices.
        """
        # 1. Read MIDI → tokens → segment
        midi_dict = self.read_midi(file_path)
        tokens = self.midi_to_tokens(midi_dict)
        segment_tokens = self.get_random_segment_from_data(tokens, segment_length)

        # 2. Optional pitch augmentation
        if kwargs.get('apply_pitch_augmentation', True):
            segment_tokens = self.pitch_augmentation(segment_tokens)

        # 3. Split instruments
        segment_tokens, accomp_tokens, selected_instrument = self.split_instrument_tokens(segment_tokens)

        # 4. Always apply corruption to the main (mix) tokens
        corrupted_mix_tokens, changed_original_indices = self.apply_corruption(segment_tokens, corruption_type, **kwargs)

        if selected_instrument is not None:
            
            corrupted_accomp_tokens, changed_accomp_indices = self.apply_corruption(
                    accomp_tokens, 'random', **kwargs
                )

            if random.random() < 0.5:
                # 1. Swap CORRUPTED inputs (main input now receives accomp stream)
                (
                    corrupted_mix_tokens, corrupted_accomp_tokens
                ) = (
                    corrupted_accomp_tokens, corrupted_mix_tokens
                )
                # 2. Swap CHANGE INDICES (main label now receives accomp label)
                (
                    changed_original_indices, changed_accomp_indices
                ) = (
                    changed_accomp_indices, changed_original_indices
                )
                # 3. Swap UNCORRUPTED ORIGINALS (main ground truth now receives accomp ground truth)
                (
                    segment_tokens, accomp_tokens 
                ) = (
                    accomp_tokens, segment_tokens
                )

        else:
            # 5b. No accompaniment → create placeholders
            corrupted_accomp_tokens, changed_accomp_indices = None, None
            accomp_tokens = None

        # 6. Convert to tensors
        genre_form_dict = self.genre_form_to_tensor(genre, form)
        genre_token_tensor = genre_form_dict['genre']
        form_token_tensor = genre_form_dict['form']
        corr_mix_token_tensors = self.tokens_to_tensor(corrupted_mix_tokens)
        original_mix_token_tensors = self.tokens_to_tensor(segment_tokens)

        if accomp_tokens is not None:
            corr_accomp_token_tensors = self.tokens_to_tensor(corrupted_accomp_tokens)
            original_accomp_token_tensors = self.tokens_to_tensor(accomp_tokens)
            changed_accomp_indices_tensor = torch.tensor(changed_accomp_indices, dtype=torch.bool)
        else:
            # Create zero / padding placeholders
            changed_accomp_indices_tensor = torch.zeros_like(torch.tensor(changed_original_indices, dtype=torch.bool))
            corr_accomp_token_tensors = {
                k: torch.full_like(v, fill_value=2) for k, v in corr_mix_token_tensors.items()
            }
            original_accomp_token_tensors = {
                k: torch.full_like(v, fill_value=2) for k, v in corr_mix_token_tensors.items()
            }

        # 7. Shared changed indices
        changed_original_indices_tensor = torch.tensor(changed_original_indices, dtype=torch.bool)

        return (
            corr_mix_token_tensors,
            corr_accomp_token_tensors,
            changed_original_indices_tensor,
            changed_accomp_indices_tensor,
            original_mix_token_tensors,
            original_accomp_token_tensors,
            genre_token_tensor,
            form_token_tensor
        )
