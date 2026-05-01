import random
import copy
import os
import json
import torch
from improvnet.tokenizer.midi import MidiDict
from improvnet.tokenizer.absolute import AbsTokenizer


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
        
    def skyline_groundline(self, tokens: list, algorithm: str) -> list:
        """
        Keeps only the highest pitch note from each chord (skyline) or masks it (groundline).
        Replaces masked notes completely with <BLANK> across all 5 attributes.
        """
        blank_tuple = ('<BLANK>', '<BLANK>', '<BLANK>', '<BLANK>', '<BLANK>')
        notes = []
        
        for i, t in enumerate(tokens):
            # Ensure it's a standard note compound token (not a special string token like <S>)
            if isinstance(t, tuple) and len(t) == 5 and isinstance(t[0], tuple) and t[0][0] == 'instrument':
                notes.append({
                    "pitch": t[1][1],
                    "onset": t[3][1] if t[3][1] is not None else 0,
                    "orig_idx": i
                })

        if not notes:
            return list(tokens)

        # Group notes into chords (notes with onset within 50ms)
        chords = []
        i = 0
        k = len(notes)
        WINDOW_MS = 50
        
        while i < k:
            start = notes[i]["onset"]
            chord = [notes[i]]
            j = i + 1
            while j < k and abs(notes[j]["onset"] - start) <= WINDOW_MS:
                chord.append(notes[j])
                j += 1
            chords.append(chord)
            i = j  # BUG FIX: Skip ahead past the notes we just grouped

        # Choose melody per chord (highest pitch)
        melody_pitch_indices = set()
        for chord in chords:
            max_pitch = max(n["pitch"] for n in chord)
            candidates = [n for n in chord if n["pitch"] == max_pitch]
            # Choose the first one encountered (earliest in original sequence)
            chosen = min(candidates, key=lambda n: n["orig_idx"])
            melody_pitch_indices.add(chosen["orig_idx"])

        out_tokens = list(tokens)
        for note in notes:
            idx = note["orig_idx"]
            if algorithm == "skyline" and idx not in melody_pitch_indices:
                # Mask non-melody notes
                out_tokens[idx] = blank_tuple
            elif algorithm == "groundline" and idx in melody_pitch_indices:
                # Mask melody notes
                out_tokens[idx] = blank_tuple

        return out_tokens

    def extract_rhythm(self, tokens: list, ratio: float = 1.0) -> list:
        """
        Masks instrument, pitch, and velocity fields in selected notes with <BLANK>.
        Leaves onset and duration intact.
        """
        augmented_tokens = copy.deepcopy(tokens)

        # Find indices of valid note compound tokens
        note_indices = [
            i for i, event in enumerate(augmented_tokens)
            if isinstance(event, tuple) and len(event) == 5 and isinstance(event[0], tuple) and event[0][0] == 'instrument'
        ]

        # Select notes to mask
        num_to_mask = int(len(note_indices) * ratio)
        mask_notes = random.sample(note_indices, num_to_mask)

        for i in mask_notes:
            event_list = list(augmented_tokens[i])
            for j, field in enumerate(event_list):
                # Mask Instrument, Pitch, and Velocity
                if isinstance(field, tuple) and field[0] in ('instrument', 'pitch', 'velocity'):
                    event_list[j] = '<BLANK>'
            augmented_tokens[i] = tuple(event_list)

        return augmented_tokens
    
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
    
    def format_into_patches(
        self, 
        token_tensors: dict[str, torch.Tensor], 
        patch_size: int, 
        n_patches: int
    ) -> torch.Tensor:
        """
        Pads sequences to the target length and reshapes them into patches.
        Returns a tensor of shape [n_patches, patch_size, 5].
        """
        token_types = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
        target_len = patch_size * n_patches
        seq_len = len(token_tensors['instrument'])
        
        stacked_tensors = []
        for token_type in token_types:
            tensor = token_tensors[token_type]
            
            # Truncate if somehow longer than the target length
            if seq_len > target_len:
                tensor = tensor[:target_len]
            
            # Pad if shorter than the target length
            elif seq_len < target_len:
                # Retrieve the specific padding ID for this attribute vocabulary
                tok_to_id = getattr(self.tokenizer, f"tok_to_id_{token_type}")
                pad_id = tok_to_id.get('<P>')
                if pad_id is None:
                    raise KeyError(f"Padding token '<P>' not found in vocab for '{token_type}'")
                
                padding = torch.full((target_len - seq_len,), pad_id, dtype=torch.long)
                tensor = torch.cat([tensor, padding])
                
            stacked_tensors.append(tensor)

        # Stack into shape: [target_len, 5]
        combined = torch.stack(stacked_tensors, dim=-1)
        
        # Reshape into patches: [n_patches, patch_size, 5]
        patched = combined.view(n_patches, patch_size, len(token_types))
        return patched