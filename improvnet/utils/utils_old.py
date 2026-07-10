import random
import copy
import os
import json
import torch
from improvnet.tokenizer.midi import MidiDict
from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.model.caddi_config import GENRES


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

        INSTRUMENT_CLASSES = [
            "Acoustic Piano", "Electric Piano", "Chromatic Percussion", "Organ", 
            "Acoustic Guitar", "Clean Electric Guitar", "Distorted Electric Guitar", 
            "Acoustic Bass", "Electric Bass", "Violin", "Viola", "Cello", "Contrabass", 
            "Orchestral Harp", "Timpani", "String Ensemble", "Synth Strings", 
            "Choir and Voice", "Orchestra Hit", "Trumpet", "Trombone", "Tuba", 
            "French Horn", "Brass Section", "Soprano/Alto Sax", "Tenor Sax", 
            "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet", "Piccolo", 
            "Flute", "Pipe", "Synth Lead", "Synth Pad", "Synth Effect", "Ethnic", 
            "Percussive", "Sound Effects"
        ]

        self.genres = GENRES

    def get_genre_id(self, genre_str: str) -> int:
        """Converts a string genre from the JSON metadata into an integer ID."""
        if not genre_str:
            return self.genres.index("unknown")
        
        g = str(genre_str).lower().strip()
        if g in self.genres:
            return self.genres.index(g)
        return self.genres.index("unknown")

    def read_midi(self, file_path: str) -> MidiDict:
        return MidiDict.from_midi(file_path)
    
    def save_midi(self, midi_dict: MidiDict, file_path: str):
        midi_dict.save(file_path)
    
    def midi_to_tokens(self, midi_dict: MidiDict) -> list:
        return self.tokenizer.tokenize(midi_dict)
    
    def tokens_to_midi(self, tokens: list) -> MidiDict:
        return self.tokenizer.detokenize(tokens).to_midi()
    
    def tokens_to_tensor(self, tokens: list) -> dict[str, torch.Tensor]:
        token_types = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
        token_type_lists = {t: [] for t in token_types}

        for compound_event in tokens:
            if not isinstance(compound_event, tuple) or len(compound_event) != len(token_types):
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
                event.append(id_to_tok[tok_id])
            compound_tokens.append(tuple(event))
        return compound_tokens

    def get_instrument_multihot(self, tokens: list) -> torch.Tensor:
        active_instruments = set()
        for event in tokens:
            if isinstance(event, tuple) and len(event) > 0 and isinstance(event[0], tuple):
                inst_name = event[0][1] 
                if inst_name in self.INSTRUMENT_CLASSES:
                    active_instruments.add(inst_name)
        
        multi_hot = torch.zeros(len(self.INSTRUMENT_CLASSES), dtype=torch.float32)
        for i, cls_name in enumerate(self.INSTRUMENT_CLASSES):
            if cls_name in active_instruments:
                multi_hot[i] = 1.0
        return multi_hot

    def format_variable_sequence(self, tokens: list, target_length: int, pad_id: int = 2) -> torch.Tensor:
        """
        Converts a list of compound tokens to a padded/truncated tensor of exact target_length.
        """
        if not tokens:
            return torch.full((target_length, 5), pad_id, dtype=torch.long)

        tensor_dict = self.tokens_to_tensor(tokens)
        current_len = tensor_dict['pitch'].shape[0]
        final_tensor = torch.full((target_length, 5), pad_id, dtype=torch.long)
        
        valid_len = min(current_len, target_length)
        if valid_len > 0:
            final_tensor[:valid_len, 0] = tensor_dict['instrument'][:valid_len]
            final_tensor[:valid_len, 1] = tensor_dict['pitch'][:valid_len]
            final_tensor[:valid_len, 2] = tensor_dict['velocity'][:valid_len]
            final_tensor[:valid_len, 3] = tensor_dict['onset'][:valid_len]
            final_tensor[:valid_len, 4] = tensor_dict['duration'][:valid_len]
            
        return final_tensor

    # --- INFERENCE STYLE TRANSFER UTILITIES ---
    
    def skyline_groundline(self, tokens: list, algorithm: str) -> list:
        """Keeps only the highest pitch note from each chord (skyline) for inference prompts."""
        blank_tuple = ('<BLANK>', '<BLANK>', '<BLANK>', '<BLANK>', '<BLANK>')
        notes = []
        
        for i, t in enumerate(tokens):
            if isinstance(t, tuple) and len(t) == 5 and isinstance(t[0], tuple) and t[0][0] == 'instrument':
                notes.append({
                    "pitch": t[1][1],
                    "onset": t[3][1] if t[3][1] is not None else 0,
                    "orig_idx": i
                })

        if not notes: return list(tokens)

        chords = []
        i, k, WINDOW_MS = 0, len(notes), 50
        
        while i < k:
            start = notes[i]["onset"]
            chord = [notes[i]]
            j = i + 1
            while j < k and abs(notes[j]["onset"] - start) <= WINDOW_MS:
                chord.append(notes[j])
                j += 1
            chords.append(chord)
            i = j

        melody_pitch_indices = set()
        for chord in chords:
            max_pitch = max(n["pitch"] for n in chord)
            candidates = [n for n in chord if n["pitch"] == max_pitch]
            chosen = min(candidates, key=lambda n: n["orig_idx"])
            melody_pitch_indices.add(chosen["orig_idx"])

        out_tokens = list(tokens)
        for note in notes:
            idx = note["orig_idx"]
            orig_t = out_tokens[idx]
            
            if algorithm == "skyline" and idx not in melody_pitch_indices:
                out_tokens[idx] = blank_tuple
            elif algorithm == "groundline" and idx in melody_pitch_indices:
                out_tokens[idx] = blank_tuple
            else:
                out_tokens[idx] = ('<BLANK>', orig_t[1], orig_t[2], orig_t[3], orig_t[4])

        return out_tokens

    def extract_rhythm(self, tokens: list, ratio: float = 1.0) -> list:
        """Leaves onset and duration intact, masks pitch/vel/inst. Used for inference prep."""
        augmented_tokens = copy.deepcopy(tokens)
        note_indices = [
            i for i, event in enumerate(augmented_tokens)
            if isinstance(event, tuple) and len(event) == 5 and isinstance(event[0], tuple) and event[0][0] == 'instrument'
        ]
        mask_notes = random.sample(note_indices, int(len(note_indices) * ratio))

        for i in mask_notes:
            event_list = list(augmented_tokens[i])
            for j, field in enumerate(event_list):
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