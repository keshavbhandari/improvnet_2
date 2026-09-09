import random
import copy
import os
import json
import torch
from improvnet.tokenizer.midi import MidiDict
from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.model.ar_config import GENRES


def read_jsonl_files(data_dirs, split="train"):
    files = []
    for file in data_dirs:
        if os.path.exists(file):
            with open(file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get("split", "train") == split:
                        files.append(data)
        else:
            print(f"Warning: {file} does not exist. Skipping.")
    return files


class ProcessData:
    def __init__(self):
        self.tokenizer = AbsTokenizer()
        self.genres = GENRES
        
        self.INSTRUMENT_CLASSES = [
            "Acoustic Piano", "Electric Piano", "Chromatic Percussion", "Organ", 
            "Acoustic Guitar", "Clean Electric Guitar", "Distorted Electric Guitar", 
            "Acoustic Bass", "Electric Bass", "Violin", "Viola", "Cello", "Contrabass", 
            "Orchestral Harp", "Timpani", "String Ensemble", "Synth Strings", 
            "Choir and Voice", "Orchestra Hit", "Trumpet", "Trombone", "Tuba", 
            "French Horn", "Brass Section", "Soprano/Alto Sax", "Tenor Sax", 
            "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet", "Piccolo", 
            "Flute", "Pipe", "Synth Lead", "Synth Pad", "Synth Effect", "Ethnic", 
            "Percussive", "Sound Effects", "drum"
        ]

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

    def _is_instrument_name(self, value) -> bool:
        return isinstance(value, str) and value in self.INSTRUMENT_CLASSES

    def _append_split_note(
        self,
        tokens: list,
        instrument: str,
        pitch: int,
        velocity: int | None,
        onset: int | None = None,
        duration: int | None = None,
    ) -> None:
        if velocity is None:
            velocity = self.tokenizer.config["drum_velocity"]
        tokens.append(("instrument", instrument))
        tokens.append(("note", pitch, velocity))
        if onset is not None:
            tokens.append(("onset", onset))
        if duration is not None:
            tokens.append(("dur", duration))

    def normalize_token_sequence(self, tokens: list) -> list:
        normalized = []
        for event in tokens:
            if not isinstance(event, tuple):
                normalized.append(event)
                continue

            tok_type = event[0] if len(event) > 0 else None
            if tok_type in ("instrument", "note"):
                normalized.append(event)
            elif tok_type == "drum" and len(event) >= 2 and isinstance(event[1], int):
                self._append_split_note(
                    normalized,
                    "drum",
                    event[1],
                    self.tokenizer.config["drum_velocity"],
                )
            elif self._is_instrument_name(tok_type) and len(event) >= 3:
                self._append_split_note(normalized, tok_type, event[1], event[2])
            else:
                normalized.append(event)
        return normalized

    def serialized_tokens_to_tokens(self, tokens_raw: list) -> list:
        tokens = []
        for event in tokens_raw:
            if (
                isinstance(event, list)
                and len(event) > 0
                and all(x == event[0] for x in event)
                and isinstance(event[0], str)
            ):
                tokens.append(event[0])
            elif (
                isinstance(event, list)
                and len(event) == 5
                and isinstance(event[0], list)
                and len(event[0]) == 2
            ):
                inst_val = event[0][1]
                pitch_val = event[1][1]
                vel_val = event[2][1]
                onset_val = event[3][1]
                dur_val = event[4][1]

                if inst_val in ('<P>', '<BLANK>', '<MASK>', '<S>', '<E>', '<T>'):
                    if inst_val not in ('<P>', '<BLANK>'):
                        tokens.append(inst_val)
                else:
                    self._append_split_note(
                        tokens,
                        inst_val,
                        pitch_val,
                        vel_val,
                        onset=onset_val,
                        duration=dur_val,
                    )
            else:
                tokens.append(tuple(event) if isinstance(event, list) else event)
        return self.normalize_token_sequence(tokens)
    
    def tokens_to_tensor(self, tokens: list) -> torch.Tensor:
        """Converts a flattened list of mixed strings/tuples directly into a 1D tensor of IDs."""
        tokens = self.normalize_token_sequence(tokens)
        ids = []
        for tok in tokens:
            if tok in self.tokenizer.tok_to_id:
                ids.append(self.tokenizer.tok_to_id[tok])
            else:
                raise KeyError(f"Token {tok} not found in vocab")
        return torch.tensor(ids, dtype=torch.long)
    
    def tensor_to_tokens(self, token_tensor: torch.Tensor) -> list:
        """Converts a 1D tensor of IDs back into a list of string/tuple tokens."""
        return [self.tokenizer.id_to_tok[idx.item()] for idx in token_tensor]

    def format_variable_sequence(self, tokens: list, target_length: int, pad_id: int = 2) -> torch.Tensor:
        """
        Converts a list of 1D tokens to a padded/truncated 1D tensor of exact target_length.
        """
        if not tokens:
            return torch.full((target_length,), pad_id, dtype=torch.long)

        tensor_seq = self.tokens_to_tensor(tokens)
        valid_len = min(tensor_seq.shape[0], target_length)
        
        final_tensor = torch.full((target_length,), pad_id, dtype=torch.long)
        if valid_len > 0:
            final_tensor[:valid_len] = tensor_seq[:valid_len]
            
        return final_tensor

    def pitch_augmentation(self, tokens: list) -> list:
        """Shifts note pitches while skipping drums."""
        semitone_shift = random.randint(-7, 7)
        augmented_tokens = copy.deepcopy(self.normalize_token_sequence(tokens))
        current_instrument = None
        for i, event in enumerate(augmented_tokens):
            if isinstance(event, tuple) and event[0] == "instrument":
                current_instrument = event[1]
                continue
            if isinstance(event, tuple) and event[0] == "note" and isinstance(event[1], int):
                if current_instrument == "drum" or "Drum" in str(current_instrument) or "Percuss" in str(current_instrument):
                    continue
                    
                new_pitch = max(0, min(127, event[1] + semitone_shift))
                augmented_tokens[i] = ("note", new_pitch, event[2])
        return augmented_tokens

    def get_instrument_multihot(self, tokens: list) -> torch.Tensor:
        active_instruments = set()
        for event in tokens:
            if isinstance(event, tuple) and event[0] == "instrument":
                inst_name = event[1]
                if inst_name in self.INSTRUMENT_CLASSES:
                    active_instruments.add(inst_name)
            elif isinstance(event, tuple) and len(event) in (2, 3):
                inst_name = event[0]
                if inst_name in self.INSTRUMENT_CLASSES:
                    active_instruments.add(inst_name)
        
        multi_hot = torch.zeros(len(self.INSTRUMENT_CLASSES), dtype=torch.float32)
        for i, cls_name in enumerate(self.INSTRUMENT_CLASSES):
            if cls_name in active_instruments:
                multi_hot[i] = 1.0
        return multi_hot

    # --- INFERENCE STYLE TRANSFER UTILITIES ---
    def extract_rhythm(self, tokens: list, ratio: float = 1.0) -> list:
        """Leaves onset and duration intact, masks note tuples using <MASK>."""
        augmented_tokens = copy.deepcopy(tokens)
        note_indices = [
            i for i, event in enumerate(augmented_tokens)
            if isinstance(event, tuple) and event[0] == "note"
        ]
        mask_notes = random.sample(note_indices, int(len(note_indices) * ratio))

        for i in mask_notes:
            augmented_tokens[i] = '<MASK>'

        return augmented_tokens
