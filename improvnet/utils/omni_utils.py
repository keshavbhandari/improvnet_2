import random
import copy
import os
import json
import torch
from improvnet.tokenizer.midi import MidiDict
from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.model.omni_config import GENRES


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
    
    def tokens_to_tensor(self, tokens: list) -> torch.Tensor:
        """Converts a flattened list of mixed strings/tuples directly into a 1D tensor of IDs."""
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
        """Shifts pitches. Safely handles 2-element or 3-element tuples, skipping drums, onsets, and durations!"""
        semitone_shift = random.randint(-7, 7)
        augmented_tokens = copy.deepcopy(tokens)
        for i, event in enumerate(augmented_tokens):
            if isinstance(event, tuple) and len(event) in (2, 3) and isinstance(event[1], int):
                if event[0] in ('onset', 'dur', 'prefix'):
                    continue
                    
                if "Drum" in str(event[0]) or "Percuss" in str(event[0]):
                    continue
                    
                new_pitch = max(0, min(127, event[1] + semitone_shift))
                if len(event) == 3:
                    augmented_tokens[i] = (event[0], new_pitch, event[2])
                else:
                    augmented_tokens[i] = (event[0], new_pitch)
        return augmented_tokens

    def get_instrument_multihot(self, tokens: list) -> torch.Tensor:
        active_instruments = set()
        for event in tokens:
            if isinstance(event, tuple) and len(event) in (2, 3):
                inst_name = event[0]
                if inst_name in self.INSTRUMENT_CLASSES:
                    active_instruments.add(inst_name)
        
        multi_hot = torch.zeros(len(self.INSTRUMENT_CLASSES), dtype=torch.float32)
        for i, cls_name in enumerate(self.INSTRUMENT_CLASSES):
            if cls_name in active_instruments:
                multi_hot[i] = 1.0
        return multi_hot

    def apply_targeted_corruption(self, ids_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies intelligent, localized noise to tokens rather than pure random vocabulary sampling.
        Preserves the syntactical structure while introducing harmonic and rhythmic errors.
        """
        tokens = self.tensor_to_tokens(ids_tensor)
        new_ids = []
        
        for idx, tok in enumerate(tokens):
            corrupted_tok = tok
            
            # Special strings like <S>, <T>, <PAD> skip corruption naturally by failing the tuple check
            if isinstance(tok, tuple):
                if tok[0] == 'onset':
                    # Shift by -100 to +100 in quantized steps of 10
                    shift = random.randint(-10, 10) * 10
                    new_val = max(0, min(4990, tok[1] + shift))
                    corrupted_tok = ('onset', new_val)
                    
                elif tok[0] == 'dur':
                    # Shift by -100 to +100 in quantized steps of 10 (min duration is usually > 0)
                    shift = random.randint(-10, 10) * 10
                    new_val = max(10, min(5000, tok[1] + shift)) 
                    corrupted_tok = ('dur', new_val)
                    
                elif len(tok) == 3:
                    # Standard Note: Keep instrument, randomize pitch and velocity
                    inst = tok[0]
                    pitch = random.randint(0, 127)
                    vel = random.randint(1, 12) * 10 # Quantized to intervals of 10 (10 to 120)
                    corrupted_tok = (inst, pitch, vel)
                    
                elif len(tok) == 2:
                    # Drum Note: Keep instrument, randomize pitch mapping
                    inst = tok[0]
                    pitch = random.randint(0, 127)
                    corrupted_tok = (inst, pitch)
            
            # Map back to ID. If quantization math resulted in an out-of-bounds token, fallback to original ID safely.
            if corrupted_tok in self.tokenizer.tok_to_id:
                new_ids.append(self.tokenizer.tok_to_id[corrupted_tok])
            else:
                new_ids.append(ids_tensor[idx].item())
                
        return torch.tensor(new_ids, dtype=torch.long, device=ids_tensor.device)