import random
import copy
import os
import json
import torch
from improvnet.tokenizer.midi import MidiDict
from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.model.config import GENRES


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