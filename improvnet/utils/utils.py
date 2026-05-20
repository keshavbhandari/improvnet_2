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

    # Define the canonical order of the 40 instrument classes for the multi-hot vector
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
    
    # def get_random_segment_from_data(self, tokens: list, segment_length: int) -> list:
    #     if len(tokens) <= segment_length:
    #         return tokens
    #     max_start = len(tokens) - segment_length
    #     start = random.randint(0, max_start)
    #     return tokens[start : start + segment_length]
    
    def get_aligned_random_segment(self, tokens: list, target_len: int) -> list:
        """
        Slices a segment of tokens, ensuring the slice always starts at the 
        beginning of a 5-second window (onset == 0 or reset token).
        """
        total_len = len(tokens)
        if total_len <= target_len:
            return tokens
            
        # 1. Find all valid starting indices where a new 5-second window begins.
        # Adjust the condition based on how your <T> tokens or onsets are formatted.
        valid_start_indices = []
        for i, event in enumerate(tokens):
            # Assuming event is a tuple of tuples: (('instrument', '...'), ..., ('onset', val))
            # We look for the onset value.
            for attr in event:
                if attr[0] == 'onset' and attr[1] == 0:
                    valid_start_indices.append(i)
                    break
        
        # Fallback to pure random if no 0-onsets are found (rare, but safe)
        if not valid_start_indices:
            start_idx = random.randint(0, total_len - target_len)
            return tokens[start_idx : start_idx + target_len]
            
        # 2. Filter out indices that are too close to the end of the song
        valid_start_indices = [idx for idx in valid_start_indices if idx <= total_len - target_len]
        
        # If the song is long but we only found windows near the very end, fallback to the last valid window
        if not valid_start_indices:
            start_idx = total_len - target_len
            return tokens[start_idx : start_idx + target_len]
            
        # 3. Pick a random, aligned starting point!
        start_idx = random.choice(valid_start_indices)
        
        return tokens[start_idx : start_idx + target_len]
    
    def format_sequence(
        self, 
        token_tensors: dict[str, torch.Tensor], 
        seq_len: int
    ) -> torch.Tensor:
        """
        Pads or truncates sequences to the exact seq_len and stacks them.
        Returns a tensor of shape [seq_len, 5].
        """
        token_types = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
        current_len = len(token_tensors['instrument'])
        
        stacked_tensors = []
        for token_type in token_types:
            tensor = token_tensors[token_type]
            
            # Truncate if longer than seq_len
            if current_len > seq_len:
                tensor = tensor[:seq_len]
            
            # Pad if shorter than seq_len
            elif current_len < seq_len:
                # Retrieve the specific padding ID for this attribute vocabulary
                tok_to_id = getattr(self.tokenizer, f"tok_to_id_{token_type}")
                pad_id = tok_to_id.get('<P>')
                if pad_id is None:
                    raise KeyError(f"Padding token '<P>' not found in vocab for '{token_type}'")
                
                padding = torch.full((seq_len - current_len,), pad_id, dtype=torch.long)
                tensor = torch.cat([tensor, padding])
                
            stacked_tensors.append(tensor)

        # Stack into shape: [seq_len, 5]
        formatted_tensor = torch.stack(stacked_tensors, dim=-1)
        return formatted_tensor
    
    def get_instrument_multihot(self, tokens: list) -> torch.Tensor:
        """
        Parses compound tokens to find active instruments and returns a 40-dim multi-hot tensor.
        Assumes token structure: (('instrument', 'Oboe'), ('pitch', 74), ...)
        """
        active_instruments = set()
        for event in tokens:
            # Extract the string value from the first attribute tuple
            inst_name = event[0][1] 
            
            # Map the specific GM instrument back to its broader category if necessary
            # (Assuming inst_name matches the keys in your INSTRUMENT_CLASSES)
            if inst_name in self.INSTRUMENT_CLASSES:
                active_instruments.add(inst_name)
        
        # Build the 40-dim tensor
        multi_hot = torch.zeros(len(self.INSTRUMENT_CLASSES), dtype=torch.float32)
        for i, cls_name in enumerate(self.INSTRUMENT_CLASSES):
            if cls_name in active_instruments:
                multi_hot[i] = 1.0
                
        return multi_hot
    
    def extract_conditioning_segments(
        self, 
        tokens: list, 
        min_notes: int = 4, 
        max_notes: int = 128
    ) -> dict:
        """
        Extracts independent random segments from the original piece for conditioning.
        Applies specific masking/augmentations to each segment.
        """
        total_len = len(tokens)
        
        def _get_random_slice():
            # Safely handle extremely short sequences
            if total_len == 0:
                return []
                
            # Ensure the minimum is never larger than the sequence itself
            actual_min = min(min_notes, total_len)
            actual_max = min(max_notes, total_len)
            
            seg_len = random.randint(actual_min, actual_max)
            start_idx = random.randint(0, total_len - seg_len)
            return tokens[start_idx : start_idx + seg_len]

        # 1. Melody Segment (Skyline)
        melody_slice = _get_random_slice()
        melody_cond = self.skyline_groundline(melody_slice, algorithm="skyline")

        # 2. Harmony Segment (Groundline + Pitch Augmentation)
        harmony_slice = _get_random_slice()
        harmony_base = self.skyline_groundline(harmony_slice, algorithm="groundline")
        harmony_cond = self.pitch_augmentation(harmony_base)

        # 3. Rhythm Segment (Extract Rhythm)
        rhythm_slice = _get_random_slice()
        rhythm_cond = self.extract_rhythm(rhythm_slice, ratio=1.0)
        
        return {
            "melody": melody_cond,
            "harmony": harmony_cond,
            "rhythm": rhythm_cond
        }