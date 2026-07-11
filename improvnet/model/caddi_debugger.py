import argparse
import math
import random
import os
import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from improvnet.model.caddi_config import *
from improvnet.utils.utils import ProcessData

SEQ_LEN = 256 
BLOCK_SIZE = 32 
PROMPT_MAX = 128 

# ---------------------------------------------------------------------------
# EXACT COPY OF TRAINING DATALOADER
# ---------------------------------------------------------------------------
class CaDDiDataset(Dataset):
    def __init__(self, jsonl_files: list[str], processor: ProcessData, augment: bool = True):
        self.jsonl_files = jsonl_files
        self.processor = processor
        self.augment = augment
        self.seq_len = SEQ_LEN
        self.block_size = BLOCK_SIZE
        
        self.file_handles = {}
        self.global_indices = []
        
        for file_idx, jsonl_path in enumerate(self.jsonl_files):
            index_path = jsonl_path.replace('.jsonl', '_index.pkl')
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Missing index! Run build_index.py on {jsonl_path}")
            with open(index_path, 'rb') as f:
                offsets = pickle.load(f)
            for offset in offsets:
                self.global_indices.append((file_idx, offset))

        # Flag to trigger the debug print only on the very first batch
        self._print_debug_once = True 

    def __len__(self):
        return len(self.global_indices)

    def _get_file_handle(self, jsonl_path):
        if jsonl_path not in self.file_handles:
            self.file_handles[jsonl_path] = open(jsonl_path, 'rb')
        return self.file_handles[jsonl_path]

    def _lists_to_tuples(self, tokens_raw):
        tokens = []
        for event in tokens_raw:
            # 1. Collapse repeating special tokens (e.g., ['<S>', '<S>', '<S>', '<S>', '<S>'] -> '<S>')
            if isinstance(event, list) and len(event) > 0 and all(x == event[0] for x in event) and isinstance(event[0], str):
                tokens.append(event[0])
                
            # 2. Translate old 5D JSONL arrays into the new flattened 1D sequence
            elif isinstance(event, list) and len(event) == 5 and isinstance(event[0], list) and len(event[0]) == 2:
                inst_val = event[0][1]
                pitch_val = event[1][1]
                vel_val = event[2][1]
                onset_val = event[3][1]
                dur_val = event[4][1]
                
                if inst_val in ('<PAD>', '<BLANK>', '<MASK>', '<S>', '<E>', '<T>'):
                    if inst_val not in ('<PAD>', '<BLANK>'):
                        tokens.append(inst_val)
                else:
                    tokens.append((inst_val, pitch_val, vel_val))
                    tokens.append(('onset', onset_val))
                    tokens.append(('dur', dur_val))
                    
            # 3. Fallback for formats already in 1D
            else:
                if isinstance(event, list):
                    tokens.append(tuple(event))
                else:
                    tokens.append(event)
                    
        return tokens

    def __getitem__(self, idx):
        file_idx, offset = self.global_indices[idx]
        jsonl_path = self.jsonl_files[file_idx]
        
        f = self._get_file_handle(jsonl_path)
        f.seek(offset)
        line_bytes = f.readline()
        
        entry = json.loads(line_bytes.decode('utf-8'))
        tokens_raw = entry.get("tokens", [])
        
        # --- DATA PIPELINE PROBE ---
        if self._print_debug_once:
            print("\n" + "="*80)
            print(">>> DIAGNOSTIC PROBE 1: RAW JSONL EXTRACTION (First 15 elements)")
            print(tokens_raw[:15])
            print("="*80)
            
        tokens = self._lists_to_tuples(tokens_raw)
        
        if self._print_debug_once:
            print("\n" + "="*80)
            print(">>> DIAGNOSTIC PROBE 2: AFTER _lists_to_tuples() (First 15 elements)")
            print(tokens[:15])
            print("="*80 + "\n")
            self._print_debug_once = False 
        # ---------------------------
        
        genre_str = entry.get("genre", "unknown")
        genre_id = torch.tensor(self.processor.get_genre_id(genre_str), dtype=torch.long)
        
        if self.augment:
            tokens = self.processor.pitch_augmentation(tokens)

        actual_len = len(tokens)
        if actual_len == 0:
            tokens = ['<S>']
            actual_len = 1

        tensor_seq = self.processor.format_variable_sequence(tokens, actual_len, pad_id=PAD_ID)

        return {
            "tokens": tensor_seq[:actual_len],
            "actual_len": actual_len,
            "genre": genre_id
        }

    def collate_fn(self, batch):
        padded_inputs = []
        padded_targets = []
        timesteps = []
        genres = []
        
        max_unrolled_len = 0
        batch_data = []

        for item in batch:
            seq = item["tokens"]
            seq_len = seq.shape[0]

            num_blocks = math.ceil(seq_len / self.block_size)
            if num_blocks == 0: num_blocks = 1
            b = random.randint(0, num_blocks - 1)

            prefix_len = min(b * self.block_size, PROMPT_MAX)
            block_start = prefix_len
            block_end = min(prefix_len + self.block_size, seq_len)

            prefix_seq = seq[:prefix_len]
            target_block = seq[block_start:block_end]
            actual_L = target_block.shape[0]

            if actual_L == 0:
                target_block = torch.full((1,), PAD_ID, dtype=torch.long)
                actual_L = 1

            t_vals = sorted([random.uniform(0.05, 1.0) for _ in range(4)], reverse=True)

            input_chunks = [prefix_seq]
            target_chunks = [torch.full_like(prefix_seq, PAD_ID)] 
            ts_chunks = [torch.zeros(prefix_len)] 

            sep_input = torch.tensor([SEP_ID], dtype=torch.long)
            sep_target = torch.tensor([PAD_ID], dtype=torch.long)
            sep_ts = torch.tensor([0.0], dtype=torch.float32)

            for t in t_vals:
                corrupted = target_block.clone()
                draft_target = torch.full_like(target_block, PAD_ID)
                
                num_to_mask = int(t * actual_L)
                if num_to_mask > 0:
                    perm = torch.randperm(actual_L)[:num_to_mask]
                    corrupted[perm] = MASK_ID
                    draft_target[perm] = target_block[perm]

                input_chunks.append(sep_input)
                target_chunks.append(sep_target)
                ts_chunks.append(sep_ts)

                input_chunks.append(corrupted)
                target_chunks.append(draft_target)
                ts_chunks.append(torch.full((actual_L,), t))

            input_tensor = torch.cat(input_chunks)
            target_tensor = torch.cat(target_chunks)
            ts_tensor = torch.cat(ts_chunks)

            max_unrolled_len = max(max_unrolled_len, input_tensor.shape[0])

            batch_data.append({
                "input": input_tensor,
                "target": target_tensor,
                "ts": ts_tensor,
                "genre": item["genre"]
            })

        for data in batch_data:
            L = data["input"].shape[0]
            pad_len = max_unrolled_len - L

            if pad_len > 0:
                pad_input = torch.full((pad_len,), PAD_ID, dtype=torch.long)
                pad_target = torch.full((pad_len,), PAD_ID, dtype=torch.long)
                pad_ts = torch.zeros(pad_len)

                padded_inputs.append(torch.cat([data["input"], pad_input]))
                padded_targets.append(torch.cat([data["target"], pad_target]))
                timesteps.append(torch.cat([data["ts"], pad_ts]))
            else:
                padded_inputs.append(data["input"])
                padded_targets.append(data["target"])
                timesteps.append(data["ts"])
            genres.append(data["genre"])
            
        return {
            "input": torch.stack(padded_inputs),
            "target": torch.stack(padded_targets),
            "timestep": torch.stack(timesteps),
            "genre": torch.stack(genres)
        }

def build_dataloader(jsonl_files: list[str], batch_size: int) -> DataLoader:
    processor = ProcessData()
    dataset = CaDDiDataset(jsonl_files=jsonl_files, processor=processor, augment=False)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
        collate_fn=dataset.collate_fn
    )

# ---------------------------------------------------------------------------
# DEBUG LOGIC
# ---------------------------------------------------------------------------
def run_debug():
    print(f"Loading data from: {JSONL_FILES}")
    dataloader = build_dataloader(JSONL_FILES, batch_size=1)
    processor = ProcessData()

    print("Fetching one batch...")
    for batch in dataloader:
        input_tensor = batch["input"][0]
        target_tensor = batch["target"][0]
        ts_tensor = batch["timestep"][0]
        genre_id = batch["genre"][0].item()
        
        genre_str = processor.genres[genre_id] if genre_id < len(processor.genres) else "unknown"
        print(f"\n==================================================")
        print(f"BATCH DEBUG | GENRE: {genre_str}")
        print(f"==================================================\n")
        
        def safe_decode(tensor_id):
            tid = tensor_id.item()
            if tid == PAD_ID: return "<PAD>"
            if tid == MASK_ID: return "<MASK>"
            if tid == SEP_ID: return "<SEP>"
            if tid == BLANK_ID: return "<BLANK>"
            try:
                return str(processor.tokenizer.id_to_tok[tid])
            except KeyError:
                return f"[UNKNOWN ID: {tid}]"

        print(f"{'INDEX':<6} | {'TIMESTEP':<8} | {'INPUT TOKEN (What model sees)':<40} | {'TARGET TOKEN (Loss calculation)':<40}")
        print("-" * 105)
        
        current_chunk = "PREFIX"
        
        for i in range(len(input_tensor)):
            inp_tok = safe_decode(input_tensor[i])
            tgt_tok = safe_decode(target_tensor[i])
            ts_val = ts_tensor[i].item()
            
            if inp_tok == "<SEP>":
                current_chunk = f"DRAFT (t={ts_tensor[i+1].item():.2f})" if i+1 < len(ts_tensor) else "DRAFT"
                print("-" * 105)
                print(f"--- ENTERING {current_chunk} ---")
                print("-" * 105)
                
            if inp_tok == "<PAD>" and tgt_tok == "<PAD>":
                print(f"{i:<6} | {ts_val:<8.2f} | {'<PAD> (End of useful sequence)':<40} | {'<PAD>':<40}")
                break
                
            print(f"{i:<6} | {ts_val:<8.2f} | {inp_tok:<40} | {tgt_tok:<40}")

        print("\n==================================================")
        print("DEBUG COMPLETE. Exiting.")
        print("==================================================\n")
        break 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run the dataloader debug printout.")
    args = parser.parse_args()

    if args.debug:
        run_debug()
    else:
        print("Run with --debug flag to execute the dataloader inspection.")