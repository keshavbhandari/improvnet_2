import argparse
import math
import random
import os
import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from improvnet.model.omni_config import *
from improvnet.utils.omni_utils import ProcessData

SEQ_LEN = 256 
BLOCK_SIZE = 64 
PROMPT_MAX = 128 

# ---------------------------------------------------------------------------
# EXACT COPY OF OMNI TRAINING DATALOADER
# ---------------------------------------------------------------------------
class OmniDataset(Dataset):
    def __init__(self, jsonl_files: list[str], processor: ProcessData, augment: bool = True):
        self.jsonl_files = jsonl_files
        self.processor = processor
        self.augment = augment
        
        self.p_u = 0.15 # Max uniform noise ratio
        
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
        """Converts lists to tuples, properly flattening old 5D JSONL formats to the 1D Time Machine structure."""
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
                
                if inst_val in ('<P>', '<BLANK>', '<MASK>', '<S>', '<E>', '<T>'):
                    if inst_val not in ('<P>', '<BLANK>'):
                        tokens.append(inst_val)
                else:
                    # Break the 5D compound array into 3 sequential tokens!
                    # Handle drums as 2-tuples (Instrument, Pitch) without velocity
                    if inst_val == 'drum':
                        tokens.append((inst_val, pitch_val))
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
            print("\n" + "="*90)
            print(">>> DIAGNOSTIC PROBE 1: RAW JSONL EXTRACTION (First 15 elements)")
            print(tokens_raw[:15])
            print("="*90)
            
        tokens = self._lists_to_tuples(tokens_raw)
        
        if self._print_debug_once:
            print("\n" + "="*90)
            print(">>> DIAGNOSTIC PROBE 2: AFTER _lists_to_tuples() (First 15 elements)")
            print(tokens[:15])
            print("="*90 + "\n")
            self._print_debug_once = False 
        # ---------------------------
        
        genre_str = entry.get("genre", "unknown")
        genre_id = torch.tensor(self.processor.get_genre_id(genre_str), dtype=torch.long)
        
        if self.augment:
            tokens = self.processor.pitch_augmentation(tokens)

        if len(tokens) == 0: tokens = ['<S>']
        
        num_blocks = math.ceil(len(tokens) / BLOCK_SIZE)
        if num_blocks == 0: num_blocks = 1
        b = random.randint(0, num_blocks - 1)
        
        prefix_start = max(0, b * BLOCK_SIZE - PROMPT_MAX)
        prefix_tokens = tokens[prefix_start : b * BLOCK_SIZE]
        target_tokens = tokens[b * BLOCK_SIZE : b * BLOCK_SIZE + BLOCK_SIZE]
        
        multi_hot = self.processor.get_instrument_multihot(target_tokens)
        
        mode = random.choice([0, 1])          # 0: STRICT, 1: EDIT
        length_ctrl = random.choice([0, 1])   # 0: FIXED, 1: ELASTIC
        
        if random.random() < 0.2:
            active_insts = set(tok[0] for tok in target_tokens if isinstance(tok, tuple) and len(tok) in (2,3) and tok[0] in self.processor.INSTRUMENT_CLASSES)
            if active_insts:
                strip_inst = random.choice(list(active_insts))
                new_target = []
                skip = 0
                for tok in target_tokens:
                    if skip > 0:
                        skip -= 1
                        continue
                    if isinstance(tok, tuple) and len(tok) in (2,3) and tok[0] == strip_inst:
                        skip = 2 
                    else:
                        new_target.append(tok)
                target_tokens = new_target
                
        if length_ctrl == 1:
            # Weighted chunk insertion to match grammatical structure
            num_insertions = random.randint(0, int(len(target_tokens) * 0.08) + 1)
            for _ in range(num_insertions):
                # 80% chance of 3 blanks (full note gap), 20% chance of 1 blank
                chunk_size = 3 if random.random() < 0.8 else 1
                idx = random.randint(0, len(target_tokens))
                for _ in range(chunk_size):
                    target_tokens.insert(idx, '<BLANK>')

        prefix_tensor = self.processor.format_variable_sequence(prefix_tokens, PROMPT_MAX, pad_id=PAD_ID)
        target_tensor = self.processor.format_variable_sequence(target_tokens, BLOCK_SIZE, pad_id=PAD_ID)

        return {
            "prefix": prefix_tensor,
            "target": target_tensor,
            "multi_hot": multi_hot,
            "mode": torch.tensor(mode, dtype=torch.long),
            "length_ctrl": torch.tensor(length_ctrl, dtype=torch.long),
            "genre": genre_id
        }

    def collate_fn(self, batch):
        input_batches = []
        target_batches = []
        ts_batches = []
        weight_batches = []
        
        genres, modes, length_ctrls, multi_hots = [], [], [], []

        sep_t = torch.tensor([SEP_ID], dtype=torch.long)
        pad_t = torch.tensor([PAD_ID], dtype=torch.long)
        zero_t = torch.tensor([0.0], dtype=torch.float32)

        for item in batch:
            prefix = item["prefix"]
            target = item["target"]
            mode = item["mode"].item()
            
            s = random.randint(1, DIFFUSION_STEPS)
            
            input_chunks = [prefix, sep_t]
            target_chunks = [torch.full_like(prefix, PAD_ID), pad_t]
            ts_chunks = [torch.zeros_like(prefix, dtype=torch.float32), zero_t]
            wt_chunks = [torch.zeros_like(prefix, dtype=torch.float32), zero_t]
            
            valid_mask = (target != PAD_ID)
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            L_valid = len(valid_indices)
            
            for k in range(4):
                t_val = max(0.0, (s - k) / float(DIFFUSION_STEPS))
                draft_input = target.clone()
                draft_wt = torch.zeros_like(target, dtype=torch.float32)
                
                if L_valid > 0:
                    r = torch.rand(L_valid)
                    
                    if mode == 1: 
                        if t_val <= 0.0:
                            p_clean, p_mask, p_uniform = 1.0, 0.0, 0.0
                            w_clean, w_mask, w_uniform = 1.0, 0.0, 0.0
                        elif t_val >= 1.0:
                            p_clean, p_mask, p_uniform = 0.0, 1.0, 0.0
                            w_clean, w_mask, w_uniform = 0.0, 2.0, 0.0
                        else:
                            B_const = 2.0 * (self.p_u / (1.0 - self.p_u))
                            c_t = B_const * math.sqrt(t_val * (1.0 - t_val))
                            C_t = 1.0 + c_t
                            
                            p_clean = (1.0 - t_val) / C_t
                            p_mask = t_val / C_t
                            p_uniform = c_t / C_t
                            
                            e_lambda = math.sqrt((c_t + t_val) / max(1.0 - t_val, 1e-5))
                            w_mask = 2.0
                            w_uniform = 1.0
                            w_clean = max(0.05, min(1.0, (B_const / VOCAB_SIZE) * e_lambda))
                            
                        do_mask = (r >= p_clean) & (r < p_clean + p_mask)
                        do_uniform = (r >= p_clean + p_mask)
                        do_clean = ~(do_mask | do_uniform)
                        
                        idx_mask = valid_indices[do_mask]
                        idx_uniform = valid_indices[do_uniform]
                        idx_clean = valid_indices[do_clean]
                        
                        draft_input[idx_mask] = MASK_ID
                        
                        if len(idx_uniform) > 0:
                            r_noise = torch.rand(len(idx_uniform))
                            is_pure_random = r_noise < 0.3
                            is_targeted = ~is_pure_random
                            
                            idx_pure = idx_uniform[is_pure_random]
                            idx_targ = idx_uniform[is_targeted]
                            
                            if len(idx_targ) > 0:
                                tokens_to_corrupt = draft_input[idx_targ]
                                draft_input[idx_targ] = self.processor.apply_targeted_corruption(tokens_to_corrupt)
                                
                            if len(idx_pure) > 0:
                                # Safe boundary for random vocab token
                                draft_input[idx_pure] = torch.randint(11, VOCAB_SIZE, (len(idx_pure),))
                            
                        draft_wt[idx_clean] = w_clean
                        draft_wt[idx_mask] = w_mask
                        draft_wt[idx_uniform] = w_uniform
                        
                    else: 
                        p_mask_strict = t_val
                        do_mask = r < p_mask_strict
                        do_clean = ~do_mask
                        
                        idx_mask = valid_indices[do_mask]
                        idx_clean = valid_indices[do_clean]
                        
                        draft_input[idx_mask] = MASK_ID
                        draft_wt[idx_clean] = 0.1 
                        draft_wt[idx_mask] = 1.0
                
                input_chunks.extend([draft_input, sep_t])
                target_chunks.extend([target, pad_t]) 
                ts_chunks.extend([torch.full_like(target, t_val, dtype=torch.float32), zero_t])
                wt_chunks.extend([draft_wt, zero_t])
                
            input_batches.append(torch.cat(input_chunks))
            target_batches.append(torch.cat(target_chunks))
            ts_batches.append(torch.cat(ts_chunks))
            weight_batches.append(torch.cat(wt_chunks))
            
            genres.append(item["genre"])
            modes.append(item["mode"])
            length_ctrls.append(item["length_ctrl"])
            multi_hots.append(item["multi_hot"])

        return {
            "input": torch.stack(input_batches),
            "target": torch.stack(target_batches),
            "timestep": torch.stack(ts_batches),
            "weights": torch.stack(weight_batches),
            "genre": torch.stack(genres),
            "mode": torch.stack(modes),
            "length_ctrl": torch.stack(length_ctrls),
            "multi_hot": torch.stack(multi_hots),
            "causal_prefix_len": PROMPT_MAX + 1, 
            "draft_size": BLOCK_SIZE + 1         
        }

def build_dataloader(jsonl_files: list[str], batch_size: int) -> DataLoader:
    processor = ProcessData()
    dataset = OmniDataset(jsonl_files=jsonl_files, processor=processor, augment=False)
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
        wt_tensor = batch["weights"][0]
        
        genre_id = batch["genre"][0].item()
        mode_id = batch["mode"][0].item()
        len_id = batch["length_ctrl"][0].item()
        mh_tensor = batch["multi_hot"][0]
        
        # Decode the control tokens
        genre_str = processor.genres[genre_id] if genre_id < len(processor.genres) else "unknown"
        mode_str = "EDIT (GIDD Corruptions)" if mode_id == 1 else "STRICT (Masks Only)"
        len_str = "ELASTIC (With <BLANK>s)" if len_id == 1 else "FIXED"
        
        active_insts = [processor.INSTRUMENT_CLASSES[i] for i, val in enumerate(mh_tensor) if val == 1.0]
        mh_str = ", ".join(active_insts) if active_insts else "None"
        
        print(f"\n==========================================================================================")
        print(f"OMNI-CADDI BATCH DEBUG")
        print(f"==========================================================================================")
        print(f"Genre:         <GENRE: {genre_str.upper()}>")
        print(f"Mode:          <MODE: {mode_str}>")
        print(f"Length Ctrl:   <LEN: {len_str}>")
        print(f"Multi-Hot:     [{mh_str}]")
        print(f"Flash Attn:    Prefix Causal Length: {batch['causal_prefix_len']}, Draft Chunk Size: {batch['draft_size']}")
        print(f"==========================================================================================\n")
        
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

        print(f"{'IDX':<5} | {'TIME':<6} | {'WEIGHT':<6} | {'INPUT TOKEN (What model sees)':<38} | {'TARGET TOKEN (Loss calculation)':<38}")
        print("-" * 105)
        
        current_chunk = "PREFIX (Sliding Context Window)"
        print(f"--- ENTERING {current_chunk} ---")
        
        for i in range(len(input_tensor)):
            inp_tok = safe_decode(input_tensor[i])
            tgt_tok = safe_decode(target_tensor[i])
            ts_val = ts_tensor[i].item()
            wt_val = wt_tensor[i].item()
            
            if inp_tok == "<SEP>":
                next_ts = ts_tensor[i+1].item() if i+1 < len(ts_tensor) else 0.0
                current_chunk = f"DRAFT (t={next_ts:.3f})"
                print("-" * 105)
                print(f"--- ENTERING {current_chunk} ---")
                print("-" * 105)
                continue # Skip printing the SEP line itself to keep boundaries clean
                
            if inp_tok == "<PAD>" and tgt_tok == "<PAD>":
                # Print the first pad just to show where it stops, then break chunk
                if i == 0 or safe_decode(input_tensor[i-1]) != "<PAD>":
                    print(f"{i:<5} | {ts_val:<6.2f} | {wt_val:<6.2f} | {'<PAD> (End of block)':<38} | {'<PAD>':<38}")
                continue 
                
            # Formatting to highlight corrupted edits vs masks
            inp_str = f"** {inp_tok} **" if (inp_tok != tgt_tok and inp_tok not in ("<MASK>", "<PAD>")) else inp_tok
                
            print(f"{i:<5} | {ts_val:<6.3f} | {wt_val:<6.3f} | {inp_str:<38} | {tgt_tok:<38}")

        print("\n==========================================================================================")
        print("DEBUG COMPLETE. Exiting.")
        print("==========================================================================================\n")
        break 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run the Omni-CaDDi dataloader debug printout.")
    args = parser.parse_args()

    if args.debug:
        run_debug()
    else:
        print("Run with --debug flag to execute the dataloader inspection.")