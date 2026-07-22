import math
import random
import os
import json
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from improvnet.model.omni_config import *
from improvnet.model.omni_model import CaDDiModel
from improvnet.utils.omni_utils import ProcessData
import contextlib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _amp_dtype():
    if not torch.cuda.is_available(): return None
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16

AMP_DTYPE = _amp_dtype()


class OmniDataset(Dataset):
    def __init__(self, jsonl_files: list[str], processor: ProcessData, augment: bool = True):
        self.jsonl_files = jsonl_files
        self.processor = processor
        self.augment = augment
        
        # Max targeted noise ratio at t=0.5 (from GIDD paper)
        self.p_u = 0.15 
        
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
        tokens = self._lists_to_tuples(tokens_raw)
        
        genre_str = entry.get("genre", "unknown")
        genre_id = torch.tensor(self.processor.get_genre_id(genre_str), dtype=torch.long)
        
        if self.augment:
            tokens = self.processor.pitch_augmentation(tokens)

        if len(tokens) == 0: tokens = ['<S>']
        
        # 1. Block Partitioning (Yields 1, 2, 3, or 4 blocks max per 2048 seq)
        num_blocks = math.ceil(len(tokens) / BLOCK_SIZE)
        if num_blocks == 0: num_blocks = 1
        b = random.randint(0, num_blocks - 1)
        
        # Continuous Sliding Window for the Prefix
        prefix_start = max(0, b * BLOCK_SIZE - PROMPT_MAX)
        prefix_tokens = tokens[prefix_start : b * BLOCK_SIZE]
        target_tokens = tokens[b * BLOCK_SIZE : b * BLOCK_SIZE + BLOCK_SIZE]
        
        # 2. Extract TRUE Multi-Hot before we apply any destructive stripping
        multi_hot = self.processor.get_instrument_multihot(target_tokens)
        
        # 3. Sample Control Axis
        mode = random.choice([0, 1])          # 0: STRICT (Mask only), 1: EDIT (Targeted Noise)
        length_ctrl = random.choice([0, 1])   # 0: FIXED, 1: ELASTIC
        
        # 4. Instrument Stripping (Infilling Task) -> 20% Chance
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
                        skip = 2 # Strip the note, plus its subsequent onset and duration
                    else:
                        new_target.append(tok)
                target_tokens = new_target
                
        # 5. Elasticity (Insert <BLANK>s randomly)
        if length_ctrl == 1:
            # Weighted chunk insertion to match grammatical structure
            num_insertions = random.randint(0, int(len(target_tokens) * 0.08) + 1)
            for _ in range(num_insertions):
                # 80% chance of 3 blanks (full note gap), 20% chance of 1 blank
                chunk_size = 3 if random.random() < 0.8 else 1
                idx = random.randint(0, len(target_tokens))
                for _ in range(chunk_size):
                    target_tokens.insert(idx, '<BLANK>')

        # 6. Format to strict Uniform Tensors
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
        """Constructs the Bidirectional 'Smooth' Time Machine Trajectory with Pure GIDD Math."""
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
                        # --- TRUE GIDD DYNAMICS (<MODE: EDIT>) ---
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
                            
                            # GIDD Eq 20 Loss Weighting
                            e_lambda = math.sqrt((c_t + t_val) / max(1.0 - t_val, 1e-5))
                            w_mask = 2.0
                            w_uniform = 1.0
                            # Floor w_clean at 0.05 so massive 68k vocabularies still learn the identity function
                            w_clean = max(0.05, min(1.0, (B_const / VOCAB_SIZE) * e_lambda))
                            
                        do_mask = (r >= p_clean) & (r < p_clean + p_mask)
                        do_uniform = (r >= p_clean + p_mask)
                        do_clean = ~(do_mask | do_uniform)
                        
                        idx_mask = valid_indices[do_mask]
                        idx_uniform = valid_indices[do_uniform]
                        idx_clean = valid_indices[do_clean]
                        
                        draft_input[idx_mask] = MASK_ID
                        
                        if len(idx_uniform) > 0:
                            # Split the uniform noise: 70% Targeted Edit, 30% Pure Random Garbage
                            r_noise = torch.rand(len(idx_uniform))
                            is_pure_random = r_noise < 0.3
                            is_targeted = ~is_pure_random
                            
                            idx_pure = idx_uniform[is_pure_random]
                            idx_targ = idx_uniform[is_targeted]
                            
                            if len(idx_targ) > 0:
                                tokens_to_corrupt = draft_input[idx_targ]
                                draft_input[idx_targ] = self.processor.apply_targeted_corruption(tokens_to_corrupt)
                                
                            if len(idx_pure) > 0:
                                # Replace with completely random tokens from the vocab (avoiding special tokens 0-10)
                                draft_input[idx_pure] = torch.randint(11, VOCAB_SIZE, (len(idx_pure),))
                            
                        draft_wt[idx_clean] = w_clean
                        draft_wt[idx_mask] = w_mask
                        draft_wt[idx_uniform] = w_uniform
                        
                    else: 
                        # --- STANDARD MASKGIT DYNAMICS (<MODE: STRICT>) ---
                        p_mask_strict = t_val
                        do_mask = r < p_mask_strict
                        do_clean = ~do_mask
                        
                        idx_mask = valid_indices[do_mask]
                        idx_clean = valid_indices[do_clean]
                        
                        draft_input[idx_mask] = MASK_ID
                        
                        # Standard weighting (prioritize masked tokens, minor weight to clean tokens for identity)
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

def build_dataloader(jsonl_files: list[str], split: str, batch_size: int, augment: bool = False, num_workers: int = 4, distributed: bool = False) -> DataLoader:
    processor = ProcessData()
    dataset = OmniDataset(jsonl_files=jsonl_files, processor=processor, augment=augment)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(not distributed),
        sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=dataset.collate_fn
    )

# ---------------------------------------------------------------------------
# DDP & Checkpoint Utilities
# ---------------------------------------------------------------------------
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, scheduler, step, best_val_loss, is_best=False):
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_to_save = model.module if isinstance(model, DDP) else model
    checkpoint = {
        'step': step, 
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, os.path.join(SAVE_DIR, "latest_checkpoint.pt"))
    if is_best:
        torch.save(checkpoint, os.path.join(SAVE_DIR, "best_model.pt"))

def load_checkpoint(model, optimizer, device):
    checkpoint_path = os.path.join(SAVE_DIR, "latest_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        return 0, float('inf')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_to_load = model.module if isinstance(model, DDP) else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step'], checkpoint['best_val_loss']

def build_scheduler(optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps: return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(n_steps: int = N_STEPS):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (local_rank == 0)

    writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, "runs", RUN_NAME)) if is_main_process else None

    train_loader = build_dataloader(JSONL_FILES, "train", BATCH_SIZE, augment=True, distributed=True)
    val_loader = build_dataloader(JSONL_FILES, "validation", BATCH_SIZE, augment=False, distributed=True)

    if is_main_process: print("Initializing Omni-CaDDi AR Diffusion Model...")
    model = CaDDiModel().to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(AMP_DTYPE == torch.float16))

    update_step, best_val_loss = 0, float('inf')
    if RESUME_TRAINING:
        update_step, best_val_loss = load_checkpoint(model, optimizer, device)

    scheduler = build_scheduler(optimizer, WARMUP_STEPS, n_steps, last_epoch=update_step - 1 if update_step > 0 else -1)
    
    micro_step = update_step * ACCUM_STEPS 
    start_epoch = micro_step // len(train_loader)
    
    running_loss, log_steps = 0.0, 0
    model.train()
    epoch = start_epoch

    while update_step < n_steps:
        if hasattr(train_loader.sampler, 'set_epoch'): train_loader.sampler.set_epoch(epoch)
        
        for batch in train_loader:
            if update_step >= n_steps: break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            is_accumulating = (micro_step + 1) % ACCUM_STEPS != 0
            ddp_context = model.no_sync() if is_accumulating else contextlib.nullcontext()
            
            with ddp_context:
                with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                    
                    logits = model(
                        target=batch["input"],          
                        timestep=batch["timestep"],
                        genre=batch["genre"],
                        mode=batch["mode"],
                        length_ctrl=batch["length_ctrl"],
                        multi_hot=batch["multi_hot"],
                        causal_prefix_len=batch["causal_prefix_len"],
                        draft_size=batch["draft_size"]
                    )
                    
                    # Unreduced CrossEntropy: Evaluate ALL tokens, weight dynamically based on Time
                    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction='none')
                    
                    raw_loss = criterion(logits.view(-1, VOCAB_SIZE), batch["target"].view(-1))
                    weights = batch["weights"].view(-1)
                    
                    weighted_loss = raw_loss * weights
                    
                    # Calculate mean over valid tokens (where weight > 0)
                    valid_tokens_count = (weights > 0).sum().clamp(min=1.0)
                    loss = (weighted_loss.sum() / valid_tokens_count) / ACCUM_STEPS

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            running_loss += (loss.item() * ACCUM_STEPS)
            log_steps += 1
            micro_step += 1

            if not is_accumulating:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    
                scheduler.step()
                optimizer.zero_grad()
                update_step += 1

                if is_main_process and (update_step % LOG_EVERY == 0) and log_steps > 0:
                    avg_loss = running_loss / log_steps
                    print(f"  Step {update_step:>6} | Loss: {avg_loss:>6.4f}")
                    writer.add_scalar('Train/Omni_Diffusion_Loss', avg_loss, update_step)
                    writer.add_scalar('Hyperparameters/LR', optimizer.param_groups[0]['lr'], update_step)
                    running_loss, log_steps = 0.0, 0

                if update_step > 0 and update_step % VAL_EVERY == 0:
                    val_loss = evaluate_validation(model, val_loader, local_rank, device)
                    if is_main_process:
                        print(f"\n--- Validation Step {update_step} | Val Loss: {val_loss:.4f} ---")
                        writer.add_scalar('Validation/Omni_Diffusion_Loss', val_loss, update_step)
                        is_best = val_loss < best_val_loss
                        if is_best: best_val_loss = val_loss
                        save_checkpoint(model, optimizer, scheduler, update_step, best_val_loss, is_best)
        epoch += 1

    if is_main_process and writer: writer.close()
    cleanup_ddp()
    return model

@torch.no_grad()
def evaluate_validation(model, val_loader, local_rank, device, max_batches: int = 50):
    model.eval()
    total_loss, steps = 0.0, 0
    iterator = tqdm(val_loader, desc="Validation", total=min(len(val_loader), max_batches), leave=False) if local_rank == 0 else val_loader

    for batch in iterator:
        if steps >= max_batches: break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
            logits = model(
                target=batch["input"],
                timestep=batch["timestep"],
                genre=batch["genre"],
                mode=batch["mode"],
                length_ctrl=batch["length_ctrl"],
                multi_hot=batch["multi_hot"],
                causal_prefix_len=batch["causal_prefix_len"],
                draft_size=batch["draft_size"]
            )
            
            criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction='none')
            raw_loss = criterion(logits.view(-1, VOCAB_SIZE), batch["target"].view(-1))
            weights = batch["weights"].view(-1)
            
            weighted_loss = raw_loss * weights
            valid_tokens_count = (weights > 0).sum().clamp(min=1.0)
            loss = weighted_loss.sum() / valid_tokens_count
            
        total_loss += loss.item()
        steps += 1

    local_avg_loss = total_loss / max(1, steps)
    if torch.distributed.is_initialized():
        metrics = torch.tensor([local_avg_loss], device=device)
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
        global_avg_loss = (metrics / torch.distributed.get_world_size())[0].item()
    else:
        global_avg_loss = local_avg_loss

    model.train()
    return global_avg_loss

if __name__ == "__main__":
    model = train(n_steps=N_STEPS)