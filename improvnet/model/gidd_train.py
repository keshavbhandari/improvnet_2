import math
import random
import os
import json
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from improvnet.model.gidd_config import *
from improvnet.model.gidd_model import PrefixARModel
from improvnet.utils.gidd_utils import ProcessData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _amp_dtype():
    if not torch.cuda.is_available(): return None
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16

AMP_DTYPE = _amp_dtype()

# ---------------------------------------------------------------------------
# Data Loading for GIDD (Mask + Uniform Noise + Dynamic Weighting)
# ---------------------------------------------------------------------------

class GIDDDataset(Dataset):
    def __init__(self, jsonl_files: list[str], processor: ProcessData, augment: bool = True):
        self.jsonl_files = jsonl_files
        self.processor = processor
        self.augment = augment
        self.seq_len = SEQ_LEN
        self.block_size = BLOCK_SIZE
        
        # Set the maximum uniform noise ratio at t=0.5 (from paper)
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
        tokens = []
        for event in tokens_raw:
            if isinstance(event, list):
                tokens.append(tuple(tuple(f) if isinstance(f, list) else f for f in event))
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

        total_len = len(tokens)
        if total_len > self.seq_len:
            start_idx = random.randint(0, total_len - self.seq_len)
            tokens = tokens[start_idx : start_idx + self.seq_len]
        
        actual_len = len(tokens)
        if actual_len == 0:
            tokens = [('<S>', '<S>', '<S>', '<S>', '<S>')]
            actual_len = 1

        tensor_seq = self.processor.format_variable_sequence(tokens, actual_len, pad_id=PAD_ID)

        return {
            "tokens": tensor_seq[:actual_len],
            "actual_len": actual_len,
            "genre": genre_id
        }

    def collate_fn(self, batch):
        max_actual = max(item["actual_len"] for item in batch)
        
        num_blocks = math.ceil(max_actual / self.block_size)
        if num_blocks == 0: num_blocks = 1
        b = random.randint(0, num_blocks - 1)
        
        prefix_len = b * self.block_size
        block_end = prefix_len + self.block_size
        input_len = block_end 
        
        padded_inputs = []
        padded_targets = []
        token_weights = []
        genres = []
        timesteps = []
        
        vocab_sizes_t = torch.tensor(VOCAB_SIZES, dtype=torch.long)
        
        for item in batch:
            seq = item["tokens"]
            seq_len = seq.shape[0]
            
            padded_seq = torch.full((input_len, 5), PAD_ID, dtype=torch.long)
            valid_len = min(seq_len, input_len)
            if valid_len > 0:
                padded_seq[:valid_len] = seq[:valid_len]
            
            input_tensor = padded_seq.clone()
            target_tensor = torch.full_like(input_tensor, PAD_ID)
            weight_tensor = torch.zeros((input_len, 5), dtype=torch.float32)
            
            curr_block_valid = max(0, min(self.block_size, valid_len - prefix_len))
            
            if curr_block_valid > 0:
                t_val = random.uniform(0.001, 0.999)
                timesteps.append(t_val)
                
                # GIDD Mathematics
                B_const = 2.0 * (self.p_u / (1.0 - self.p_u))
                c_t = B_const * math.sqrt(t_val * (1.0 - t_val))
                C_t = 1.0 + c_t
                
                p_clean = (1.0 - t_val) / C_t
                p_mask = t_val / C_t
                
                # Compute e^{-\lambda_t/2} dynamically for clean weights
                e_lambda = math.sqrt((c_t + t_val) / max(1.0 - t_val, 1e-5))
                
                r = torch.rand(curr_block_valid, 5)
                mask_corrupt = (r >= p_clean) & (r < p_clean + p_mask)
                uniform_corrupt = (r >= p_clean + p_mask)
                clean_tokens = ~(mask_corrupt | uniform_corrupt)
                
                block_start = prefix_len
                block_valid_end = prefix_len + curr_block_valid
                
                original_block = input_tensor[block_start:block_valid_end].clone()
                input_block = input_tensor[block_start:block_valid_end]
                
                # Apply MASK_ID
                input_block[mask_corrupt] = MASK_ID
                
                # Apply Uniform Random Noise
                rand_tokens = torch.randint(0, 10000, (curr_block_valid, 5)) % vocab_sizes_t
                input_block[uniform_corrupt] = rand_tokens[uniform_corrupt]
                
                input_tensor[block_start:block_valid_end] = input_block
                
                # FULL GROUND TRUTH TARGETS (We want loss on ALL valid tokens in the block)
                target_tensor[block_start:block_valid_end] = original_block
                
                # --- APPLY GIDD EQ 20: DYNAMIC WEIGHTING (w_dyn) ---
                weight_block = torch.zeros((curr_block_valid, 5), dtype=torch.float32)
                weight_block[mask_corrupt] = 2.0
                weight_block[uniform_corrupt] = 1.0
                
                # Clean token weight scales depending on vocabulary size and timestep (clamped to 1.0)
                for i in range(5):
                    N_i = vocab_sizes_t[i].item()
                    w_clean_i = min(1.0, (B_const / N_i) * e_lambda)
                    weight_block[clean_tokens[:, i], i] = w_clean_i
                
                weight_tensor[block_start:block_valid_end] = weight_block
            else:
                timesteps.append(0.0) 
                
            padded_inputs.append(input_tensor)
            padded_targets.append(target_tensor)
            token_weights.append(weight_tensor)
            genres.append(item["genre"])
            
        return {
            "input": torch.stack(padded_inputs),
            "target": torch.stack(padded_targets),
            "token_weights": torch.stack(token_weights),
            "prefix_len": prefix_len,
            "genre": torch.stack(genres),
            "timestep": torch.tensor(timesteps, dtype=torch.float32)
        }


def build_dataloader(jsonl_files: list[str], split: str, batch_size: int, augment: bool = False, num_workers: int = 8, distributed: bool = False) -> DataLoader:
    processor = ProcessData()
    dataset = GIDDDataset(jsonl_files=jsonl_files, processor=processor, augment=augment)
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

def amp_context(dtype):
    return torch.amp.autocast(DEVICE, dtype=dtype) if dtype is None else torch.amp.autocast(DEVICE, dtype=dtype)

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

    if is_main_process: print("Initializing Flash Attention GIDD Model...")
    model = PrefixARModel().to(device)

    # Print model parameters
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16))

    start_step, best_val_loss = 0, float('inf')
    if RESUME_TRAINING:
        start_step, best_val_loss = load_checkpoint(model, optimizer, device)

    opt_warmup_steps = max(1, WARMUP_STEPS // ACCUM_STEPS)
    opt_total_steps = max(1, n_steps // ACCUM_STEPS)
    last_epoch = (start_step // ACCUM_STEPS) - 1 if start_step > 0 else -1
    scheduler = build_scheduler(optimizer, opt_warmup_steps, opt_total_steps, last_epoch=last_epoch)

    steps_per_epoch = len(train_loader)
    start_epoch = start_step // steps_per_epoch
    
    running_loss, log_steps = 0.0, 0
    model.train()
    step, epoch = start_step, start_epoch

    while step < n_steps:
        if hasattr(train_loader.sampler, 'set_epoch'): train_loader.sampler.set_epoch(epoch)
        
        for batch in train_loader:
            if step >= n_steps: break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with amp_context(AMP_DTYPE):
                logits_list = model(
                    target=batch["input"],          
                    genre=batch["genre"],
                    prefix_len=batch["prefix_len"],
                    timestep=batch["timestep"] 
                )
                
                # Unreduced CrossEntropy: Loss across ALL tokens, weighted individually
                criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction='none')
                raw_loss = 0.0
                
                for i, logits in enumerate(logits_list):
                    logits_reshaped = logits.transpose(1, 2)
                    target_attr = batch["target"][:, :, i]
                    weights = batch["token_weights"][:, :, i]
                    
                    unreduced_loss = criterion(logits_reshaped.float(), target_attr)
                    
                    # Apply empirical GIDD Weights (Eq. 20) pre-computed by dataloader
                    weighted_loss = unreduced_loss * weights
                    
                    # Target tensor only has valid targets for the diffusion block. Prefix is PAD_ID.
                    valid_tokens_mask = (target_attr != PAD_ID).float()
                    tokens_per_sequence = valid_tokens_mask.sum(dim=1).clamp(min=1.0)
                    
                    # Average over the block length, then batch mean
                    per_sequence_loss = weighted_loss.sum(dim=1) / tokens_per_sequence
                    raw_loss += per_sequence_loss.mean()
                    
                loss = raw_loss / ACCUM_STEPS

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += raw_loss.item() 
            log_steps += 1

            if (step + 1) % ACCUM_STEPS == 0:
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

            step += 1

            if is_main_process and (step % LOG_EVERY == 0) and log_steps > 0:
                avg_loss = running_loss / log_steps
                print(f"  Step {step:>6} | Loss: {avg_loss:>6.4f}")
                writer.add_scalar('Train/Diffusion_CE_Loss', avg_loss, step)
                writer.add_scalar('Hyperparameters/LR', optimizer.param_groups[0]['lr'], step)
                running_loss, log_steps = 0.0, 0

            if step > 0 and step % VAL_EVERY == 0:
                val_loss = evaluate_validation(model, val_loader, local_rank, device)
                if is_main_process:
                    print(f"\n--- Validation Step {step} | Val Loss: {val_loss:.4f} ---")
                    writer.add_scalar('Validation/Diffusion_CE_Loss', val_loss, step)
                    is_best = val_loss < best_val_loss
                    if is_best: best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scheduler, step, best_val_loss, is_best)
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
        
        with amp_context(AMP_DTYPE):
            logits_list = model(
                target=batch["input"],
                genre=batch["genre"],
                prefix_len=batch["prefix_len"],
                timestep=batch["timestep"]
            )
            
            criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction='none')
            loss = 0.0
            
            for i, logits in enumerate(logits_list):
                logits_reshaped = logits.transpose(1, 2)
                target_attr = batch["target"][:, :, i]
                weights = batch["token_weights"][:, :, i]
                
                unreduced_loss = criterion(logits_reshaped.float(), target_attr)
                weighted_loss = unreduced_loss * weights
                
                valid_tokens_mask = (target_attr != PAD_ID).float()
                tokens_per_sequence = valid_tokens_mask.sum(dim=1).clamp(min=1.0)
                
                per_sequence_loss = weighted_loss.sum(dim=1) / tokens_per_sequence
                loss += per_sequence_loss.mean()
            
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