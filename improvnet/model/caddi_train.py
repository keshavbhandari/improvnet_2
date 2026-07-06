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
from improvnet.model.caddi_config import *
from improvnet.model.caddi_model import CaDDiModel
from improvnet.utils.utils import ProcessData
import contextlib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _amp_dtype():
    if not torch.cuda.is_available(): return None
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16

AMP_DTYPE = _amp_dtype()

# ---------------------------------------------------------------------------
# Data Loading for Unrolled AR-Diffusion
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
        """Builds the 4-step unrolled Causal Time Machine trajectory!"""
        padded_inputs = []
        padded_targets = []
        timesteps = []
        genres = []
        
        max_unrolled_len = 0
        batch_data = []

        for item in batch:
            seq = item["tokens"]
            seq_len = seq.shape[0]

            # 1. Pick a target block to diffuse
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
                target_block = torch.full((1, 5), PAD_ID, dtype=torch.long)
                actual_L = 1

            # 2. Generate 4 random continuous timesteps (descending)
            t_vals = sorted([random.uniform(0.05, 1.0) for _ in range(4)], reverse=True)

            input_chunks = [prefix_seq]
            target_chunks = [torch.full_like(prefix_seq, PAD_ID)] # Loss ignored on prefix
            ts_chunks = [torch.zeros(prefix_len)] # Prefix is perfectly clean (t=0.0)

            # 3. Create the corrupted drafts and append them sequentially!
            for t in t_vals:
                corrupted = target_block.clone()
                num_to_mask = int(t * actual_L)
                if num_to_mask > 0:
                    for attr in range(5):
                        perm = torch.randperm(actual_L)[:num_to_mask]
                        corrupted[perm, attr] = MASK_ID

                input_chunks.append(corrupted)
                target_chunks.append(target_block) # Target is ALWAYS the clean block
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

        # Pad to max unrolled length in this batch
        for data in batch_data:
            L = data["input"].shape[0]
            pad_len = max_unrolled_len - L

            if pad_len > 0:
                pad_input = torch.full((pad_len, 5), PAD_ID, dtype=torch.long)
                pad_target = torch.full((pad_len, 5), PAD_ID, dtype=torch.long)
                pad_ts = torch.zeros(pad_len)

                padded_inputs.append(torch.cat([data["input"], pad_input]))
                padded_targets.append(torch.cat([data["target"], pad_target]))
                timesteps.append(torch.cat([data["ts"], pad_ts]))
            else:
                padded_inputs.append(data["input"])
                padded_targets.append(data["target"])
                timesteps.append(data["ts"])
            genres.append(data["genre"])
            
        # The dataloader handles the linear RoPE coordinates natively!
        seq_len = padded_inputs[0].shape[0]
        b_size = len(padded_inputs)
        
        # coords shape: [B, T, 6] (1 for sequence position, 5 for attributes)
        coords = torch.zeros(b_size, seq_len, 6, dtype=torch.long)
        pos = torch.arange(seq_len).unsqueeze(0).expand(b_size, -1)
        coords[:, :, 0] = pos
        coords[:, :, 1:] = torch.stack(padded_inputs)

        return {
            "input": torch.stack(padded_inputs),
            "target": torch.stack(padded_targets),
            "timestep": torch.stack(timesteps),
            "coords": coords,
            "genre": torch.stack(genres)
        }

def build_dataloader(jsonl_files: list[str], split: str, batch_size: int, augment: bool = False, num_workers: int = 4, distributed: bool = False) -> DataLoader:
    processor = ProcessData()
    dataset = CaDDiDataset(jsonl_files=jsonl_files, processor=processor, augment=augment)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(not distributed),
        sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=dataset.collate_fn
    )

# ---------------------------------------------------------------------------
# DDP & Training Utilities
# ---------------------------------------------------------------------------

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, scheduler, update_step, best_val_loss, is_best=False):
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_to_save = model.module if isinstance(model, DDP) else model
    checkpoint = {
        'step': update_step, # We now save the true update step
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

def train(n_steps: int = N_STEPS):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (local_rank == 0)

    writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, "runs", RUN_NAME)) if is_main_process else None

    train_loader = build_dataloader(JSONL_FILES, "train", BATCH_SIZE, augment=True, distributed=True)
    val_loader = build_dataloader(JSONL_FILES, "validation", BATCH_SIZE, augment=False, distributed=True)

    if is_main_process: print("Initializing CaDDi Pure Causal Model...")
    model = CaDDiModel().to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16))

    update_step, best_val_loss = 0, float('inf')
    if RESUME_TRAINING:
        update_step, best_val_loss = load_checkpoint(model, optimizer, device)

    # Scheduler steps are purely based on optimizer updates now
    scheduler = build_scheduler(optimizer, WARMUP_STEPS, n_steps, last_epoch=update_step - 1 if update_step > 0 else -1)

    steps_per_epoch = len(train_loader)
    
    # We calculate the starting micro-batch based on saved update_step
    micro_step = update_step * ACCUM_STEPS 
    start_epoch = micro_step // steps_per_epoch
    
    running_loss, log_steps = 0.0, 0
    model.train()
    epoch = start_epoch

    while update_step < n_steps:
        if hasattr(train_loader.sampler, 'set_epoch'): train_loader.sampler.set_epoch(epoch)
        
        for batch in train_loader:
            if update_step >= n_steps: break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Identify if this is a sync step
            is_accumulating = (micro_step + 1) % ACCUM_STEPS != 0
            
            # Massive DDP Speedup: Only sync gradients on the final accumulation step!
            ddp_context = model.no_sync() if is_accumulating else contextlib.nullcontext()
            
            with ddp_context:
                with amp_context(AMP_DTYPE):
                    logits_list = model(
                        target=batch["input"],          
                        coords=batch["coords"],
                        timestep=batch["timestep"],
                        genre=batch["genre"]
                    )
                    
                    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
                    raw_loss = 0.0
                    
                    for i, logits in enumerate(logits_list):
                        logits_reshaped = logits.transpose(1, 2)
                        raw_loss += criterion(logits_reshaped.float(), batch["target"][:, :, i])
                        
                    loss = raw_loss / ACCUM_STEPS

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            running_loss += raw_loss.item() 
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

                # Logging is now correctly bound to the true update_step!
                if is_main_process and (update_step % LOG_EVERY == 0) and log_steps > 0:
                    avg_loss = running_loss / log_steps
                    print(f"  Step {update_step:>6} | Loss: {avg_loss:>6.4f}")
                    writer.add_scalar('Train/Diffusion_CE_Loss', avg_loss, update_step)
                    writer.add_scalar('Hyperparameters/LR', optimizer.param_groups[0]['lr'], update_step)
                    running_loss, log_steps = 0.0, 0

                if update_step > 0 and update_step % VAL_EVERY == 0:
                    val_loss = evaluate_validation(model, val_loader, local_rank, device)
                    if is_main_process:
                        print(f"\n--- Validation Step {update_step} | Val Loss: {val_loss:.4f} ---")
                        writer.add_scalar('Validation/Diffusion_CE_Loss', val_loss, update_step)
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
        
        with amp_context(AMP_DTYPE):
            logits_list = model(
                target=batch["input"],
                coords=batch["coords"],
                timestep=batch["timestep"],
                genre=batch["genre"]
            )
            criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
            loss = 0.0
            for i, logits in enumerate(logits_list):
                logits_reshaped = logits.transpose(1, 2)
                loss += criterion(logits_reshaped.float(), batch["target"][:, :, i])
            
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