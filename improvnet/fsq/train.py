"""
Compound FSQ Autoencoder with MRA (Moonbeam Tokenizer)
-------------------------------------------------------
Architecture:
  • Moonbeam Compound Encoding — Note events are represented as 5 attributes:
    (instrument, pitch, velocity, onset, duration).
  • Multidimensional Relative Attention (MRA) — The encoder utilizes a 5-group
    attention mechanism, where each group calculates relative rotations based on
    one specific attribute dimension.
  • FSQ Bottleneck — Finite Scalar Quantization mapping continuous latent patches
    into discrete grid levels.
"""

import math
import random
import os
import json
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from improvnet.fsq.config import (
    PATCH_SIZE, N_PATCHES, LATENT_DIM, 
    BATCH_SIZE, ACCUM_STEPS, N_STEPS, WARMUP_STEPS, LOG_EVERY,
    GRAD_CLIP, USE_CKPT, VAL_EVERY, SAVE_DIR, VAL_LOG_FILE, 
    RESUME_TRAINING, JSONL_FILES
)
from improvnet.fsq.model import FSQAutoencoder
from improvnet.utils.utils import ProcessData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _amp_dtype():
    if not torch.cuda.is_available():
        return None
    major = torch.cuda.get_device_capability()[0]
    if major >= 8:
        return torch.bfloat16
    return torch.float16

AMP_DTYPE = _amp_dtype()

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

# class MIDIPatchDataset(Dataset):
#     def __init__(
#         self, 
#         jsonl_files: list[str], 
#         split: str, 
#         processor: ProcessData, 
#         patch_size: int, 
#         n_patches: int, 
#         augment: bool = False
#     ):
#         self.files = read_jsonl_files(jsonl_files, split=split)
#         self.processor = processor
#         self.patch_size = patch_size
#         self.n_patches = n_patches
#         self.target_seq_len = patch_size * n_patches
#         self.augment = augment
#         self.split = split

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         file_path = self.files[idx].get("midi_filepath") 
#         if not file_path:
#             raise KeyError(f"JSONL entry missing 'midi_path' key: {self.files[idx]}")
        
#         # 1. Load and Tokenize
#         midi_dict = self.processor.read_midi(file_path)
#         tokens = self.processor.midi_to_tokens(midi_dict)
        
#         # 2. Extract Segment
#         tokens = self.processor.get_random_segment_from_data(tokens, self.target_seq_len)
        
#         # 3. Dynamic Masking Router (Train only, 10% each)
#         if self.split == "train":
#             rand_val = random.random()
#             if rand_val < 0.10:
#                 tokens = self.processor.skyline_groundline(tokens, algorithm="skyline")
#             elif rand_val < 0.20:
#                 tokens = self.processor.skyline_groundline(tokens, algorithm="groundline")
#             elif rand_val < 0.30:
#                 # ratio=1.0 ensures the entire segment becomes pure rhythm
#                 tokens = self.processor.extract_rhythm(tokens, ratio=1.0) 
#             # Remaining 70% falls through as Raw MIDI
        
#         # 4. Augment (Pitch shift safely ignores <BLANK> tokens)
#         if self.augment and self.split == "train":
#             tokens = self.processor.pitch_augmentation(tokens)
            
#         # 5. Convert to Tensors & Format to Patches
#         token_tensors = self.processor.tokens_to_tensor(tokens)
#         patched_tensor = self.processor.format_into_patches(token_tensors, self.patch_size, self.n_patches)
        
#         return patched_tensor

class MIDIPatchDataset(Dataset):
    def __init__(
        self, 
        jsonl_files: list[str], 
        split: str, 
        processor, 
        patch_size: int, 
        n_patches: int, 
        augment: bool = False
    ):
        self.jsonl_files = jsonl_files
        self.processor = processor
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.target_seq_len = patch_size * n_patches
        self.augment = augment
        self.split = split
        
        # Dictionary to hold open file handles for each worker process
        self.file_handles = {}
        
        # Build a unified global index across all provided JSONL files
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
        """Lazy-loads the file handle per worker to ensure thread safety."""
        if jsonl_path not in self.file_handles:
            # Open in binary mode for precise byte seeking
            self.file_handles[jsonl_path] = open(jsonl_path, 'rb')
        return self.file_handles[jsonl_path]

    def _lists_to_tuples(self, tokens_raw):
        """Recursively converts JSON lists back to tuples."""
        tokens = []
        for event in tokens_raw:
            if isinstance(event, list):
                event_tuple = tuple(tuple(f) if isinstance(f, list) else f for f in event)
                tokens.append(event_tuple)
            else:
                tokens.append(event)
        return tokens

    def __getitem__(self, idx):
        # 1. Get the exact file and byte location for this sample
        file_idx, offset = self.global_indices[idx]
        jsonl_path = self.jsonl_files[file_idx]
        
        # 2. Seek to the exact byte and read the line natively from the hard drive
        f = self._get_file_handle(jsonl_path)
        f.seek(offset)
        line_bytes = f.readline()
        line_str = line_bytes.decode('utf-8')
        
        # 3. Parse JSON and load tokens
        entry = json.loads(line_str.strip())
        tokens_raw = entry.get("tokens")
        
        if not tokens_raw:
            raise KeyError("JSON entry missing 'tokens' key.")
            
        tokens = self._lists_to_tuples(tokens_raw)
        
        # 4. Extract Segment
        tokens = self.processor.get_random_segment_from_data(tokens, self.target_seq_len)
        
        # 5. Dynamic Masking Router
        if self.split == "train":
            rand_val = random.random()
            if rand_val < 0.10:
                tokens = self.processor.skyline_groundline(tokens, algorithm="skyline")
            elif rand_val < 0.20:
                tokens = self.processor.skyline_groundline(tokens, algorithm="groundline")
            elif rand_val < 0.30:
                tokens = self.processor.extract_rhythm(tokens, ratio=1.0) 
        
        # 6. Augment 
        if self.augment and self.split == "train":
            tokens = self.processor.pitch_augmentation(tokens)
            
        # 7. Convert to Tensors & Format
        token_tensors = self.processor.tokens_to_tensor(tokens)
        patched_tensor = self.processor.format_into_patches(token_tensors, self.patch_size, self.n_patches)
        
        return patched_tensor


def build_dataloader(
    jsonl_files: list[str],
    split: str,
    patch_size: int,
    n_patches: int,
    batch_size: int,
    augment: bool = False,
    num_workers: int = 4,
    shuffle: bool = True,
    distributed: bool = False
) -> DataLoader:
    """Convenience function to instantiate the dataset and dataloader."""
    processor = ProcessData()
    
    dataset = MIDIPatchDataset(
        jsonl_files=jsonl_files,
        split=split,
        processor=processor,
        patch_size=patch_size,
        n_patches=n_patches,
        augment=augment
    )
    
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # DistributedSampler handles shuffling internally

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True, # Recommended for faster host-to-GPU memory copies
        drop_last=True   # Useful if maintaining exact batch dimensions is important
    )
    
    return loader
    

# ---------------------------------------------------------------------------
# DDP & Checkpoint Utilities
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def log_val_metrics(step, val_loss, val_acc):
    """Appends validation metrics to a text file (Rank 0 only)."""
    with open(VAL_LOG_FILE, "a") as f:
        f.write(f"Step: {step:>6} | Val Loss: {val_loss:>8.4f} | Val Acc: {val_acc*100:>6.2f}%\n")

def save_checkpoint(model, optimizer, scheduler, step, best_val_loss, is_best=False):
    """Saves model checkpoint. Saves a separate file if it's the best model."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Unwrap DDP model
    model_to_save = model.module if isinstance(model, DDP) else model
    
    checkpoint = {
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }
    
    # Always save the latest checkpoint for resuming
    torch.save(checkpoint, os.path.join(SAVE_DIR, "latest_checkpoint.pt"))
    
    # Save a separate copy if it improved validation loss
    if is_best:
        torch.save(checkpoint, os.path.join(SAVE_DIR, "best_model.pt"))
        print(f"  --> Saved new best model at step {step}!")

def load_checkpoint(model, optimizer, scheduler, device):
    """Loads the latest checkpoint to resume training."""
    checkpoint_path = os.path.join(SAVE_DIR, "latest_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Starting from scratch.")
        return 0, float('inf')
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Unwrap DDP model
    model_to_load = model.module if isinstance(model, DDP) else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Resumed from checkpoint at step {checkpoint['step']}")
    return checkpoint['step'], checkpoint['best_val_loss']


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def amp_context(dtype):
    if dtype is None:
        return torch.amp.autocast("cpu", enabled=False)
    return torch.amp.autocast(DEVICE, dtype=dtype)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n_steps: int = N_STEPS):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (local_rank == 0)

    if is_main_process:
        print("=" * 70 + "\nCompound FSQ Autoencoder with DDP\n" + "=" * 70)

    train_loader = build_dataloader(
        jsonl_files=JSONL_FILES, 
        split="train",
        patch_size=PATCH_SIZE,
        n_patches=N_PATCHES,
        batch_size=BATCH_SIZE,
        augment=True,
        num_workers=4,
        distributed=True
    )
    
    val_loader = build_dataloader(
        jsonl_files=JSONL_FILES, 
        split="validation",
        patch_size=PATCH_SIZE,
        n_patches=N_PATCHES,
        batch_size=BATCH_SIZE*4,  # Larger batch for faster validation
        augment=False,
        num_workers=4,
        distributed=True
    )

    # 2. Setup Model & DDP
    model = FSQAutoencoder().to(device)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.95))
    scheduler = build_scheduler(optimizer, WARMUP_STEPS, n_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16))

    if is_main_process:
        print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  Device: {DEVICE}")
        print(f"  Precision: {str(AMP_DTYPE).replace('torch.', '') if AMP_DTYPE else 'fp32 (no AMP)'}")
        print(f"  Batch Size: {BATCH_SIZE} (micro) × {ACCUM_STEPS} accum = {BATCH_SIZE * ACCUM_STEPS} effective")
        print(f"  Grad Checkpointing: {'Enabled' if USE_CKPT else 'Disabled'}")
        print(f"  Warmup Steps: {WARMUP_STEPS}")
        print(f"  Total Steps: {n_steps}")
        print(f"  Grad Clip Norm: {GRAD_CLIP}\n")
        print(f"  {'Step':>6}  {'Train Loss':>10}  {'Acc':>6}  {'Val Loss':>10}  {'Val Acc':>8}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*8}")

    # 3. Resume from Checkpoint
    start_step = 0
    best_val_loss = float('inf')
    if RESUME_TRAINING:
        start_step, best_val_loss = load_checkpoint(model, optimizer, scheduler, device)

    # 4. Training Loop
    model.train()
    step = start_step
    epoch = 0

    while step < n_steps:
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        for x in train_loader:
            if step >= n_steps:
                break
                
            x = x.to(device)
            
            with amp_context(AMP_DTYPE):
                logits_list, z_q, indices = model(x)
                loss = model.module.loss(logits_list, x) / ACCUM_STEPS

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step optimizer every ACCUM_STEPS
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

            # --- Logging ---
            if is_main_process and (step % LOG_EVERY == 0):
                acc = model.module.calculate_accuracy(logits_list, x)
                flat = indices.reshape(-1, LATENT_DIM)
                unique = flat.unique(dim=0).shape[0]
                print(f"  Step {step:>6} | Train Loss: {loss.item() * ACCUM_STEPS:>8.4f} | Acc: {acc*100:>6.2f}% | Codes: {unique:>5,}/{flat.shape[0]:,}")

            # --- Validation & Checkpointing ---
            if step > 0 and step % VAL_EVERY == 0:
                print(f"\nRunning validation at step {step}...")
                val_loss, val_acc = evaluate_validation(model, val_loader, local_rank, device, max_batches=500)
                
                if is_main_process:
                    print(f"\n--- Validation at Step {step} ---")
                    print(f"    Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
                    
                    log_val_metrics(step, val_loss, val_acc)
                    
                    # Save checkpoint if it's the best
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                        
                    save_checkpoint(model, optimizer, scheduler, step, best_val_loss, is_best=is_best)
                    print("-" * 31 + "\n")

        epoch += 1

    cleanup_ddp()
    return model


from tqdm import tqdm

def evaluate_validation(model, val_loader, local_rank, device, max_batches=50):
    """Evaluates the model on a subset of the validation set and syncs metrics across GPUs."""
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_acc = torch.tensor(0.0, device=device)
    batches = 0
    is_main_process = (local_rank == 0)

    with torch.no_grad():
        # Wrap the loader in tqdm, but only display it on the main GPU (rank 0)
        loader = tqdm(val_loader, desc="Validation", disable=not is_main_process)
        
        for x in loader:
            if batches >= max_batches:
                break
                
            x = x.to(device)
            with amp_context(AMP_DTYPE):
                # model() is wrapped in DDP, so loss calculation calls model.module
                logits_list, _, _ = model(x)
                loss = model.module.loss(logits_list, x)
                acc = model.module.calculate_accuracy(logits_list, x)
                
            total_loss += loss
            total_acc += acc
            batches += 1
            
            # Update progress bar with the running loss
            if is_main_process:
                loader.set_postfix(loss=f"{(total_loss / batches).item():.4f}")

    # Avoid division by zero if loader was somehow empty
    if batches == 0:
        batches = 1

    # Average metrics locally
    avg_loss = total_loss / batches
    avg_acc = total_acc / batches

    # Sync across all GPUs safely
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
    
    world_size = dist.get_world_size()
    avg_loss /= world_size
    avg_acc /= world_size

    model.train()
    return avg_loss.item(), avg_acc.item()


if __name__ == "__main__":
    # Uncomment to test memory/logic on a tiny batch before training
    # sanity_check()

    model = train(n_steps=N_STEPS)
    evaluate_validation(model)