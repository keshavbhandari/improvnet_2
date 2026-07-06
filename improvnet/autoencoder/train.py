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
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from improvnet.autoencoder.config import (
    SEQ_LEN, TARGET_BETA, BATCH_SIZE, ACCUM_STEPS, N_STEPS, WARMUP_STEPS, LOG_EVERY,
    GRAD_CLIP, VAL_EVERY, SAVE_DIR, VAL_LOG_FILE, RESUME_TRAINING, 
    JSONL_FILES, RUN_NAME
)
from improvnet.autoencoder.model import ContinuousAutoencoder
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

class MIDISequenceDataset(Dataset):
    def __init__(
        self, 
        jsonl_files: list[str], 
        split: str, 
        processor, 
        seq_len: int, 
        augment: bool = False
    ):
        self.jsonl_files = jsonl_files
        self.processor = processor
        self.seq_len = seq_len
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
        # 1. Fetch file location and read line
        file_idx, offset = self.global_indices[idx]
        jsonl_path = self.jsonl_files[file_idx]
        
        f = self._get_file_handle(jsonl_path)
        f.seek(offset)
        line_bytes = f.readline()
        line_str = line_bytes.decode('utf-8')
        
        # 2. Parse JSON
        entry = json.loads(line_str.strip())
        tokens_raw = entry.get("tokens")
        if not tokens_raw:
            raise KeyError("JSON entry missing 'tokens' key.")
            
        tokens = self._lists_to_tuples(tokens_raw)
        
        # 3. Extract random sequence length
        tokens = self.processor.get_aligned_random_segment(tokens, self.seq_len)
        
        # 4. Augmentations (Train only)
        if self.split == "train":
            rand_val = random.random()
            if rand_val < 0.10:
                tokens = self.processor.skyline_groundline(tokens, algorithm="skyline")
            elif rand_val < 0.20:
                tokens = self.processor.skyline_groundline(tokens, algorithm="groundline")
            elif rand_val < 0.30:
                tokens = self.processor.extract_rhythm(tokens, ratio=1.0) 
        
            if self.augment:
                tokens = self.processor.pitch_augmentation(tokens)
            
        # 5. Convert to Tensors and Format via Processor
        # Returns a dict of 1D tensors
        token_tensors_dict = self.processor.tokens_to_tensor(tokens)
        
        # Formats dict into a padded, stacked 2D tensor [seq_len, 5]
        final_tensor = self.processor.format_sequence(token_tensors_dict, self.seq_len)
        
        return final_tensor


def build_dataloader(
    jsonl_files: list[str],
    split: str,
    seq_len: int,
    batch_size: int,
    augment: bool = False,
    num_workers: int = 4,
    shuffle: bool = True,
    distributed: bool = False
) -> DataLoader:
    """Convenience function to instantiate the dataset and dataloader."""
    # Assuming ProcessData() is properly imported/available
    processor = ProcessData()
    
    dataset = MIDISequenceDataset(
        jsonl_files=jsonl_files,
        split=split,
        processor=processor,
        seq_len=seq_len,
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

    writer = None
    if is_main_process:
        print("=" * 70 + "\nContinuous VAE (Beta-VAE)\n" + "=" * 70)
        log_dir = os.path.join(SAVE_DIR, "runs", RUN_NAME)
        writer = SummaryWriter(log_dir=log_dir)

    # Note: build_dataloader no longer needs patch_size or n_patches arguments.
    # It just needs seq_len (e.g., 2048) to chunk the raw MIDI streams.
    train_loader = build_dataloader(
        jsonl_files=JSONL_FILES, split="train",
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE, augment=True, distributed=True
    )
    
    val_loader = build_dataloader(
        jsonl_files=JSONL_FILES, split="validation",
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE, augment=False, distributed=True
    )

    model = ContinuousAutoencoder().to(device)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = build_scheduler(optimizer, WARMUP_STEPS, n_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16))

    start_step, best_val_loss = 0, float('inf')
    if RESUME_TRAINING:
        start_step, best_val_loss = load_checkpoint(model, optimizer, scheduler, device)

    # Add trackers for the new loss components
    running_loss, running_ce, running_kl, running_acc, log_steps = 0.0, 0.0, 0.0, 0.0, 0
    model.train()
    step, epoch = start_step, 0

    while step < n_steps:
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        for x in train_loader:
            if step >= n_steps: break
            x = x.to(device)
            
            with amp_context(AMP_DTYPE):
                logits_list, mu, logvar = model(x)
                
                # --- KL Annealing ---
                # Linearly scale Beta from 0.0 to 0.05 over the first 20,000 steps
                warmup_steps = 20000.0
                current_beta = TARGET_BETA * min(1.0, step / warmup_steps)
                
                loss_dict = model.module.loss(
                    logits_list, x, mu=mu, logvar=logvar, beta=current_beta
                )
                raw_loss = loss_dict["loss"]
                loss = raw_loss / ACCUM_STEPS

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Calculate accuracy on the primary (noisy) predictions
            current_acc = model.module.calculate_accuracy(logits_list, x)
            
            # Accumulate metrics
            running_loss += raw_loss.item() 
            running_ce += loss_dict["ce_loss"].item()
            running_kl += loss_dict["kl_loss"].item() if isinstance(loss_dict["kl_loss"], torch.Tensor) else loss_dict["kl_loss"]
            running_acc += current_acc
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

            # Logging
            if is_main_process and (step % LOG_EVERY == 0) and log_steps > 0:
                avg_loss = running_loss / log_steps
                avg_ce = running_ce / log_steps
                avg_kl = running_kl / log_steps
                avg_acc = running_acc / log_steps
                
                # --- UPDATED: Console Print ---
                print(f"  Step {step:>6} | Loss: {avg_loss:>6.4f} (CE: {avg_ce:.4f}, KL: {avg_kl:.4f}) | Acc: {avg_acc*100:>6.2f}%")

                # --- UPDATED: Tensorboard Logging ---
                writer.add_scalar('Train/Total_Loss', avg_loss, step)
                writer.add_scalar('Train/CE_Loss', avg_ce, step)
                writer.add_scalar('Train/KL_Loss', avg_kl, step)
                writer.add_scalar('Train/Accuracy', avg_acc * 100, step)
                writer.add_scalar('Hyperparameters/LR', optimizer.param_groups[0]['lr'], step)

                running_loss, running_ce, running_kl, running_acc, log_steps = 0.0, 0.0, 0.0, 0.0, 0

            # Validation
            if step > 0 and step % VAL_EVERY == 0:
                val_loss, val_acc = evaluate_validation(model, val_loader, local_rank, device, max_batches=1000)
                if is_main_process:
                    print(f"\n--- Validation Step {step} | Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% ---")
                    writer.add_scalar('Validation/Loss', val_loss, step)
                    writer.add_scalar('Validation/Accuracy', val_acc * 100, step)
                    
                    is_best = val_loss < best_val_loss
                    if is_best: best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scheduler, step, best_val_loss, is_best)
                    print("-" * 50 + "\n")

        epoch += 1

    if is_main_process and writer: writer.close()
    cleanup_ddp()
    return model

@torch.no_grad()
def evaluate_validation(model, val_loader, local_rank, device, max_batches: int = 100):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    
    model_obj = model.module if hasattr(model, 'module') else model
    is_main_process = (local_rank == 0)
    total_steps = min(len(val_loader), max_batches)
    iterator = tqdm(val_loader, desc="Validation", total=total_steps, leave=False) if is_main_process else val_loader

    for x in iterator:
        if steps >= max_batches:
            break
            
        x = x.to(device)
        
        with amp_context(AMP_DTYPE):
            # --- VAE Forward Pass ---
            # During eval, the reparameterization trick is disabled and it just returns mu
            logits_list, mu, logvar = model(x)
            
            # --- VAE Loss ---
            loss_dict = model_obj.loss(logits_list, x, mu=mu, logvar=logvar, beta=0.05)
            loss = loss_dict["loss"] 
            
            acc = model_obj.calculate_accuracy(logits_list, x)
            
        total_loss += loss.item()
        total_acc += acc
        steps += 1

    local_avg_loss = total_loss / max(1, steps)
    local_avg_acc = total_acc / max(1, steps)
    
    if torch.distributed.is_initialized():
        metrics = torch.tensor([local_avg_loss, local_avg_acc], device=device)
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
        metrics /= torch.distributed.get_world_size()
        global_avg_loss, global_avg_acc = metrics.tolist()
    else:
        global_avg_loss, global_avg_acc = local_avg_loss, local_avg_acc

    model.train()
    return global_avg_loss, global_avg_acc


if __name__ == "__main__":

    model = train(n_steps=N_STEPS)