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
from improvnet.model.config_fm import *
from improvnet.model.model_fm import FlowMatchingModel
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

class FMSequenceDataset(Dataset):
    def __init__(
        self, 
        jsonl_files: list[str], 
        processor, 
        max_target_len: int = 1024,
        max_cond_len: int = 128,  # Max length for the extracted segments
        patch_size: int = 8
    ):
        self.jsonl_files = jsonl_files
        self.processor = processor
        self.max_target_len = max_target_len
        self.max_cond_len = max_cond_len
        self.patch_size = patch_size
        
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

    def _pad_and_mask(self, tokens: list, target_len: int):
        """Pads tokens, ensures PATCH_SIZE divisibility, and returns a boolean mask."""
        # 1. Truncate if too long
        if len(tokens) > target_len:
            tokens = tokens[:target_len]
            
        # 2. Ensure divisibility by patch_size (so Autoencoder doesn't crash)
        remainder = len(tokens) % self.patch_size
        if remainder != 0:
            tokens = tokens[: -remainder]
            
        actual_len = len(tokens)
        
        # 3. Format using your processor (handles the <P> padding up to target_len)
        token_tensors_dict = self.processor.tokens_to_tensor(tokens)
        padded_tensor = self.processor.format_sequence(token_tensors_dict, target_len)
        
        # 4. Create Boolean Mask (True = Is Padding = Ignore in Attention)
        # Sequence enters as [T, 5]. Autoencoder squishes to [T // Patch_Size, 128].
        # We need the mask to match the Latent sequence length!
        num_patches = target_len // self.patch_size
        actual_patches = actual_len // self.patch_size
        
        mask = torch.ones(num_patches, dtype=torch.bool)
        mask[:actual_patches] = False # False means "do not ignore"
        
        return padded_tensor, mask

    def __getitem__(self, idx):
        # Fetch file location and read line
        file_idx, offset = self.global_indices[idx]
        jsonl_path = self.jsonl_files[file_idx]
        
        f = self._get_file_handle(jsonl_path)
        f.seek(offset)
        line_bytes = f.readline()
        line_str = line_bytes.decode('utf-8')
        
        # Parse JSON
        entry = json.loads(line_str.strip())
        tokens_raw = entry.get("tokens")
        if not tokens_raw:
            raise KeyError("JSON entry missing 'tokens' key.")
            
        tokens = self._lists_to_tuples(tokens_raw)
        
        # Extract full target sequence
        target_tokens = self.processor.get_aligned_random_segment(tokens, self.max_target_len)
        target_tensor, target_mask = self._pad_and_mask(target_tokens, self.max_target_len)
        
        # Extract Conditioning Segments (Dictionary with melody, harmony, rhythm)
        cond_dict = self.processor.extract_conditioning_segments(
            target_tokens, min_notes=4, max_notes=self.max_cond_len
        )
        
        mel_tensor, mel_mask = self._pad_and_mask(cond_dict['melody'], self.max_cond_len)
        har_tensor, har_mask = self._pad_and_mask(cond_dict['harmony'], self.max_cond_len)
        rhy_tensor, rhy_mask = self._pad_and_mask(cond_dict['rhythm'], self.max_cond_len)
        
        # Extract Instruments
        inst_multihot = self.processor.get_instrument_multihot(target_tokens)
        
        # RETURN DICT
        return {
            "target": target_tensor,
            "target_mask": target_mask,
            "melody": mel_tensor, "mel_mask": mel_mask,
            "harmony": har_tensor, "har_mask": har_mask,
            "rhythm": rhy_tensor, "rhy_mask": rhy_mask,
            "inst_multihot": inst_multihot
        }


def build_dataloader(
    jsonl_files: list[str],
    split: str,
    batch_size: int,
    augment: bool = False,
    num_workers: int = 4,
    shuffle: bool = True,
    distributed: bool = False
) -> DataLoader:
    """Convenience function to instantiate the Flow Matching dataset and dataloader."""
    processor = ProcessData()
    
    dataset = FMSequenceDataset(
        jsonl_files=jsonl_files,
        processor=processor,
        max_target_len=MAX_LATENT_SEQ_LEN * PATCH_SIZE, # E.g., 2048 * 8 = 16,384 tokens
        max_cond_len=128, #* PATCH_SIZE, # E.g., 1024 tokens for condition segments
        patch_size=PATCH_SIZE
    )
    
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True, 
        drop_last=True   
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

@torch.no_grad()
def chunked_encode(autoencoder, x: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """Safely encodes massive sequences through the frozen AE without blowing up VRAM."""
    B, T, _ = x.shape
    latents = []
    
    for i in range(0, T, chunk_size):
        x_chunk = x[:, i : i + chunk_size, :]
        z_chunk = autoencoder.encode_to_latents(x_chunk)
        latents.append(z_chunk)
        
    return torch.cat(latents, dim=1)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n_steps: int = N_STEPS):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (local_rank == 0)

    writer = None
    if is_main_process:
        print("=" * 70 + "\nStage 2: Flow Matching Generative Model\n" + "=" * 70)
        log_dir = os.path.join(SAVE_DIR, "runs", RUN_NAME)
        writer = SummaryWriter(log_dir=log_dir)

    train_loader = build_dataloader(
        jsonl_files=JSONL_FILES, split="train", batch_size=BATCH_SIZE, augment=True, distributed=True
    )
    val_loader = build_dataloader(
        jsonl_files=JSONL_FILES, split="validation", batch_size=BATCH_SIZE, augment=False, distributed=True
    )

    # 1. Load the FROZEN Stage 1 Autoencoder
    if is_main_process: print("Loading frozen Autoencoder...")
    autoencoder = ContinuousAutoencoder().to(device)
    ae_state = torch.load(AE_CHECKPOINT, map_location=device)
    autoencoder.load_state_dict(ae_state['model_state_dict'] if 'model_state_dict' in ae_state else ae_state)
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False # Freeze completely

    # 2. Initialize the Stage 2 Flow Matching Model
    model = FlowMatchingModel(
        latent_dim=LATENT_DIM, hidden_dim=FM_HIDDEN_DIM, 
        num_layers=FM_LAYERS, num_heads=FM_HEADS, num_inst_classes=NUM_INSTRUMENT_CLASSES
    ).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = build_scheduler(optimizer, WARMUP_STEPS, n_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16))

    start_step, best_val_loss = 0, float('inf')
    if RESUME_TRAINING:
        start_step, best_val_loss = load_checkpoint(model, optimizer, scheduler, device)

    running_loss, log_steps = 0.0, 0
    model.train()
    step, epoch = start_step, 0

    while step < n_steps:
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        for batch in train_loader:
            if step >= n_steps: break
            
            # Move all tensors to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # --- CFG Dropout generation (Per Batch) ---
            is_uncond = random.random() < P_UNCOND
            cfg_drops = {
                "melody": is_uncond or (random.random() < P_DROP_MELODY),
                "harmony": is_uncond or (random.random() < P_DROP_HARMONY),
                "rhythm": is_uncond or (random.random() < P_DROP_RHYTHM),
                "inst": is_uncond or (random.random() < P_DROP_INST)
            }

            # --- Encode Tokens to Latents (No Gradients) ---
            with torch.no_grad():
                # We wrap this in amp_context so the frozen AE runs in ultra-fast FP16/BF16
                with amp_context(AMP_DTYPE): 
                    # 1. Chunk the massive target sequence (16,384 tokens)
                    z_1 = chunked_encode(autoencoder, batch["target"], chunk_size=1024)
                    
                    # 2. The conditions are short (max 1024 tokens), so they can be encoded instantly
                    z_mel = autoencoder.encode_to_latents(batch["melody"])
                    z_har = autoencoder.encode_to_latents(batch["harmony"])
                    z_rhy = autoencoder.encode_to_latents(batch["rhythm"])
            
            # --- Flow Matching Math ---
            B = z_1.shape[0]
            # 1. Sample pure Gaussian noise
            z_0 = torch.randn_like(z_1)
            # 2. Sample time t ~ U(0, 1)
            t = torch.rand((B,), device=device)
            t_expand = t.view(B, 1, 1) 
            
            # 3. Interpolate latents: z_t = t * z_1 + (1 - t) * z_0
            z_t = t_expand * z_1 + (1.0 - t_expand) * z_0
            
            # 4. Target vector field: v_target = z_1 - z_0
            v_target = z_1 - z_0

            # --- Forward Pass ---
            with amp_context(AMP_DTYPE):
                v_pred = model(
                    z_t=z_t, time=t, 
                    z_mel=z_mel, mel_mask=batch["mel_mask"], 
                    z_har=z_har, har_mask=batch["har_mask"], 
                    z_rhy=z_rhy, rhy_mask=batch["rhy_mask"], 
                    inst_multihot=batch["inst_multihot"], 
                    cfg_drops=cfg_drops
                )
                
                # 1. Calculate unreduced loss [Batch, Seq_Len, Latent_Dim]
                unreduced_loss = F.mse_loss(v_pred, v_target, reduction='none')
                
                # 2. Invert the mask (True = ignore padding, False = valid music)
                # Convert to float and add feature dimension: [Batch, Seq_Len, 1]
                valid_mask = (~batch["target_mask"]).unsqueeze(-1).float()
                
                # 3. Zero out the loss on padding tokens
                masked_loss = unreduced_loss * valid_mask
                
                # 4. Calculate the true mean ONLY over the valid tokens
                # We multiply valid_mask.sum() by LATENT_DIM to get total valid elements
                raw_loss = masked_loss.sum() / (valid_mask.sum() * LATENT_DIM + 1e-8)
                
                loss = raw_loss / ACCUM_STEPS

            # --- Backward Pass ---
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

            # Logging
            if is_main_process and (step % LOG_EVERY == 0) and log_steps > 0:
                avg_loss = running_loss / log_steps
                print(f"  Step {step:>6} | Flow MSE Loss: {avg_loss:>6.4f}")

                writer.add_scalar('Train/Flow_MSE_Loss', avg_loss, step)
                writer.add_scalar('Hyperparameters/LR', optimizer.param_groups[0]['lr'], step)

                running_loss, log_steps = 0.0, 0

            # Validation
            if step > 0 and step % VAL_EVERY == 0:
                val_loss = evaluate_validation(model, autoencoder, val_loader, local_rank, device, max_batches=50)
                if is_main_process:
                    print(f"\n--- Validation Step {step} | Flow MSE Loss: {val_loss:.4f} ---")
                    writer.add_scalar('Validation/Flow_MSE_Loss', val_loss, step)
                    
                    is_best = val_loss < best_val_loss
                    if is_best: best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scheduler, step, best_val_loss, is_best)
                    print("-" * 50 + "\n")

        epoch += 1

    if is_main_process and writer: writer.close()
    cleanup_ddp()
    return model

@torch.no_grad()
def evaluate_validation(model, autoencoder, val_loader, local_rank, device, max_batches: int = 50):
    model.eval()
    autoencoder.eval()

    torch.cuda.empty_cache()

    total_loss = 0.0
    steps = 0
    
    is_main_process = (local_rank == 0)
    total_steps = min(len(val_loader), max_batches)
    iterator = tqdm(val_loader, desc="Validation", total=total_steps, leave=False) if is_main_process else val_loader

    # Fully conditional evaluation (No drops)
    cfg_drops_val = {"melody": False, "harmony": False, "rhythm": False, "inst": False}

    for batch in iterator:
        if steps >= max_batches: break
            
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # 1. Encode Target and Conditions
        with torch.amp.autocast('cuda', enabled=(AMP_DTYPE == torch.float16)):
            z_1 = chunked_encode(autoencoder, batch["target"], chunk_size=1024)
            z_mel = autoencoder.encode_to_latents(batch["melody"])
            z_har = autoencoder.encode_to_latents(batch["harmony"])
            z_rhy = autoencoder.encode_to_latents(batch["rhythm"])
        
        B, seq_len, latent_dim = z_1.shape
        z_0 = torch.randn_like(z_1)
        t = torch.rand((B,), device=device)
        t_expand = t.view(B, 1, 1) 
        
        z_t = t_expand * z_1 + (1.0 - t_expand) * z_0
        v_target = z_1 - z_0
        
        with torch.amp.autocast('cuda', enabled=(AMP_DTYPE == torch.float16)):
            v_pred = model(
                z_t=z_t, time=t, 
                z_mel=z_mel, mel_mask=batch["mel_mask"], 
                z_har=z_har, har_mask=batch["har_mask"], 
                z_rhy=z_rhy, rhy_mask=batch["rhy_mask"], 
                inst_multihot=batch["inst_multihot"], 
                cfg_drops=cfg_drops_val
            )
            
            # 1. Calculate unreduced loss
            unreduced_loss = F.mse_loss(v_pred, v_target, reduction='none')
            
            # 2. Invert the mask (True = valid music, False = padding)
            valid_mask = (~batch["target_mask"]).unsqueeze(-1).float()
            
            # 3. Zero out the padding loss
            masked_loss = unreduced_loss * valid_mask
            
            # 4. Calculate the true mean over the valid elements
            loss = masked_loss.sum() / (valid_mask.sum() * latent_dim + 1e-8)
            
        total_loss += loss.item()
        steps += 1

    local_avg_loss = total_loss / max(1, steps)
    
    if torch.distributed.is_initialized():
        metrics = torch.tensor([local_avg_loss], device=device)
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
        metrics /= torch.distributed.get_world_size()
        global_avg_loss = metrics[0].item()
    else:
        global_avg_loss = local_avg_loss

    model.train()
    return global_avg_loss


if __name__ == "__main__":

    model = train(n_steps=N_STEPS)