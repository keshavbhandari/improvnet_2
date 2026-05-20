import os
import sys
import gc

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# Hugging Face
from transformers import get_scheduler

# Utilities
import json
import math
import random
import numpy as np
from tqdm import tqdm
from training_config import *
from improvnet.utils.utils import ProcessData, read_jsonl_files, setup_distributed

# --- MODIFICATION: Import the new ImprovNet model and config ---
# Make sure 'improvnet.model.model' points to the file with your latest ImprovNet class
from improvnet.model.model import AmortizedImprovNet, ImprovNetConfig, NUM_ATTRIBUTES

# -----------------------
# Dataset Class
# -----------------------
class ImprovnetDataset(Dataset):
    def __init__(self, data, split="train", train_type="pretraining"):
        self.data = data
        self.split = split
        if self.split == "train":
            random.seed(42)
            random.shuffle(self.data)
            self.apply_augmentation = True
        else:
            self.apply_augmentation = False
        self.train_type = train_type
        self.filter_data_by_training_type(self.train_type)
        print(f"Initializing {split} dataset with {len(self.data)} samples")
        self.processor = ProcessData()

    def __len__(self):
        return len(self.data)

    def on_epoch_end(self):
        if self.split == "train":
            random.shuffle(self.data)

    def filter_data_by_training_type(self, training_type):
        if training_type == "pretraining":
            # Example filter logic
            self.data = [item for item in self.data if "My Old FlameGM.mid" not in item['midi_filepath']]
        else:
            pass

    def __getitem__(self, idx):
        item = self.data[idx]
        filepath = item['midi_filepath']
        genre = item.get('genre', None)
        form = item.get('form', None)
        try:
            # Call the new Amortized Pipeline in utils.py
            # Returns 7 items: (enc_main, enc_accom, dec_main, dec_accom, genre, form, timestep)
            (
                enc_main, enc_accom, 
                dec_main, dec_accom, 
                genre_tok, form_tok, 
                timestep
            ) = self.processor.pretraining_pipeline(
                filepath, genre=genre, form=form, 
                segment_length=MAX_LEN,
                apply_pitch_augmentation=self.apply_augmentation
            )
            
            return {
                "enc_main": enc_main,
                "enc_accom": enc_accom,
                "dec_main": dec_main,
                "dec_accom": dec_accom,
                "genre": genre_tok,
                "form": form_tok,
                "timestep": timestep
            }
        except Exception as e:
            print(f"Error processing file {filepath}: {e}", file=sys.stderr)
            return None

# -----------------------
# Collate Function
# -----------------------
def collate_fn(batch):
    """
    Collates data for AmortizedImprovNet.
    Concatenates Main + Accom into [B, 2*L, 5] tensors for Encoder/Decoder inputs.
    Creates labels with -100 masking for padding.
    """
    # Filter failed samples
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
        
    ATTR_ORDER = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
    
    def _process_stream(token_dicts, pad_val=2):
        """Helper to stack dicts of tensors into a single [B, L, 5] tensor with padding."""
        inputs_list = []
        for token_dict in token_dicts:
            try:
                # Stack attributes [L, 5]
                input_tensor = torch.stack([token_dict[attr] for attr in ATTR_ORDER], dim=1)
            except (KeyError, RuntimeError) as e:
                print(f"Collate Error: {e}", file=sys.stderr)
                continue
                
            seq_len = input_tensor.shape[0]
            pad_len = MAX_LEN - seq_len
            
            if pad_len > 0:
                # Pad
                input_pad = torch.full((pad_len, len(ATTR_ORDER)), pad_val, dtype=torch.long)
                inputs_list.append(torch.cat([input_tensor, input_pad], dim=0))
            else:
                # Trim safety net
                inputs_list.append(input_tensor[:MAX_LEN])

        if not inputs_list:
            return None
        return torch.stack(inputs_list) # (B, MAX_LEN, 5)

    # 1. Process Encoder Streams (Main & Accom) -> Noisy Inputs
    enc_main_tens = _process_stream([item['enc_main'] for item in batch])
    enc_accom_tens = _process_stream([item['enc_accom'] for item in batch])
    
    # 2. Process Decoder Streams (Main & Accom) -> Clean Targets
    dec_main_tens = _process_stream([item['dec_main'] for item in batch])
    dec_accom_tens = _process_stream([item['dec_accom'] for item in batch])

    if enc_main_tens is None or dec_main_tens is None:
        return {}

    # 3. Concatenate [Main; Accom] -> (B, 2*MAX_LEN, 5)
    input_attributes_encoder = torch.cat([enc_main_tens, enc_accom_tens], dim=1)
    input_attributes_decoder = torch.cat([dec_main_tens, dec_accom_tens], dim=1)
    
    # 4. Create Labels (for Decoder)
    labels_main = dec_main_tens.clone()
    labels_main[labels_main == 2] = -100
    
    labels_accom = dec_accom_tens.clone()
    labels_accom[labels_accom == 2] = -100

    # 5. Metadata
    try:
        genre_tokens = torch.stack([item['genre'] for item in batch]).squeeze(1)
        form_tokens = torch.stack([item['form'] for item in batch]).squeeze(1)
        timesteps = torch.stack([item['timestep'] for item in batch]).squeeze(1)
    except Exception as e:
        print(f"Error collating meta: {e}", file=sys.stderr)
        return {}

    return {
        "input_attributes_encoder": input_attributes_encoder,
        "input_attributes_decoder": input_attributes_decoder,
        "labels_main": labels_main, 
        "labels_accom": labels_accom,
        "genre": genre_tokens,
        "form": form_tokens,
        "timestep": timesteps
    }


# ---------- Helper functions ----------
def save_checkpoint(model, optimizer, scheduler, epoch, metrics, ckpt_dir, rank):
    """Save model + optimizer + scheduler + metrics."""
    if rank != 0:
        return

    os.makedirs(ckpt_dir, exist_ok=True)

    # Save model & config (compatible with Hugging Face)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.save_pretrained(ckpt_dir)
    else:
        model.save_pretrained(ckpt_dir)

    # Save training states
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": metrics.get("best_val_loss", float("inf")),
            "ema_train_loss": metrics.get("ema_train_loss", None),
        },
        os.path.join(ckpt_dir, "training_state.pt")
    )
    
    print(f"[Rank 0] Saved checkpoint to {ckpt_dir}")


def load_checkpoint(ckpt_dir, model_cls, device, rank):
    """
    Load model + optimizer + scheduler + training metadata.
    Optimized to avoid OOM by NOT broadcasting the massive optimizer state across ranks.
    """
    print(f"[Rank {rank}] 🔍 Loading checkpoint from {ckpt_dir}")

    # --- Step 1: Synchronize Model Loading (Weights) ---
    # Rank 0 loads weights first for caching/safe_tensors optimization
    if rank != 0:
        dist.barrier()
    
    model = model_cls.from_pretrained(ckpt_dir).to(device)

    if rank == 0:
        dist.barrier()

    # --- Step 2: Handle Training State (Optimizer/Epoch) ---
    training_state_path = os.path.join(ckpt_dir, "training_state.pt")
    
    # Default values
    state = {}
    epoch = 0
    best_val_loss = float("inf")
    ema_loss = None

    if os.path.exists(training_state_path):
        # A. Broadcast ONLY tiny Metadata to all ranks
        if rank == 0:
            print(f"[Rank 0] Reading training state metadata...")
            temp_state = torch.load(training_state_path, map_location='cpu')
            
            epoch = temp_state.get("epoch", 0)
            best_val_loss = temp_state.get("best_val_loss", float("inf"))
            ema_loss = temp_state.get("ema_train_loss", None)
            
            metadata = [epoch, best_val_loss, ema_loss]
            del temp_state # Free memory
            gc.collect()   
        else:
            metadata = [None, None, None]

        dist.broadcast_object_list(metadata, src=0)
        epoch, best_val_loss, ema_loss = metadata
        
        # B. Load full state individually from disk on each rank
        # This prevents the massive RAM spike that occurs during state broadcasting.
        print(f"[Rank {rank}] Loading full training state from disk...")
        state = torch.load(training_state_path, map_location='cpu')
        print(f"[Rank {rank}] State loaded successfully.")
    else:
        print(f"[Rank {rank}] No 'training_state.pt' found. Starting fresh.")

    return model, state, epoch, best_val_loss, ema_loss


# ---------- Main ----------
def main(rank, local_rank, device, train_loader, val_loader, args):
    
    # Initialize processor to retrieve vocabulary sizes
    temp_processor = ProcessData()
    vocab_sizes = [
        len(temp_processor.tokenizer.tok_to_id_instrument),
        len(temp_processor.tokenizer.tok_to_id_pitch),
        len(temp_processor.tokenizer.tok_to_id_velocity),
        len(temp_processor.tokenizer.tok_to_id_onset),
        len(temp_processor.tokenizer.tok_to_id_duration)
    ]
    num_genres = len(temp_processor.tokenizer.tok_to_id_genre)
    num_forms = len(temp_processor.tokenizer.tok_to_id_form)
    del temp_processor # free memory

    config = ImprovNetConfig(
        hidden_size=args.embed_dim,
        num_heads=args.heads,
        num_decoder_layers=args.num_decoder_layers, 
        num_encoder_layers=args.num_encoder_layers,
        ffn_dim=args.embed_dim * args.mlp_mult,
        vocab_sizes=vocab_sizes,
        seq_len=args.max_seq_len, 
        num_genres=num_genres,
        num_forms=num_forms,
        no_bias=args.no_bias,
        gradient_checkpointing=args.grad_checkpointing
    )

    # Load from checkpoint if resuming
    if args.resume and os.path.exists(args.checkpoint_dir):
        model, training_state, start_epoch, best_val_loss, ema_loss = load_checkpoint(
            args.checkpoint_dir, AmortizedImprovNet, device, rank
        )
        print(f"[Rank {rank}] Resumed from epoch {start_epoch}")
    else:
        model = AmortizedImprovNet(config).to(device)
        training_state, start_epoch, best_val_loss, ema_loss = {}, 0, float("inf"), None

    print(f"[Rank {rank}] Model Parameters: {sum(p.numel() for p in model.parameters())}")

    # Wrap in DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Optimizer & Scheduler Setup
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = int(args.num_epochs * num_update_steps_per_epoch / args.grad_accum_steps)
    
    # Calculate warmup steps: use ratio but cap it with MAX_WARMUP_STEPS
    num_warmup_steps = min(int(WARMUP_RATIO * max_train_steps), MAX_WARMUP_STEPS)
    
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Restore States
    if "optimizer" in training_state:
        optimizer.load_state_dict(training_state["optimizer"])
        print(f"[Rank {rank}] Optimizer state restored.")

    if "scheduler" in training_state:
        # Restore scheduler state to resume from the correct point in the LR curve
        lr_scheduler.load_state_dict(training_state["scheduler"])
        print(f"[Rank {rank}] Scheduler state restored. Last step: {lr_scheduler.last_epoch}")
    
    del training_state
    gc.collect()                  
    torch.cuda.empty_cache()      

    scaler = torch.amp.GradScaler()
    tb_writer = SummaryWriter(args.tensorboard_dir) if rank == 0 and args.use_tensorboard else None

    # ---- Training Loop ----
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_sampler.set_epoch(epoch) # Critical for DDP shuffling

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", disable=(rank != 0), 
                                 leave=False, position=0, dynamic_ncols=True)
        for step, batch in enumerate(train_loader_tqdm):
            if not batch: 
                continue
                
            input_enc = batch["input_attributes_encoder"].to(device)
            input_dec = batch["input_attributes_decoder"].to(device)
            labels_main = batch["labels_main"].to(device)
            labels_accom = batch["labels_accom"].to(device)
            genre = batch['genre'].to(device)
            form = batch['form'].to(device)
            timestep = batch['timestep'].to(device)

            # Mixed precision forward pass
            with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                output = model(
                    input_attributes_encoder=input_enc,
                    input_attributes_decoder=input_dec,
                    genre=genre,
                    form=form,
                    timestep=timestep,
                    labels_main=labels_main,   
                    labels_accom=labels_accom, 
                    return_dict=True
                )
                loss = output["loss"]

            loss_val = loss.item()
            train_loss_sum += loss_val * input_enc.size(0)
            ema_loss = loss_val if ema_loss is None else (0.98 * ema_loss + 0.02 * loss_val)

            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step() # Move scheduler forward
                optimizer.zero_grad(set_to_none=True)

            if step % LOG_STEP == 0 and rank == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                train_loader_tqdm.set_postfix(loss=f"{loss_val:.4f}", ema=f"{ema_loss:.4f}", lr=f"{current_lr:.2e}")
                if args.use_tensorboard:
                    global_step = epoch * len(train_loader) + step
                    tb_writer.add_scalar("Train/StepLoss", loss_val, global_step)
                    tb_writer.add_scalar("Train/StepEMA", ema_loss, global_step)
                    tb_writer.add_scalar("Train/LearningRate", current_lr, global_step)

        # Force garbage collection
        del batch, input_enc, labels_main, input_dec, labels_accom, genre, form, timestep
        torch.cuda.empty_cache()

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0.0
        val_samples_sum = 0.0
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", disable=(rank != 0), 
                                 leave=False, position=0, dynamic_ncols=True)
        with torch.no_grad():
            for batch in val_loader_tqdm:
                if not batch:
                    continue
                
                input_enc = batch["input_attributes_encoder"].to(device)
                input_dec = batch["input_attributes_decoder"].to(device)
                labels_main = batch["labels_main"].to(device)
                labels_accom = batch["labels_accom"].to(device)
                genre = batch['genre'].to(device)
                form = batch['form'].to(device)
                timestep = batch['timestep'].to(device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = model(
                        input_attributes_encoder=input_enc,
                        input_attributes_decoder=input_dec,
                        genre=genre, form=form, timestep=timestep,
                        labels_main=labels_main, labels_accom=labels_accom,
                        return_dict=True
                    )
                    loss = output["loss"]
                val_loss_sum += loss.item() * input_enc.size(0)
                val_samples_sum += input_enc.size(0)

        # Reduce validation metrics across processes
        val_metrics_t = torch.tensor([val_loss_sum, val_samples_sum], device=device)
        dist.all_reduce(val_metrics_t, op=dist.ReduceOp.SUM)
        avg_val_loss_global = val_metrics_t[0].item() / (val_metrics_t[1].item() + 1e-9)

        if rank == 0:
            print(f"[Epoch {epoch+1}] EMA Train Loss: {ema_loss:.4f}, Val Loss: {avg_val_loss_global:.4f}")
            if args.use_tensorboard:
                tb_writer.add_scalar("Train/EpochEMA", ema_loss, epoch+1)
                tb_writer.add_scalar("Val/EpochLoss", avg_val_loss_global, epoch+1)

        # Save Best Checkpoint
        if rank == 0 and avg_val_loss_global < best_val_loss:
            best_val_loss = avg_val_loss_global
            metrics = {
                "epoch": epoch + 1,
                "ema_train_loss": ema_loss,
                "val_loss": avg_val_loss_global,
                "best_val_loss": best_val_loss,
            }
            save_checkpoint(model, optimizer, lr_scheduler, epoch + 1, metrics, args.checkpoint_dir, rank)

        # Final cleanup for epoch
        del batch, input_enc, labels_main, input_dec, labels_accom, genre, form, timestep
        torch.cuda.empty_cache()

    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":

    class Args:
        max_seq_len = MAX_LEN
        embed_dim = EMBED_DIM
        heads = HEADS
        mlp_mult = MLP_MULT
        num_decoder_layers = NUM_DECODER_LAYERS 
        num_encoder_layers = NUM_ENCODER_LAYERS         
        no_bias = NO_BIAS
        
        lr = LR
        num_epochs = NUM_EPOCHS
        grad_accum_steps = GRADIENT_ACCUMULATION_STEPS
        grad_checkpointing = GRADIENT_CHECKPOINTING
        checkpoint_dir = CHECKPOINTS_DIR
        tensorboard_dir = TENSORBOARD_LOG_DIR
        use_tensorboard = USE_TENSORBOARD
        resume = LOAD_FROM_CHECKPOINT
        train_type = TRAIN_TYPE

    args = Args()

    # --- Validating Configuration ---
    print("--- Validating Configuration ---")
    if args.heads % NUM_ATTRIBUTES != 0:
        print(f"🔴 CONFIG ERROR: HEADS ({args.heads}) must be divisible by NUM_ATTRIBUTES ({NUM_ATTRIBUTES}).")
        sys.exit(1)
    if args.embed_dim % NUM_ATTRIBUTES != 0:
        print(f"🔴 CONFIG ERROR: EMBED_DIM ({args.embed_dim}) must be divisible by NUM_ATTRIBUTES ({NUM_ATTRIBUTES}).")
        sys.exit(1)
    if args.embed_dim % args.heads != 0:
        print(f"🔴 CONFIG ERROR: EMBED_DIM ({args.embed_dim}) must be divisible by HEADS ({args.heads}).")
        sys.exit(1)
    print("✅ Config OK.")

    # Data collection
    train_files = read_jsonl_files(DATA_DIRS, split="train")
    validation_files = read_jsonl_files(DATA_DIRS, split="validation")
    
    if DEBUG:
        args.use_tensorboard = False
        train_files = train_files[:1000] if len(train_files) > 1000 else train_files
        validation_files = validation_files[:100] if len(validation_files) > 100 else validation_files

    # DDP Setup
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Dataset & Sampler initialization
    train_dataset = ImprovnetDataset(train_files, split="train", train_type=args.train_type)
    del train_files
    gc.collect()
    val_dataset = ImprovnetDataset(validation_files, split="validation", train_type=args.train_type)
    del validation_files
    gc.collect()
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE//4, 
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Start Training
    main(rank, local_rank, device, train_loader, val_loader, args)
