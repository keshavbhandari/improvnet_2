import os
import sys

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
from improvnet.model.model_with_cache import ImprovNet, ImprovNetConfig, NUM_ATTRIBUTES

# -----------------------
# Audio Text Dataset
# (This class is unchanged as its output is what we'll adapt to)
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
            self.data = [item for item in self.data if "My Old FlameGM.mid" not in item['midi_filepath'] and "Whilewereyoung.mid" not in item['midi_filepath']]
        elif training_type == "finetuning":
            # Finetuning is not implemented in this script
            pass
        else:
            raise ValueError(f"Unknown training type: {training_type}")

    def __getitem__(self, idx):
        item = self.data[idx]
        filepath = item['midi_filepath']
        genre = item.get('genre', None)
        form = item.get('form', None)
        try:
            if self.train_type == "pretraining":
                (
                    corrupted_tokens, corrupted_accomp_tokens, 
                    changed_indices, changed_accomp_indices, 
                    original_tokens, accomp_tokens, 
                    genre_token, form_token
                ) = self.processor.pretraining_pipeline(
                    filepath, genre=genre, form=form, 
                    corruption_type='random',
                    segment_length=MAX_LEN,
                    mask_token='<MASK>',
                    apply_pitch_augmentation=self.apply_augmentation
                )
            elif self.train_type == "finetuning":
                # This script is set for pretraining, but you would call your
                # finetuning pipeline here if TRAIN_TYPE was 'finetuning'
                raise NotImplementedError("Finetuning pipeline not fully implemented in this script")
            else:
                raise ValueError(f"Unknown training type: {self.train_type}")
        except Exception as e:
            print(f"Error processing file {filepath}: {e}", file=sys.stderr)
            # Return None to be filtered by collate_fn
            return None

        # Return all parts from the pipeline
        return {
            "corrupted_input": corrupted_tokens,
            "corrupted_accomp_input": corrupted_accomp_tokens,
            "changed_indices": changed_indices,
            "changed_accomp_indices": changed_accomp_indices,
            "original_input": original_tokens,
            "original_accomp_input": accomp_tokens,
            "genre": genre_token,
            "form": form_token
        }

# -----------------------
# MODIFICATION: New Collate Function
# -----------------------
def collate_fn(batch):
    """
    Collates data from the ImprovnetDataset for the new ImprovNet model.
    Handles padding/trimming and dictionary-to-tensor conversion.
    """
    # Filter out None values from failed __getitem__ calls
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
        
    # --- Attribute order MUST match the model's input ---
    # [instrument, pitch, velocity, onset, duration]
    ATTR_ORDER = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
    
    def _process_stream(token_dicts, label_dicts, mask_tensors):
        """Helper to process one stream (main or accom)."""
        inputs_list = []
        labels_list = []
        masks_list = []

        for token_dict, label_dict, mask in zip(token_dicts, label_dicts, mask_tensors):
            # 1. Convert dict of tensors to (L, 5) tensor
            # This is crucial: it stacks the individual attribute tensors
            try:
                input_tensor = torch.stack([token_dict[attr] for attr in ATTR_ORDER], dim=1)
                label_tensor = torch.stack([label_dict[attr] for attr in ATTR_ORDER], dim=1)
            except KeyError as e:
                print(f"Collate Error: Missing key {e}. Token dict keys: {token_dict.keys()}", file=sys.stderr)
                continue
            except RuntimeError as e:
                # This happens if tensors in the dict have different lengths
                print(f"Collate Error: Mismatched tensor lengths. {e}", file=sys.stderr)
                print(f"Lengths: {[len(token_dict[attr]) for attr in ATTR_ORDER]}", file=sys.stderr)
                continue
                
            # 2. Get length and compute padding
            seq_len = input_tensor.shape[0]
            pad_len = MAX_LEN - seq_len
            
            # 3. Pad or trim all three tensors
            if pad_len > 0:
                # Pad inputs with 2 (assuming 2 is a <PAD> token ID)
                input_pad = torch.full((pad_len, len(ATTR_ORDER)), 2, dtype=torch.long)
                inputs_list.append(torch.cat([input_tensor, input_pad], dim=0))
                
                # Pad labels with -100 (ignored by loss function)
                label_pad = torch.full((pad_len, len(ATTR_ORDER)), -100, dtype=torch.long)
                labels_list.append(torch.cat([label_tensor, label_pad], dim=0))
                
                # Pad mask with False (0)
                mask_pad = torch.full((pad_len, len(ATTR_ORDER)), False, dtype=torch.bool)
                masks_list.append(torch.cat([mask, mask_pad], dim=0))
                
            else: # seq_len >= MAX_LEN, so we trim
                inputs_list.append(input_tensor[:MAX_LEN])
                labels_list.append(label_tensor[:MAX_LEN])
                masks_list.append(mask[:MAX_LEN])

        if not inputs_list:
            return None, None, None

        # 4. Stack into a batch
        return (
            torch.stack(inputs_list),  # (B, MAX_LEN, 5)
            torch.stack(labels_list),  # (B, MAX_LEN, 5)
            torch.stack(masks_list)    # (B, MAX_LEN, 5)
        )

    # --- Process Main Stream ---
    main_inputs, main_labels, main_masks = _process_stream(
        [item['corrupted_input'] for item in batch],
        [item['original_input'] for item in batch],
        [item['changed_indices'] for item in batch]
    )

    # --- Process Accompaniment Stream ---
    accom_inputs, accom_labels, accom_masks = _process_stream(
        [item['corrupted_accomp_input'] for item in batch],
        [item['original_accomp_input'] for item in batch],
        [item['changed_accomp_indices'] for item in batch]
    )
    
    # If any stream failed (e.g., all items in batch had errors for that stream)
    if main_inputs is None or accom_inputs is None:
        return {}

    # --- Process Genre/Form Tokens ---
    # .squeeze(1) to remove the (1,) dim from the util function
    try:
        genre_tokens = torch.stack([item['genre'] for item in batch]).squeeze(1)
        form_tokens = torch.stack([item['form'] for item in batch]).squeeze(1)
    except Exception as e:
        print(f"Error collating genre/form tokens: {e}", file=sys.stderr)
        return {}

    return {
        "input_attributes_main": main_inputs,
        "labels_main": main_labels,
        "loss_mask_main": main_masks,
        "input_attributes_accom": accom_inputs,
        "labels_accom": accom_labels,
        "loss_mask_accom": accom_masks,
        "genre": genre_tokens,
        "form": form_tokens
    }


# ---------- Helper functions ----------
def save_checkpoint(model, optimizer, scheduler, epoch, metrics, ckpt_dir, rank):
    """Save model + optimizer + scheduler + metrics."""
    if rank != 0:
        return

    os.makedirs(ckpt_dir, exist_ok=True)

    # Save model & config (like Hugging Face)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.save_pretrained(ckpt_dir)
    else:
        model.save_pretrained(ckpt_dir)

    # Save optimizer/scheduler state and epoch metadata
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
    """Load model + optimizer + scheduler + training metadata."""
    print(f"[Rank {rank}] 🔍 Loading checkpoint from {ckpt_dir}")

    # --- Step 1: Synchronize Model Loading ---
    if rank != 0:
        dist.barrier()
    
    model = model_cls.from_pretrained(ckpt_dir).to(device)

    if rank == 0:
        dist.barrier()

    # --- Step 2: Load Training State (DDP-safe) ---
    training_state_path = os.path.join(ckpt_dir, "training_state.pt")
    
    if os.path.exists(training_state_path):
        if rank == 0:
            print(f"[Rank 0] Loading 'training_state.pt' from disk...")
            state = torch.load(training_state_path, map_location='cpu')
            obj_list = [state]
        else:
            obj_list = [None]
        
        dist.broadcast_object_list(obj_list, src=0)
        state = obj_list[0]
        
        epoch = state.get("epoch", 0)
        best_val_loss = state.get("best_val_loss", float("inf"))
        ema_loss = state.get("ema_train_loss", None)
        print(f"[Rank {rank}] Successfully loaded training state in memory.")
    
    else:
        print(f"[Rank {rank}] No 'training_state.pt' found. Starting fresh.")
        state, epoch, best_val_loss, ema_loss = {}, 0, float("inf"), None

    return model, state, epoch, best_val_loss, ema_loss


# ---------- Main ----------
def main(rank, local_rank, device, train_loader, val_loader, args):
    
    # --- MODIFICATION: Create config for the NEW ImprovNet ---
    # We load vocab sizes from the tokenizer processor
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
        num_layers=args.num_layers,
        ffn_dim=args.embed_dim * args.mlp_mult,
        vocab_sizes=vocab_sizes,      # This is the list of 5 vocabs
        seq_len=args.max_seq_len,     # This is L (e.g., 2048)
        num_genres=num_genres,
        num_forms=num_forms,
        no_bias=args.no_bias,
        gradient_checkpointing=args.grad_checkpointing
    )

    # Load model if resuming
    if args.resume and os.path.exists(args.checkpoint_dir):
        model, training_state, start_epoch, best_val_loss, ema_loss = load_checkpoint(
            args.checkpoint_dir, ImprovNet, device, rank
        )
        print(f"[Rank {rank}] Resumed from epoch {start_epoch + 1}")
    else:
        model = ImprovNet(config).to(device)
        training_state, start_epoch, best_val_loss, ema_loss = {}, 0, float("inf"), None

    print(f"[Rank {rank}] Model has {sum(p.numel() for p in model.parameters())} parameters")

    # DDP setup
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = int(args.num_epochs * num_update_steps_per_epoch / args.grad_accum_steps)
    num_warmup_steps = int(0.01 * max_train_steps)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Restore optimizer/scheduler states
    if "optimizer" in training_state:
        optimizer.load_state_dict(training_state["optimizer"])
    if "scheduler" in training_state:
        lr_scheduler.load_state_dict(training_state["scheduler"])
    del training_state

    scaler = torch.amp.GradScaler()
    tb_writer = SummaryWriter(args.tensorboard_dir) if rank == 0 and args.use_tensorboard else None

    # ---- Training Loop ----
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_sampler.set_epoch(epoch) # Important for DDP shuffling

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", disable=(rank != 0), 
                                 leave=False, position=0, dynamic_ncols=True)
        for step, batch in enumerate(train_loader_tqdm):
            if not batch: # Skip empty batches if collate_fn filtered all items
                continue
                
            # --- MODIFICATION: Load all data from collate_fn ---
            input_main = batch["input_attributes_main"].to(device)
            labels_main = batch["labels_main"].to(device)
            mask_main = batch["loss_mask_main"].to(device)
            input_accom = batch["input_attributes_accom"].to(device)
            labels_accom = batch["labels_accom"].to(device)
            mask_accom = batch["loss_mask_accom"].to(device)
            genre = batch['genre'].to(device)
            form = batch['form'].to(device)

            # --- MODIFICATION: New model forward pass ---
            with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                output = model(
                    input_attributes_main=input_main,
                    input_attributes_accom=input_accom,
                    genre=genre,
                    form=form,
                    labels_main=labels_main,
                    labels_accom=labels_accom,
                    loss_mask_main=mask_main,
                    loss_mask_accom=mask_accom,
                    return_dict=True
                )
                loss = output["loss"]

            loss_val = loss.item()
            train_loss_sum += loss_val * input_main.size(0)
            ema_loss = loss_val if ema_loss is None else (0.98 * ema_loss + 0.02 * loss_val)

            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % LOG_STEP == 0 and rank == 0:
                train_loader_tqdm.set_postfix(loss=f"{loss_val:.4f}", ema_loss=f"{ema_loss:.4f}")
                if args.use_tensorboard:
                    global_step = epoch * len(train_loader) + step
                    tb_writer.add_scalar("Train/StepLoss", loss_val, global_step)
                    tb_writer.add_scalar("Train/StepEMA", ema_loss, global_step)
                    tb_writer.add_scalar("Train/LearningRate", optimizer.param_groups[0]["lr"], global_step)

        # Clear memory
        del batch, input_main, labels_main, mask_main, input_accom, labels_accom, mask_accom, genre, form
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
                
                # --- MODIFICATION: Load all data from collate_fn ---
                input_main = batch["input_attributes_main"].to(device)
                labels_main = batch["labels_main"].to(device)
                mask_main = batch["loss_mask_main"].to(device)
                input_accom = batch["input_attributes_accom"].to(device)
                labels_accom = batch["labels_accom"].to(device)
                mask_accom = batch["loss_mask_accom"].to(device)
                genre = batch['genre'].to(device)
                form = batch['form'].to(device)

                # --- MODIFICATION: New model forward pass ---
                with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                    output = model(
                        input_attributes_main=input_main,
                        input_attributes_accom=input_accom,
                        genre=genre,
                        form=form,
                        labels_main=labels_main,
                        labels_accom=labels_accom,
                        loss_mask_main=mask_main,
                        loss_mask_accom=mask_accom,
                        return_dict=True
                    )
                    loss = output["loss"]
                val_loss_sum += loss.item() * input_main.size(0)
                val_samples_sum += input_main.size(0)

        # Reduce metrics
        val_metrics_t = torch.tensor([val_loss_sum, val_samples_sum], device=device)
        dist.all_reduce(val_metrics_t, op=dist.ReduceOp.SUM)
        avg_val_loss_global = val_metrics_t[0].item() / (val_metrics_t[1].item() + 1e-9)

        # ---- Logging ----
        if rank == 0:
            print(f"[Epoch {epoch+1}] EMA Train Loss: {ema_loss:.4f}, Val Loss: {avg_val_loss_global:.4f}")
            if args.use_tensorboard:
                tb_writer.add_scalar("Train/EpochEMA", ema_loss, epoch+1)
                tb_writer.add_scalar("Val/EpochLoss", avg_val_loss_global, epoch+1)

        # ---- Save best ----
        if rank == 0 and avg_val_loss_global < best_val_loss:
            best_val_loss = avg_val_loss_global
            metrics = {
                "epoch": epoch + 1,
                "ema_train_loss": ema_loss,
                "val_loss": avg_val_loss_global,
                "best_val_loss": best_val_loss,
            }
            save_checkpoint(model, optimizer, lr_scheduler, epoch + 1, metrics, args.checkpoint_dir, rank)

        # Free memory
        del batch, input_main, labels_main, mask_main, input_accom, labels_accom, mask_accom, genre, form
        torch.cuda.empty_cache()

    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":

    class Args:
        max_seq_len = MAX_LEN
        embed_dim = EMBED_DIM
        heads = HEADS
        mlp_mult = MLP_MULT
        num_layers = NUM_LAYERS # Corrected from num_global_layers
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

    # Add validation for 6D MRA config ---
    print("--- Validating Configuration ---")
    if args.heads % NUM_ATTRIBUTES != 0:
        print(f"🔴 CONFIG ERROR: HEADS ({args.heads}) must be divisible by NUM_ATTRIBUTES ({NUM_ATTRIBUTES}).")
        print(f"Please change HEADS in training_config.py to a multiple of 6 (e.g., 12, 18).")
        sys.exit(1)
    if args.embed_dim % NUM_ATTRIBUTES != 0:
        print(f"🔴 CONFIG ERROR: EMBED_DIM ({args.embed_dim}) must be divisible by NUM_ATTRIBUTES ({NUM_ATTRIBUTES}).")
        print(f"Please change EMBED_DIM in training_config.py to a multiple of 6 (e.g., 1020, 780).")
        sys.exit(1)
    if args.embed_dim % args.heads != 0:
        print(f"🔴 CONFIG ERROR: EMBED_DIM ({args.embed_dim}) must be divisible by HEADS ({args.heads}).")
        sys.exit(1)
    print("✅ Config OK.")
    # --- End of validation ---

    # Read all jsonl files in the DATA_DIRS
    train_files = read_jsonl_files(DATA_DIRS, split="train")
    validation_files = read_jsonl_files(DATA_DIRS, split="validation")
    
    if DEBUG:
        args.use_tensorboard = False
        train_files = train_files[:1000]
        validation_files = validation_files[:100]

    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Create datasets and dataloaders
    train_dataset = ImprovnetDataset(train_files, split="train", train_type=args.train_type)
    val_dataset = ImprovnetDataset(validation_files, split="validation", train_type=args.train_type)
    
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

    # Start training
    main(rank, local_rank, device, train_loader, val_loader, args)