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
            self.data = [item for item in self.data if "My Old FlameGM.mid" not in item['midi_filepath']]
        else:
            pass

    def __getitem__(self, idx):
        item = self.data[idx]
        filepath = item['midi_filepath']
        genre = item.get('genre', None)
        form = item.get('form', None)
        try:
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
    batch = [item for item in batch if item is not None]
    if not batch: return {}
        
    ATTR_ORDER = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
    
    def _process_stream(token_dicts, pad_val=2):
        inputs_list = []
        for token_dict in token_dicts:
            try:
                input_tensor = torch.stack([token_dict[attr] for attr in ATTR_ORDER], dim=1)
            except (KeyError, RuntimeError):
                continue
            seq_len = input_tensor.shape[0]
            pad_len = MAX_LEN - seq_len
            if pad_len > 0:
                input_pad = torch.full((pad_len, len(ATTR_ORDER)), pad_val, dtype=torch.long)
                inputs_list.append(torch.cat([input_tensor, input_pad], dim=0))
            else:
                inputs_list.append(input_tensor[:MAX_LEN])
        return torch.stack(inputs_list) if inputs_list else None

    enc_main_tens = _process_stream([item['enc_main'] for item in batch])
    enc_accom_tens = _process_stream([item['enc_accom'] for item in batch])
    dec_main_tens = _process_stream([item['dec_main'] for item in batch])
    dec_accom_tens = _process_stream([item['dec_accom'] for item in batch])

    if enc_main_tens is None or dec_main_tens is None: return {}

    input_attributes_encoder = torch.cat([enc_main_tens, enc_accom_tens], dim=1)
    input_attributes_decoder = torch.cat([dec_main_tens, dec_accom_tens], dim=1)
    
    labels_main = dec_main_tens.clone()
    labels_main[labels_main == 2] = -100
    labels_accom = dec_accom_tens.clone()
    labels_accom[labels_accom == 2] = -100

    try:
        genre_tokens = torch.stack([item['genre'] for item in batch]).squeeze(1)
        form_tokens = torch.stack([item['form'] for item in batch]).squeeze(1)
        timesteps = torch.stack([item['timestep'] for item in batch]).squeeze(1)
    except Exception: return {}

    return {
        "input_attributes_encoder": input_attributes_encoder,
        "input_attributes_decoder": input_attributes_decoder,
        "labels_main": labels_main, 
        "labels_accom": labels_accom,
        "genre": genre_tokens,
        "form": form_tokens,
        "timestep": timesteps
    }

# -----------------------
# Checkpoint Helpers
# -----------------------
def save_checkpoint(model, optimizer, scheduler, epoch, metrics, ckpt_dir, rank):
    if rank != 0: return
    os.makedirs(ckpt_dir, exist_ok=True)
    if isinstance(model, DDP):
        model.module.save_pretrained(ckpt_dir)
    else:
        model.save_pretrained(ckpt_dir)
    
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": metrics.get("best_val_loss", float("inf")),
        "ema_train_loss": metrics.get("ema_train_loss", None),
    }, os.path.join(ckpt_dir, "training_state.pt"))
    print(f"[Rank 0] Saved checkpoint to {ckpt_dir}")

def load_checkpoint(ckpt_dir, model_cls, device, rank):
    if rank != 0: dist.barrier()
    model = model_cls.from_pretrained(ckpt_dir).to(device)
    if rank == 0: dist.barrier()

    training_state_path = os.path.join(ckpt_dir, "training_state.pt")
    state, epoch, best_val_loss, ema_loss = {}, 0, float("inf"), None

    if os.path.exists(training_state_path):
        if rank == 0:
            temp_state = torch.load(training_state_path, map_location='cpu')
            metadata = [temp_state.get("epoch", 0), temp_state.get("best_val_loss", float("inf")), temp_state.get("ema_train_loss", None)]
            del temp_state
        else:
            metadata = [None, None, None]
        dist.broadcast_object_list(metadata, src=0)
        epoch, best_val_loss, ema_loss = metadata
        state = torch.load(training_state_path, map_location='cpu')
    return model, state, epoch, best_val_loss, ema_loss

# -----------------------
# Main Process
# -----------------------
def main(rank, local_rank, device, train_loader, val_loader, args):
    # Initialize processor to get vocab metadata
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
    del temp_processor

    config = ImprovNetConfig(
        hidden_size=args.embed_dim, num_heads=args.heads,
        num_decoder_layers=args.num_decoder_layers, num_encoder_layers=args.num_encoder_layers,
        ffn_dim=args.embed_dim * args.mlp_mult, vocab_sizes=vocab_sizes,
        seq_len=args.max_seq_len, num_genres=num_genres, num_forms=num_forms,
        no_bias=args.no_bias, gradient_checkpointing=args.grad_checkpointing
    )

    # Resume Logic
    if args.resume and os.path.exists(args.checkpoint_dir):
        model, training_state, start_epoch, best_val_loss, ema_loss = load_checkpoint(
            args.checkpoint_dir, AmortizedImprovNet, device, rank
        )
        print(f"[Rank {rank}] Resumed from epoch {start_epoch}")
    else:
        model = AmortizedImprovNet(config).to(device)
        training_state, start_epoch, best_val_loss, ema_loss = {}, 0, float("inf"), None

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # --- LR Scheduler Calculation ---
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = int(args.num_epochs * num_update_steps_per_epoch / args.grad_accum_steps)
    
    # Calculate warmup steps: use min() to cap the warmup period regardless of dataset size
    num_warmup_steps = min(int(WARMUP_RATIO * max_train_steps), MAX_WARMUP_STEPS)
    
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps,
    )

    # Restore states from checkpoint
    if "optimizer" in training_state:
        optimizer.load_state_dict(training_state["optimizer"])
        print(f"[Rank {rank}] Optimizer state restored.")

    if "scheduler" in training_state:
        # Crucial for continuing the learning rate curve without restarting warmup
        lr_scheduler.load_state_dict(training_state["scheduler"])
        print(f"[Rank {rank}] Scheduler state restored. Continuing from step {lr_scheduler.last_epoch}.")
    
    del training_state
    gc.collect()
    torch.cuda.empty_cache()

    scaler = torch.amp.GradScaler()
    tb_writer = SummaryWriter(args.tensorboard_dir) if rank == 0 and args.use_tensorboard else None

    # ---- Training Loop ----
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", disable=(rank != 0), 
                                 leave=False, position=0, dynamic_ncols=True)
        
        for step, batch in enumerate(train_loader_tqdm):
            if not batch: continue
            
            input_enc = batch["input_attributes_encoder"].to(device)
            input_dec = batch["input_attributes_decoder"].to(device)
            labels_main = batch["labels_main"].to(device)
            labels_accom = batch["labels_accom"].to(device)
            genre, form, timestep = batch['genre'].to(device), batch['form'].to(device), batch['timestep'].to(device)

            with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                output = model(
                    input_attributes_encoder=input_enc, input_attributes_decoder=input_dec,
                    genre=genre, form=form, timestep=timestep,
                    labels_main=labels_main, labels_accom=labels_accom, return_dict=True
                )
                loss = output["loss"]

            loss_val = loss.item()
            ema_loss = loss_val if ema_loss is None else (0.98 * ema_loss + 0.02 * loss_val)
            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step() # Progress the scheduler
                optimizer.zero_grad(set_to_none=True)

            if step % LOG_STEP == 0 and rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                train_loader_tqdm.set_postfix(loss=f"{loss_val:.4f}", lr=f"{current_lr:.2e}")
                if args.use_tensorboard:
                    global_step = epoch * len(train_loader) + step
                    tb_writer.add_scalar("Train/LearningRate", current_lr, global_step)

        # ---- Validation ----
        model.eval()
        val_loss_sum, val_samples_sum = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                if not batch: continue
                input_enc = batch["input_attributes_encoder"].to(device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = model(
                        input_attributes_encoder=input_enc, 
                        input_attributes_decoder=batch["input_attributes_decoder"].to(device),
                        genre=batch['genre'].to(device), form=batch['form'].to(device), timestep=batch['timestep'].to(device),
                        labels_main=batch["labels_main"].to(device), labels_accom=batch["labels_accom"].to(device), return_dict=True
                    )
                val_loss_sum += output["loss"].item() * input_enc.size(0)
                val_samples_sum += input_enc.size(0)

        val_metrics_t = torch.tensor([val_loss_sum, val_samples_sum], device=device)
        dist.all_reduce(val_metrics_t, op=dist.ReduceOp.SUM)
        avg_val_loss = val_metrics_t[0].item() / (val_metrics_t[1].item() + 1e-9)

        if rank == 0:
            print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}, Best: {min(best_val_loss, avg_val_loss):.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, lr_scheduler, epoch + 1, 
                                {"best_val_loss": best_val_loss, "ema_train_loss": ema_loss}, args.checkpoint_dir, rank)

    if tb_writer: tb_writer.close()

if __name__ == "__main__":
    class Args:
        max_seq_len, embed_dim, heads, mlp_mult = MAX_LEN, EMBED_DIM, HEADS, MLP_MULT
        num_decoder_layers, num_encoder_layers, no_bias = NUM_DECODER_LAYERS, NUM_ENCODER_LAYERS, NO_BIAS
        lr, num_epochs, grad_accum_steps = LR, NUM_EPOCHS, GRADIENT_ACCUMULATION_STEPS
        grad_checkpointing, checkpoint_dir, tensorboard_dir = GRADIENT_CHECKPOINTING, CHECKPOINTS_DIR, TENSORBOARD_LOG_DIR
        use_tensorboard, resume, train_type = USE_TENSORBOARD, LOAD_FROM_CHECKPOINT, TRAIN_TYPE

    args = Args()
    train_files = read_jsonl_files(DATA_DIRS, split="train")
    validation_files = read_jsonl_files(DATA_DIRS, split="validation")
    
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    train_dataset = ImprovnetDataset(train_files, split="train", train_type=args.train_type)
    val_dataset = ImprovnetDataset(validation_files, split="validation", train_type=args.train_type)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE//4, sampler=val_sampler, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    main(rank, local_rank, device, train_loader, val_loader, args)

