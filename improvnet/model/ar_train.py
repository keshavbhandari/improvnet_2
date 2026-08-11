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
from improvnet.model.ar_config import *
from improvnet.model.ar_model import ARContextModel
from improvnet.utils.ar_utils import ProcessData
import contextlib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _amp_dtype():
    if not torch.cuda.is_available(): return None
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16

AMP_DTYPE = _amp_dtype()

class ARContextDataset(Dataset):
    def __init__(self, jsonl_files: list[str], processor: ProcessData, augment: bool = True):
        self.jsonl_files = jsonl_files
        self.processor = processor
        self.augment = augment
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
            if isinstance(event, list) and len(event) > 0 and all(x == event[0] for x in event) and isinstance(event[0], str):
                tokens.append(event[0])
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
                    if inst_val == 'drum':
                        tokens.append((inst_val, pitch_val))
                    else:
                        tokens.append((inst_val, pitch_val, vel_val))
                    tokens.append(('onset', onset_val))
                    tokens.append(('dur', dur_val))
            else:
                if isinstance(event, list):
                    tokens.append(tuple(event))
                else:
                    tokens.append(event)
        return tokens

    def __getitem__(self, idx):
        file_idx, offset = self.global_indices[idx]
        f = self._get_file_handle(self.jsonl_files[file_idx])
        f.seek(offset)
        line_bytes = f.readline()
        
        entry = json.loads(line_bytes.decode('utf-8'))
        tokens_raw = entry.get("tokens", [])
        tokens = self._lists_to_tuples(tokens_raw)
        
        genre_id = torch.tensor(self.processor.get_genre_id(entry.get("genre", "unknown")), dtype=torch.long)
        
        if self.augment:
            tokens = self.processor.pitch_augmentation(tokens)

        if len(tokens) == 0: tokens = ['<S>']
        
        # We need SEQ_LEN + 1 tokens total to create [0:T] inputs and [1:T+1] targets
        total_needed = SEQ_LEN + 1
        
        if len(tokens) > total_needed:
            start_idx = random.randint(0, len(tokens) - total_needed)
            sliced_tokens = tokens[start_idx : start_idx + total_needed]
        else:
            sliced_tokens = tokens
            
        multi_hot = self.processor.get_instrument_multihot(sliced_tokens)
        tensor_seq = self.processor.format_variable_sequence(sliced_tokens, total_needed, pad_id=PAD_ID)

        # Standard Next-Token Prediction Offset
        input_tensor = tensor_seq[:-1]
        target_tensor = tensor_seq[1:]

        return {
            "input": input_tensor,
            "target": target_tensor,
            "genre": genre_id,
            "multi_hot": multi_hot
        }

def build_dataloader(jsonl_files: list[str], split: str, batch_size: int, augment: bool = False, num_workers: int = 4, distributed: bool = False) -> DataLoader:
    processor = ProcessData()
    dataset = ARContextDataset(jsonl_files=jsonl_files, processor=processor, augment=augment)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    
    def collate_fn(batch):
        return {
            "input": torch.stack([b["input"] for b in batch]),
            "target": torch.stack([b["target"] for b in batch]),
            "genre": torch.stack([b["genre"] for b in batch]),
            "multi_hot": torch.stack([b["multi_hot"] for b in batch])
        }
        
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(not distributed),
        sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn
    )

def resume_epoch_loader(loader: DataLoader, start_batch_idx: int) -> DataLoader:
    if start_batch_idx <= 0:
        return loader
    if loader.sampler is None:
        return loader

    start_sample_idx = start_batch_idx * loader.batch_size
    remaining_indices = list(loader.sampler)[start_sample_idx:]

    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=remaining_indices,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        collate_fn=loader.collate_fn
    )

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, scheduler, step, best_val_loss, epoch, next_batch_idx, micro_step, scaler=None, is_best=False):
    os.makedirs(SAVE_DIR, exist_ok=True)
    model_to_save = model.module if isinstance(model, DDP) else model
    checkpoint = {
        'step': step, 
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'epoch': epoch,
        'next_batch_idx': next_batch_idx,
        'micro_step': micro_step
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    torch.save(checkpoint, os.path.join(SAVE_DIR, "latest_checkpoint.pt"))
    if is_best:
        torch.save(checkpoint, os.path.join(SAVE_DIR, "best_model.pt"))

def _checkpoint_base_lrs(checkpoint):
    scheduler_state = checkpoint.get('scheduler_state_dict', {})
    if 'base_lrs' in scheduler_state:
        return scheduler_state['base_lrs']
    return [
        group.get('initial_lr', group['lr'])
        for group in checkpoint['optimizer_state_dict']['param_groups']
    ]

def _lrs_match_config(checkpoint_lrs, config_lr: float) -> bool:
    return all(math.isclose(lr, config_lr, rel_tol=1e-12, abs_tol=1e-16) for lr in checkpoint_lrs)

def load_checkpoint(model, optimizer, scheduler, device, scaler=None):
    checkpoint_path = os.path.join(SAVE_DIR, "latest_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        return 0, float('inf'), False, False, None, None
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    has_resume_position = all(
        key in checkpoint for key in ('epoch', 'next_batch_idx', 'micro_step')
    )
    resume_state = {
        'has_resume_position': has_resume_position,
        'epoch': checkpoint.get('epoch'),
        'next_batch_idx': checkpoint.get('next_batch_idx'),
        'micro_step': checkpoint.get('micro_step')
    }
    model_to_load = model.module if isinstance(model, DDP) else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    checkpoint_lrs = _checkpoint_base_lrs(checkpoint)
    lr_changed = not _lrs_match_config(checkpoint_lrs, LR)
    scheduler_restored = 'scheduler_state_dict' in checkpoint
    resume_start_lrs = [group['lr'] for group in optimizer.param_groups]

    if scheduler_restored and not lr_changed:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group['lr'] = lr

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    step = checkpoint['step']
    best_val_loss = checkpoint['best_val_loss']
    del checkpoint
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return step, best_val_loss, scheduler_restored, lr_changed, resume_start_lrs, resume_state

def build_scheduler(optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps: return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def build_resume_warmup_scheduler(
    optimizer,
    start_lrs: list[float],
    target_lr: float,
    warmup_steps: int,
    total_steps: int,
    resume_step: int
) -> LambdaLR:
    target_lrs = [target_lr for _ in optimizer.param_groups]
    start_factors = [
        start_lr / target_lr if target_lr > 0 else 1.0
        for start_lr in start_lrs
    ]
    
    for group, lr in zip(optimizer.param_groups, target_lrs):
        group['lr'] = lr
        group['initial_lr'] = lr

    def make_lr_lambda(start_factor: float):
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                alpha = step / max(1, warmup_steps)
                return start_factor + (1.0 - start_factor) * alpha
            progress = (step - warmup_steps) / max(1, total_steps - resume_step - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_lambda

    return LambdaLR(optimizer, [make_lr_lambda(factor) for factor in start_factors])

def train(n_steps: int = N_STEPS):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (local_rank == 0)

    writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, "runs", RUN_NAME)) if is_main_process else None

    train_loader = build_dataloader(JSONL_FILES, "train", BATCH_SIZE, augment=True, distributed=True)
    val_loader = build_dataloader(JSONL_FILES, "validation", BATCH_SIZE, augment=False, distributed=True)

    if is_main_process: print("Initializing AR Context Pretraining Tower...")
    model = ARContextModel().to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(AMP_DTYPE == torch.float16))

    update_step, best_val_loss = 0, float('inf')
    resume_state = None
    scheduler = build_scheduler(optimizer, WARMUP_STEPS, n_steps)
    if RESUME_TRAINING:
        update_step, best_val_loss, scheduler_restored, lr_changed, resume_start_lrs, resume_state = load_checkpoint(model, optimizer, scheduler, device, scaler)
        if lr_changed and resume_start_lrs is not None:
            scheduler = build_resume_warmup_scheduler(
                optimizer, resume_start_lrs, LR, WARMUP_STEPS, n_steps, update_step
            )
        elif not scheduler_restored:
            scheduler = build_scheduler(optimizer, WARMUP_STEPS, n_steps, last_epoch=update_step - 1 if update_step > 0 else -1)
    
    has_resume_position = resume_state is not None and resume_state['has_resume_position']
    micro_step = resume_state['micro_step'] if has_resume_position else update_step * ACCUM_STEPS
    start_epoch = resume_state['epoch'] if has_resume_position else micro_step // len(train_loader)
    resume_batch_idx = resume_state['next_batch_idx'] if has_resume_position else 0

    if is_main_process and resume_state is not None:
        if has_resume_position:
            print(f"Resuming from step {update_step}, epoch {start_epoch}, batch {resume_batch_idx}.")
        else:
            print(f"Resuming legacy checkpoint from step {update_step}; starting at epoch {start_epoch}, batch 0.")
    
    running_loss, log_steps = 0.0, 0
    model.train()
    epoch = start_epoch

    while update_step < n_steps:
        if hasattr(train_loader.sampler, 'set_epoch'): train_loader.sampler.set_epoch(epoch)
        epoch_loader = resume_epoch_loader(train_loader, resume_batch_idx)
        
        for batch_idx, batch in enumerate(epoch_loader, start=resume_batch_idx):
            if update_step >= n_steps: break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            is_accumulating = (micro_step + 1) % ACCUM_STEPS != 0
            ddp_context = model.no_sync() if is_accumulating else contextlib.nullcontext()
            
            with ddp_context:
                with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                    
                    logits = model(
                        target=batch["input"],          
                        genre=batch["genre"],
                        multi_hot=batch["multi_hot"]
                    )
                    
                    # Standard Auto-Regressive Cross Entropy calculation
                    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
                    loss = criterion(logits.view(-1, VOCAB_SIZE), batch["target"].view(-1)) / ACCUM_STEPS

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
                    perplexity = math.exp(min(avg_loss, 20.0))
                    print(f"  Step {update_step:>6} | CE Loss: {avg_loss:>6.4f} | PPL: {perplexity:>6.2f}")
                    writer.add_scalar('Train/AR_CE_Loss', avg_loss, update_step)
                    writer.add_scalar('Train/AR_Perplexity', perplexity, update_step)
                    writer.add_scalar('Hyperparameters/LR', optimizer.param_groups[0]['lr'], update_step)
                    running_loss, log_steps = 0.0, 0

                if update_step > 0 and update_step % VAL_EVERY == 0:
                    val_loss = evaluate_validation(model, val_loader, local_rank, device)
                    if is_main_process:
                        val_ppl = math.exp(min(val_loss, 20.0))
                        print(f"\n--- Validation Step {update_step} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} ---")
                        writer.add_scalar('Validation/AR_CE_Loss', val_loss, update_step)
                        is_best = val_loss < best_val_loss
                        if is_best: best_val_loss = val_loss
                        next_batch_idx = batch_idx + 1
                        next_epoch = epoch
                        if next_batch_idx >= len(train_loader):
                            next_epoch += 1
                            next_batch_idx = 0
                        save_checkpoint(model, optimizer, scheduler, update_step, best_val_loss, next_epoch, next_batch_idx, micro_step, scaler, is_best)
        resume_batch_idx = 0
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
                genre=batch["genre"],
                multi_hot=batch["multi_hot"]
            )
            
            criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
            loss = criterion(logits.view(-1, VOCAB_SIZE), batch["target"].view(-1))
            
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
