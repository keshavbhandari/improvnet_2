import math
import random
import os
import json
import pickle
import contextlib
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import improvnet.model.twotower_config as twotower_config
from improvnet.model.twotower_config import *
from improvnet.utils.ar_utils import (
    ProcessData,
    build_optimizer,
    collect_rng_state_for_checkpoint,
    distributed_rank_world,
    load_checkpoint,
    load_training_checkpoint,
    resume_epoch_loader,
    save_checkpoint,
)
from improvnet.model.ar_model import ARContextModel
from improvnet.model.twotower_denoiser import TwoTowerDenoiser

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _amp_dtype():
    if not torch.cuda.is_available(): return None
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16

AMP_DTYPE = _amp_dtype()

class TwoTowerDataset(Dataset):
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
        
        genre_id = self.processor.get_genre_id(entry.get("genre", "unknown"))
        
        if self.augment:
            tokens = self.processor.pitch_augmentation(tokens)

        # Enforce structural integrity: Add <E> if missing, but strictly to the VERY end of the master sequence
        if len(tokens) > 0 and tokens[-1] != '<E>':
            tokens.append('<E>')
            
        if len(tokens) == 0: tokens = ['<S>', '<E>']
        
        total_needed = PROMPT_MAX + BLOCK_SIZE
        
        if len(tokens) > total_needed:
            start_idx = random.randint(0, len(tokens) - total_needed)
            sliced_tokens = tokens[start_idx : start_idx + total_needed]
        else:
            sliced_tokens = tokens

        prefix_tokens = sliced_tokens[:PROMPT_MAX]
        target_tokens = sliced_tokens[PROMPT_MAX:PROMPT_MAX + BLOCK_SIZE]
        
        multi_hot = self.processor.get_instrument_multihot(sliced_tokens)

        return {
            "prefix": prefix_tokens,
            "target": target_tokens,
            "genre": genre_id,
            "multi_hot": multi_hot
        }

    def collate_fn(self, batch):
        padded_prefixes = []
        padded_traj_inputs = []
        padded_traj_targets = []
        padded_traj_ts = []
        padded_traj_wts = []
        genres = []
        multi_hots = []

        for item in batch:
            prefix_tokens = item["prefix"]
            target_tokens = item["target"]
            
            genres.append(torch.tensor(item["genre"], dtype=torch.long))
            multi_hots.append(item["multi_hot"])

            active_insts_names = set()
            for tok in target_tokens:
                if isinstance(tok, tuple) and len(tok) in (2, 3) and tok[0] in self.processor.INSTRUMENT_CLASSES:
                    active_insts_names.add(tok[0])
            is_multi_track = len(active_insts_names) > 1

            # 1. Elasticity on target labels (80% chance for multi-track, 30% for solo piano)
            elasticity_prob = 0.8 if is_multi_track else 0.3
            if random.random() < elasticity_prob:
                # Slightly increased the maximum possible insertions for multi-track pieces
                num_insertions = random.randint(0, int(len(target_tokens) * 0.10) + 1)
                for _ in range(num_insertions):
                    chunk_size = 3 if random.random() < 0.8 else 1
                    idx = random.randint(0, len(target_tokens))
                    for _ in range(chunk_size):
                        target_tokens.insert(idx, '<BLANK>')

            prefix_tensor = self.processor.format_variable_sequence(prefix_tokens, PROMPT_MAX, pad_id=PAD_ID)
            target_tensor = self.processor.format_variable_sequence(target_tokens, BLOCK_SIZE, pad_id=PAD_ID)

            valid_indices = (target_tensor != PAD_ID).nonzero(as_tuple=True)[0]
            L_valid = len(valid_indices)

            traj_input = []
            traj_target = []
            traj_ts = []
            traj_wt = []

            # Create random descending timesteps for a dynamic non-Markovian trajectory
            steps = sorted([random.randint(0, DIFFUSION_STEPS) for _ in range(NUM_DRAFTS)], reverse=True)
            
            for i, s in enumerate(steps):
                t_val = max(0.0, s / float(DIFFUSION_STEPS))
                draft_input = target_tensor.clone()
                draft_wt = torch.zeros_like(target_tensor, dtype=torch.float32)

                if L_valid > 0:
                    r = torch.rand(L_valid)
                    do_mask = r > (1.0 - t_val)
                    do_clean = ~do_mask

                    idx_mask = valid_indices[do_mask]
                    idx_clean = valid_indices[do_clean]
                    
                    draft_input[idx_mask] = MASK_ID
                    
                    # 2. Targeted Instrument Stripping (80% chance for multi-track, 0% for solo)
                    strip_prob = 0.8 if is_multi_track else 0.0
                    if random.random() < strip_prob:
                        active_insts = set()
                        for v_idx in valid_indices:
                            tok_id = target_tensor[v_idx].item()
                            tok_str = self.processor.tokenizer.id_to_tok.get(tok_id)
                            if isinstance(tok_str, tuple) and len(tok_str) in (2,3) and tok_str[0] in self.processor.INSTRUMENT_CLASSES:
                                active_insts.add(tok_str[0])
                        
                        if len(active_insts) > 1:
                            strip_inst = random.choice(list(active_insts))
                            skip_count = 0
                            for v_idx in valid_indices:
                                if skip_count > 0:
                                    draft_input[v_idx] = MASK_ID
                                    skip_count -= 1
                                    continue
                                    
                                tok_id = target_tensor[v_idx].item()
                                tok_str = self.processor.tokenizer.id_to_tok.get(tok_id)
                                if isinstance(tok_str, tuple) and len(tok_str) in (2,3) and tok_str[0] == strip_inst:
                                    draft_input[v_idx] = MASK_ID
                                    skip_count = 2 # Also mask the subsequent onset and duration!

                    # Weights: Low emphasis on clean notes, high emphasis on hallucinating masked notes
                    draft_wt[valid_indices] = 0.05 
                    draft_wt[draft_input == MASK_ID] = 2.0 

                traj_input.append(draft_input)
                traj_target.append(target_tensor)
                
                t_tensor = torch.full((BLOCK_SIZE,), t_val, dtype=torch.float32)
                traj_ts.append(t_tensor)
                traj_wt.append(draft_wt)
                
                # Append <SEP> delimiter
                if i < NUM_DRAFTS - 1:
                    traj_input.append(torch.tensor([SEP_ID], dtype=torch.long))
                    traj_target.append(torch.tensor([PAD_ID], dtype=torch.long)) # <SEP> target is ignored
                    traj_ts.append(torch.tensor([t_val], dtype=torch.float32))
                    traj_wt.append(torch.tensor([0.0], dtype=torch.float32))

            padded_prefixes.append(prefix_tensor)
            padded_traj_inputs.append(torch.cat(traj_input))
            padded_traj_targets.append(torch.cat(traj_target))
            padded_traj_ts.append(torch.cat(traj_ts))
            padded_traj_wts.append(torch.cat(traj_wt))

        return {
            "prefix": torch.stack(padded_prefixes),
            "draft_traj": torch.stack(padded_traj_inputs),
            "targets": torch.stack(padded_traj_targets),
            "timesteps": torch.stack(padded_traj_ts),
            "weights": torch.stack(padded_traj_wts),
            "genre": torch.stack(genres),
            "multi_hot": torch.stack(multi_hots)
        }

def build_dataloader(jsonl_files: list[str], split: str, batch_size: int, augment: bool = False, num_workers: int = 4, distributed: bool = False) -> DataLoader:
    processor = ProcessData()
    dataset = TwoTowerDataset(jsonl_files=jsonl_files, processor=processor, augment=augment)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(not distributed),
        sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=dataset.collate_fn
    )

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def build_scheduler(optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps: return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def train(n_steps: int = N_STEPS):
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (local_rank == 0)
    _, world_size = distributed_rank_world()

    writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, "runs", RUN_NAME)) if is_main_process else None

    train_loader = build_dataloader(JSONL_FILES, "train", BATCH_SIZE, augment=True, distributed=True)
    val_loader = build_dataloader(JSONL_FILES, "validation", BATCH_SIZE, augment=False, distributed=True)

    if is_main_process: print("Initializing Tower A (Frozen AR Musician)...")
    model_ar = ARContextModel().to(device)
    ar_checkpoint = torch.load(AR_MODEL_PATH, map_location=device)
    model_ar.load_state_dict(ar_checkpoint['model_state_dict'] if 'model_state_dict' in ar_checkpoint else ar_checkpoint)
    model_ar.eval()
    for param in model_ar.parameters():
        param.requires_grad = False

    if is_main_process: print("Initializing Tower B (Trainable Diffusion Editor)...")
    model_denoiser = TwoTowerDenoiser().to(device)

    model_denoiser = DDP(model_denoiser, device_ids=[local_rank], find_unused_parameters=False)
    checkpoint = load_training_checkpoint(config=twotower_config) if RESUME_TRAINING else None
    optimizer, optimizer_backend = build_optimizer(
        model_denoiser.parameters(),
        checkpoint=checkpoint,
        is_main_process=is_main_process,
        config=twotower_config
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(AMP_DTYPE == torch.float16))

    update_step, best_val_loss = 0, float('inf')
    resume_state = None
    scheduler = build_scheduler(optimizer, WARMUP_STEPS, n_steps)
    if RESUME_TRAINING:
        update_step, best_val_loss, scheduler_restored, lr_changed, resume_start_lrs, resume_state = load_checkpoint(
            model_denoiser,
            optimizer,
            scheduler,
            device,
            scaler,
            checkpoint=checkpoint,
            config=twotower_config
        )
        if lr_changed or not scheduler_restored:
            scheduler = build_scheduler(
                optimizer,
                WARMUP_STEPS,
                n_steps,
                last_epoch=update_step - 1 if update_step > 0 else -1
            )
    del checkpoint

    has_resume_position = resume_state is not None and resume_state['has_resume_position']
    checkpoint_accum_steps = resume_state.get('checkpoint_accum_steps') if resume_state is not None else None
    if has_resume_position and checkpoint_accum_steps in (None, ACCUM_STEPS):
        micro_step = resume_state['micro_step']
    else:
        micro_step = update_step * ACCUM_STEPS

    start_epoch = resume_state['epoch'] if has_resume_position else micro_step // len(train_loader)
    checkpoint_world_size = resume_state.get('checkpoint_world_size') if resume_state is not None else None
    if has_resume_position and resume_state.get('next_global_sample_idx') is not None:
        resume_global_sample_idx = resume_state['next_global_sample_idx']
    elif has_resume_position and resume_state.get('next_sample_idx') is not None:
        resume_global_sample_idx = resume_state['next_sample_idx'] * (checkpoint_world_size or world_size)
    elif has_resume_position:
        checkpoint_batch_size = resume_state.get('checkpoint_batch_size') or BATCH_SIZE
        resume_global_sample_idx = (
            resume_state['next_batch_idx']
            * checkpoint_batch_size
            * (checkpoint_world_size or world_size)
        )
    else:
        resume_global_sample_idx = 0

    if has_resume_position and resume_state.get('next_sample_idx') is not None:
        resume_sample_idx = resume_state['next_sample_idx']
    else:
        resume_sample_idx = resume_global_sample_idx // world_size
    resume_batch_idx = resume_global_sample_idx // max(1, BATCH_SIZE * world_size)

    if is_main_process and resume_state is not None:
        if has_resume_position:
            print(
                f"Resuming from step {update_step}, epoch {start_epoch}, "
                f"global sample {resume_global_sample_idx}, "
                f"rank sample {resume_sample_idx}, batch {resume_batch_idx}."
            )
            if resume_state.get('next_global_sample_idx') is None:
                print(
                    "Checkpoint has no next_global_sample_idx; inferred resume "
                    "position from legacy metadata."
                )
            if checkpoint_world_size not in (None, world_size):
                print(
                    f"World size changed from {checkpoint_world_size} to {world_size}; "
                    "re-sharding remaining epoch samples."
                )
            if checkpoint_accum_steps not in (None, ACCUM_STEPS):
                print(
                    f"ACCUM_STEPS changed from {checkpoint_accum_steps} to {ACCUM_STEPS}; "
                    "starting a fresh accumulation window."
                )
        else:
            print(f"Resuming legacy checkpoint from step {update_step}; starting at epoch {start_epoch}, batch 0.")
        if resume_state.get('optimizer_restored'):
            print(f"Restored optimizer state ({resume_state['optimizer_backend']}).")
        else:
            print(f"Optimizer state not restored: {resume_state.get('optimizer_skipped_reason', 'not available')}.")
        if resume_state.get('rng_restored'):
            print("Restored saved RNG state for training ranks.")
    
    running_loss, log_steps = 0.0, 0
    model_denoiser.train()
    epoch = start_epoch

    while update_step < n_steps:
        if hasattr(train_loader.sampler, 'set_epoch'): train_loader.sampler.set_epoch(epoch)
        epoch_loader = resume_epoch_loader(
            train_loader,
            epoch=epoch,
            start_global_sample_idx=resume_global_sample_idx
        )
        
        for local_batch_idx, batch in enumerate(epoch_loader):
            if update_step >= n_steps: break
            global_sample_idx = resume_global_sample_idx + (local_batch_idx * BATCH_SIZE * world_size)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            is_accumulating = (micro_step + 1) % ACCUM_STEPS != 0
            ddp_context = model_denoiser.no_sync() if is_accumulating else contextlib.nullcontext()
            
            with ddp_context:
                with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                    # 1. Forward Pass through Tower A (Frozen) to generate static KV Context
                    with torch.no_grad():
                        _, past_kv = model_ar(
                            target=batch["prefix"],          
                            genre=batch["genre"],
                            multi_hot=batch["multi_hot"],
                            use_cache=True,
                            past_key_values=None,
                            seq_offset=0
                        )

                    # 2. Forward Pass through Tower B (Trainable) to Denoise the Trajectory
                    # Denoiser picks up exactly after the prefix RoPE indices (PROMPT_MAX + 2)
                    logits = model_denoiser(
                        noisy_target=batch["draft_traj"],
                        timestep=batch["timesteps"],
                        seq_offset=PROMPT_MAX + 2,
                        context_kv_cache=past_kv,
                        draft_size=BLOCK_SIZE + 1
                    )
                    
                    flat_logits = logits.view(-1, VOCAB_SIZE)
                    flat_targets = batch["targets"].view(-1)
                    flat_weights = batch["weights"].view(-1)

                    ce_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
                    loss = (ce_loss * flat_weights).sum() / max(1.0, flat_weights.sum())
                    loss = loss / ACCUM_STEPS

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
                    nn.utils.clip_grad_norm_(model_denoiser.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model_denoiser.parameters(), GRAD_CLIP)
                    optimizer.step()
                    
                scheduler.step()
                optimizer.zero_grad()
                update_step += 1

                if is_main_process and (update_step % LOG_EVERY == 0) and log_steps > 0:
                    avg_loss = running_loss / log_steps
                    print(f"  Step {update_step:>6} | Denoiser Loss: {avg_loss:>6.4f}")
                    writer.add_scalar('Train/TwoTower_Loss', avg_loss, update_step)
                    writer.add_scalar('Hyperparameters/LR', optimizer.param_groups[0]['lr'], update_step)
                    running_loss, log_steps = 0.0, 0

                if update_step > 0 and update_step % VAL_EVERY == 0:
                    val_loss = evaluate_validation(model_ar, model_denoiser, val_loader, local_rank, device)
                    checkpoint_rng_state = collect_rng_state_for_checkpoint(device)
                    if is_main_process:
                        print(f"\n--- Validation Step {update_step} | Val Loss: {val_loss:.4f} ---")
                        writer.add_scalar('Validation/TwoTower_Loss', val_loss, update_step)
                        is_best = val_loss < best_val_loss
                        if is_best: best_val_loss = val_loss
                        next_global_sample_idx = global_sample_idx + (BATCH_SIZE * world_size)
                        next_sample_idx = next_global_sample_idx // world_size
                        next_batch_idx = next_global_sample_idx // max(1, BATCH_SIZE * world_size)
                        next_epoch = epoch
                        if next_global_sample_idx >= len(train_loader.dataset):
                            next_epoch += 1
                            next_batch_idx = 0
                            next_sample_idx = 0
                            next_global_sample_idx = 0
                        save_checkpoint(
                            model_denoiser,
                            optimizer,
                            scheduler,
                            update_step,
                            best_val_loss,
                            next_epoch,
                            next_batch_idx,
                            next_sample_idx,
                            next_global_sample_idx,
                            micro_step,
                            scaler,
                            is_best,
                            optimizer_backend=optimizer_backend,
                            rng_state=checkpoint_rng_state,
                            config=twotower_config
                        )
        resume_global_sample_idx = 0
        epoch += 1

    if is_main_process and writer: writer.close()
    cleanup_ddp()
    return model_denoiser

@torch.no_grad()
def evaluate_validation(model_ar, model_denoiser, val_loader, local_rank, device, max_batches: int = 50):
    model_denoiser.eval()
    total_loss, steps = 0.0, 0
    iterator = tqdm(val_loader, desc="Validation", total=min(len(val_loader), max_batches), leave=False) if local_rank == 0 else val_loader

    for batch in iterator:
        if steps >= max_batches: break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
            _, past_kv = model_ar(
                target=batch["prefix"],          
                genre=batch["genre"],
                multi_hot=batch["multi_hot"],
                use_cache=True,
                past_key_values=None,
                seq_offset=0
            )

            logits = model_denoiser(
                noisy_target=batch["draft_traj"],
                timestep=batch["timesteps"],
                seq_offset=PROMPT_MAX + 2,
                context_kv_cache=past_kv,
                draft_size=BLOCK_SIZE + 1
            )
            
            flat_logits = logits.view(-1, VOCAB_SIZE)
            flat_targets = batch["targets"].view(-1)
            flat_weights = batch["weights"].view(-1)

            ce_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
            loss = (ce_loss * flat_weights).sum() / max(1.0, flat_weights.sum())
            
        total_loss += loss.item()
        steps += 1

    local_avg_loss = total_loss / max(1, steps)
    if torch.distributed.is_initialized():
        metrics = torch.tensor([local_avg_loss], device=device)
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
        global_avg_loss = (metrics / torch.distributed.get_world_size())[0].item()
    else:
        global_avg_loss = local_avg_loss

    model_denoiser.train()
    return global_avg_loss

if __name__ == "__main__":
    model = train(n_steps=N_STEPS)
