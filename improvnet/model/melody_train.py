"""AR-style melody fine-tuning: DDP, AMP, accumulation, validation and resume."""
import argparse
import contextlib
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from improvnet.model import melody_config as cfg
from improvnet.model.ar_train import all_ranks_finite, build_scheduler, checkpoint_requested, next_training_position
from improvnet.model.melody_model import (MelodyRefinementModel, checkpoint_metadata,
                                        initialize_from_ar, validate_checkpoint)
from improvnet.utils.ar_utils import (build_optimizer, collect_rng_state_for_checkpoint,
                                    load_checkpoint, resume_epoch_loader, save_checkpoint)
from improvnet.utils.melody_data import build_dataloader


def autocast(device):
    if device.type != 'cuda':
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    return torch.amp.autocast('cuda', dtype=dtype)


def loss_for_batch(model, batch, ignore_index=-100):
    logits = model(target=batch['input'], genre=batch['genre'],
                   multi_hot=batch['multi_hot'], strength=batch['strength'])
    return F.cross_entropy(logits.flatten(0, 1), batch['target'].flatten(), ignore_index=ignore_index)


@torch.no_grad()
def evaluate_validation(model, loader, device, config=cfg, max_batches=100):
    model.eval()
    # Sums/counts per task avoid comparing losses with different token counts.
    totals = torch.zeros(3, 2, dtype=torch.float64, device=device)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device):
            logits = model(target=batch['input'], genre=batch['genre'],
                           multi_hot=batch['multi_hot'], strength=batch['strength'])
            losses = F.cross_entropy(logits.transpose(1, 2), batch['target'],
                                     ignore_index=config.IGNORE_INDEX, reduction='none')
        for task in range(3):
            mask = (batch['task'] == task).unsqueeze(1) & (batch['target'] != config.IGNORE_INDEX)
            totals[task, 0] += losses[mask].sum().double()
            totals[task, 1] += mask.sum()
    if dist.is_initialized():
        dist.all_reduce(totals)
    model.train()
    metrics = {name: (totals[i, 0] / totals[i, 1]).item() for i, name in
               enumerate(('refine', 'rhythm', 'harmonize')) if totals[i, 1] > 0}
    metrics['loss'] = (totals[:, 0].sum() / totals[:, 1].sum().clamp_min(1)).item()
    return metrics


def train(config=cfg, n_steps=None):
    n_steps = config.N_STEPS if n_steps is None else n_steps
    if not 0 < n_steps <= config.N_STEPS or config.ACCUM_STEPS <= 0:
        raise ValueError('Stop step must be in [1, N_STEPS]; accumulation must be positive')
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    if distributed:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    if rank == 0:
        print(f'Melody refiner: batch/GPU={config.BATCH_SIZE}, accumulation={config.ACCUM_STEPS}, '
              f'effective batch={config.BATCH_SIZE * config.ACCUM_STEPS * world}, stop step={n_steps}', flush=True)
    random.seed(config.DATA_SPLIT_SEED + rank)
    torch.manual_seed(config.DATA_SPLIT_SEED + rank)
    train_loader = build_dataloader('train', config.BATCH_SIZE, distributed, config)
    val_loader = build_dataloader('validation', config.BATCH_SIZE, distributed, config)
    model = MelodyRefinementModel(**config.MODEL_KWARGS)
    metadata = checkpoint_metadata(model, config)
    latest = Path(config.SAVE_DIR) / 'latest_checkpoint.pt'
    checkpoint = torch.load(latest, map_location='cpu', weights_only=False, mmap=True) if (
        config.RESUME_TRAINING and latest.exists()) else None
    if checkpoint is not None:
        validate_checkpoint(checkpoint, metadata)
        for key, value in [('batch_size', config.BATCH_SIZE), ('accum_steps', config.ACCUM_STEPS), ('world_size', world)]:
            if checkpoint.get(key) != value:
                raise ValueError(f'Resume {key} changed; keep it unchanged or use a fresh run')
    elif config.INIT_CHECKPOINT:
        if not Path(config.INIT_CHECKPOINT).is_file():
            raise FileNotFoundError(config.INIT_CHECKPOINT)
        initial = torch.load(config.INIT_CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
        initialize_from_ar(model, initial)
        del initial
        if rank == 0:
            print('Loaded AR weights; initialized task rows/strength condition; fresh optimizer.', flush=True)
    model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None)
    optimizer = build_optimizer(model.parameters(), is_main_process=rank == 0, config=config)
    scheduler = build_scheduler(optimizer, config.WARMUP_STEPS, config.N_STEPS, config.MIN_LR, config.DECAY_STEPS)
    use_scaler = device.type == 'cuda' and torch.cuda.get_device_capability(device)[0] < 8
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    step, best, epoch, consumed, micro_step = 0, float('inf'), 0, 0, 0
    if checkpoint is not None:
        step, best, restored, changed, resume = load_checkpoint(
            model, optimizer, scheduler, device, scaler, checkpoint=checkpoint, config=config)
        epoch, consumed = resume['epoch'], resume['next_global_sample_idx']
        micro_step = resume['micro_step']
        if not restored or changed:
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] = config.LR
            scheduler = build_scheduler(optimizer, config.WARMUP_STEPS, config.N_STEPS,
                                        config.MIN_LR, config.DECAY_STEPS, last_epoch=step - 1)
    del checkpoint
    writer = SummaryWriter(str(Path(config.SAVE_DIR) / 'runs' / config.RUN_NAME)) if rank == 0 else None
    optimizer.zero_grad(set_to_none=True)
    model.train()
    stop, last_position = False, (epoch, 0, consumed // world, consumed)
    window_loss, window_batches = 0.0, 0

    def save(position, is_best=False):
        rng = collect_rng_state_for_checkpoint(device)
        if rank == 0:
            save_checkpoint(model, optimizer, scheduler, step, best, *position,
                            micro_step, scaler=scaler, is_best=is_best,
                            rng_state=rng, config=config, metadata=metadata)
        if distributed:
            dist.barrier()

    while step < n_steps and not stop:
        train_loader.dataset.set_epoch(epoch)
        train_loader.sampler.set_epoch(epoch)
        loader = resume_epoch_loader(train_loader, epoch=epoch, start_global_sample_idx=consumed)
        for batch_idx, batch in enumerate(loader):
            global_sample = consumed + batch_idx * config.BATCH_SIZE * world
            batch = {k: v.to(device) for k, v in batch.items()}
            accumulating = (micro_step + 1) % config.ACCUM_STEPS != 0
            sync = model.no_sync() if distributed and accumulating else contextlib.nullcontext()
            with sync:
                with autocast(device):
                    loss = loss_for_batch(model, batch, config.IGNORE_INDEX) / config.ACCUM_STEPS
                if not all_ranks_finite(loss, device):
                    optimizer.zero_grad(set_to_none=True)
                    micro_step += config.ACCUM_STEPS - micro_step % config.ACCUM_STEPS
                    window_loss, window_batches = 0.0, 0
                    continue
                scaler.scale(loss).backward()
            micro_step += 1
            window_loss += loss.item() * config.ACCUM_STEPS
            window_batches += 1
            if accumulating:
                continue
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            if not all_ranks_finite(norm, device):
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.update()
                window_loss, window_batches = 0.0, 0
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            last_position = next_training_position(global_sample, epoch, len(train_loader.dataset), config.BATCH_SIZE, world)
            if rank == 0 and step % config.LOG_EVERY == 0:
                mean_loss = window_loss / max(1, window_batches)
                print(f'Step {step:>6} | CE {mean_loss:.4f} | PPL {math.exp(min(mean_loss, 20)):.2f}', flush=True)
                writer.add_scalar('Train/CE', mean_loss, step)
                writer.add_scalar('Hyperparameters/LR', optimizer.param_groups[0]['lr'], step)
            window_loss, window_batches = 0.0, 0
            is_best = False
            validation_due = step % config.VAL_EVERY == 0
            if validation_due:
                metrics = evaluate_validation(model, val_loader, device, config)
                is_best = metrics['loss'] < best
                best = min(best, metrics['loss'])
                if rank == 0:
                    print(f'Validation at {step}: {metrics}', flush=True)
                    for name, value in metrics.items():
                        writer.add_scalar('Validation/' + name, value, step)
            stop = checkpoint_requested(device)
            if validation_due or step % config.CHECKPOINT_EVERY == 0 or stop or step >= n_steps:
                save(last_position, is_best)
            if stop or step >= n_steps:
                break
        consumed = 0
        epoch += 1
    if writer:
        writer.close()
    if distributed:
        dist.destroy_process_group()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=cfg.DATA_FILE)
    parser.add_argument('--save-dir', default=cfg.SAVE_DIR)
    parser.add_argument('--steps', type=int, default=cfg.N_STEPS,
                        help='Stop at this update; scheduler horizon remains config N_STEPS')
    args = parser.parse_args()
    settings = SimpleNamespace(**{k: v for k, v in vars(cfg).items() if k.isupper()})
    settings.DATA_FILE, settings.SAVE_DIR = args.data, args.save_dir
    train(settings, args.steps)
