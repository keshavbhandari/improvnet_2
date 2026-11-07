"""AR-style melody fine-tuning: DDP, AMP, accumulation, validation and resume."""
import argparse
import contextlib
import logging
import math
import os
from pathlib import Path
import random
import sys
from itertools import islice
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from improvnet.model import melody_config as cfg
from improvnet.model.ar_train import all_ranks_finite, build_scheduler, checkpoint_requested, next_training_position
from improvnet.utils.melody_stems import DRUM_KIT_PITCHES
from improvnet.model.melody_model import (MelodyRefinementModel, checkpoint_metadata,
                                        initialize_from_ar, validate_checkpoint)
from improvnet.utils.ar_utils import (ProcessData, build_optimizer, collect_rng_state_for_checkpoint,
                                    load_checkpoint, resume_epoch_loader, save_checkpoint)
from improvnet.utils.melody_data import build_dataloader,DATA_STAT_NAMES
from improvnet.utils.melody import TASKS
from improvnet.model.encoders.runtime import FrozenConditioners, pin_checkpoint


def autocast(device):
    if device.type != 'cuda':
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    return torch.amp.autocast('cuda', dtype=dtype)


def loss_for_batch(model, batch, ignore_index=-100):
    logits = model(target=batch['input'], genre=batch['genre'],
                   multi_hot=batch['multi_hot'], strength=batch['strength'],
                   style_embedding=batch['style_embedding'], style_mask=batch['style_mask'],beat_embedding=batch.get('beat_embedding'),beat_mask=batch.get('beat_mask'))
    return F.cross_entropy(logits.flatten(0, 1), batch['target'].flatten(), ignore_index=ignore_index)


def progress_logger(save_dir):
    """Keep a flushed progress record independent of Slurm's stdout forwarding."""
    path=Path(save_dir)/'training.log'
    path.parent.mkdir(parents=True,exist_ok=True)
    logger=logging.Logger('melody-training',level=logging.INFO)
    formatter=logging.Formatter('%(asctime)s | %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    for handler in (logging.StreamHandler(sys.stdout),logging.FileHandler(path,mode='a')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def report_exposures(counts,report):
    if dist.is_initialized():
        dist.all_reduce(counts)
    rows=counts[:-1].reshape(len(TASKS),2).tolist()
    report('Exposure (non-jazz, jazz/blues): '+str(dict(zip(TASKS,rows)))+
           f'; missing-drum examples={counts[-1].item()}')
    return {f'exposure/{task}/'+genre:rows[i][j] for i,task in enumerate(TASKS)
            for j,genre in enumerate(('other','jazz_blues'))}


def report_data_stats(stats,split,report):
    if dist.is_initialized():
        dist.all_reduce(stats)
    counts=dict(zip(DATA_STAT_NAMES,stats.tolist()))
    attempts,retries=counts['attempts'],counts['retries']
    accepted=attempts-retries
    retry_percent=100*retries/max(1,attempts)
    routed_percent=100*counts['routed']/max(1,accepted)
    reasons=', '.join(f'{k}={counts[k]}' for k in DATA_STAT_NAMES[3:] if counts[k]) or 'none'
    report(f'Data ({split}): retries={retries}/{attempts} ({retry_percent:.2f}% of attempts); '
           f'routed={counts["routed"]}/{accepted} ({routed_percent:.2f}% of accepted); reasons: {reasons}.')
    return dict(data_retry_percent=retry_percent,data_routed_percent=routed_percent)


@torch.no_grad()
def evaluate_validation(model, loader, device, config=cfg, max_batches=100, report=None, reference_encoder=None):
    model.eval()
    tasks = getattr(config, 'TASK_NAMES', TASKS)
    # Sums/counts per task avoid comparing losses with different token counts.
    totals = torch.zeros(len(tasks), 2, dtype=torch.float64, device=device)
    instruments = ProcessData().INSTRUMENT_CLASSES
    stem_totals = torch.zeros(len(instruments), 2, dtype=torch.float64, device=device)
    data_stats=torch.zeros(len(DATA_STAT_NAMES),dtype=torch.long,device=device)
    for i, batch in enumerate(islice(loader,max_batches)):
        batch = {k: v.to(device) for k, v in batch.items()}
        reference_encoder.condition_batch(batch)
        if 'data_stats' in batch:
            data_stats+=batch['data_stats']
        with autocast(device):
            logits = model(target=batch['input'], genre=batch['genre'],
                           multi_hot=batch['multi_hot'], strength=batch['strength'],
                           style_embedding=batch['style_embedding'], style_mask=batch['style_mask'],beat_embedding=batch.get('beat_embedding'),beat_mask=batch.get('beat_mask'))
            losses = F.cross_entropy(logits.transpose(1, 2), batch['target'],
                                     ignore_index=config.IGNORE_INDEX, reduction='none')
        for task in range(len(tasks)):
            mask = (batch['task'] == task).unsqueeze(1) & (batch['target'] != config.IGNORE_INDEX)
            totals[task, 0] += losses[mask].sum().double()
            totals[task, 1] += mask.sum()
        if 'stem_mask' in batch:
            for instrument in range(len(stem_totals)):
                mask = batch['stem_mask'] & (batch['stem_instrument'] == instrument).unsqueeze(1)
                stem_totals[instrument, 0] += losses[mask].sum().double()
                stem_totals[instrument, 1] += mask.sum()
        if report is not None and (i+1)%25==0:
            report(f'Validation progress: {i+1}/{min(max_batches,len(loader))} batches per rank')
    if dist.is_initialized():
        dist.all_reduce(totals)
        dist.all_reduce(stem_totals)
    model.train()
    metrics = {name: (totals[i, 0] / totals[i, 1]).item() for i, name in
               enumerate(tasks) if totals[i, 1] > 0}
    metrics['loss'] = (totals[:, 0].sum() / totals[:, 1].sum().clamp_min(1)).item()
    for i, name in enumerate(instruments):
        if stem_totals[i, 1] > 0:
            metrics['stem_target/' + name.replace(' ', '_')] = (stem_totals[i, 0] / stem_totals[i, 1]).item()
    if stem_totals[:, 1].sum() > 0:
        metrics['stem_target'] = (stem_totals[:, 0].sum() / stem_totals[:, 1].sum()).item()
    if report is not None:
        metrics.update(report_data_stats(data_stats,'validation',report))
    return metrics


def train(config=cfg, n_steps=None):
    tasks = TASKS
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
    logger=progress_logger(config.SAVE_DIR) if rank==0 else None
    report=logger.info if logger else lambda message: None
    if rank == 0:
        report(f'Melody refiner: batch/GPU={config.BATCH_SIZE}, accumulation={config.ACCUM_STEPS}, '
               f'effective batch={config.BATCH_SIZE * config.ACCUM_STEPS * world}, stop step={n_steps}')
        report(f'Direct progress log: {Path(config.SAVE_DIR)/"training.log"}')
        report(f'Data loader: workers/rank={config.NUM_WORKERS}, prefetch={config.PREFETCH_FACTOR}, '
               f'cache/worker<={config.WORKER_RECORD_CACHE} records / '
               f'{getattr(config,"WORKER_CACHE_NOTES",100000)} notes.')
    random.seed(config.DATA_SPLIT_SEED + rank)
    torch.manual_seed(config.DATA_SPLIT_SEED + rank)
    latest = Path(config.SAVE_DIR) / 'latest_checkpoint.pt'
    checkpoint = torch.load(latest, map_location='cpu', weights_only=False, mmap=True) if (
        config.RESUME_TRAINING and latest.exists()) else None
    if checkpoint is not None:
        validate_checkpoint(checkpoint)  # Reject incompatible protocols before allocating the model.
    paths={}
    for name in ('style','beat'):
        key=name.upper()+'_ENCODER'
        path=checkpoint['refinement'][name+'_encoder'] if checkpoint is not None else getattr(config,key)
        if not Path(path).is_file():raise FileNotFoundError(f'Train the {name} encoder first: {path}')
        paths[name]=path
    for name,path in paths.items():
        pinned=path if checkpoint is not None else pin_checkpoint(path,Path(config.SAVE_DIR)/'encoders'/name)
        setattr(config,name.upper()+'_ENCODER',pinned)
    reference_encoder=FrozenConditioners(config.STYLE_ENCODER,config.BEAT_ENCODER,device,
        style_sha256=checkpoint['refinement']['style_encoder_sha256'] if checkpoint is not None else None,
        beat_sha256=checkpoint['refinement']['beat_encoder_sha256'] if checkpoint is not None else None)
    for name in ('style','beat'):
        encoder=getattr(reference_encoder,name)
        if encoder.dim!=getattr(config,name.upper()+'_DIM'):raise ValueError(f'{name} encoder dimension differs')
        setattr(config,name.upper()+'_ENCODER_SHA256',encoder.sha256)
        if rank==0:report(f'{name} encoder: frozen=True; step={encoder.step}; sha256={encoder.sha256}; checkpoint={encoder.path}')
    train_loader = build_dataloader('train', config.BATCH_SIZE, distributed, config)
    val_loader = build_dataloader('validation', config.BATCH_SIZE, distributed, config)
    if rank == 0:
        report(f'Data: {config.DATA_FILE}; training candidate slots/epoch={len(train_loader.dataset)}, '
              f'validation files={len(val_loader.dataset)}; requested task weights={dict(zip(tasks, config.TASK_WEIGHTS))}')
        report(f'{config.RUN_NAME} tasks={tasks}; jazz/blues full file passes={config.JAZZ_BLUES_REPEATS}; '
               f'non-jazz candidate pool={config.CLASSICAL_RATIO}x; eligible mixture tasks sampled separately; '
               f'drum priority={config.STEM_DRUM_PROB:.0%}; kit-only drum sounds={len(DRUM_KIT_PITCHES)}; neighbor-reference visibility={config.REFERENCE_PROB:.0%}; '
               f'history refine/mix={config.HISTORY_PROB:.0%}/{config.POLY_HISTORY_PROB:.0%}; history corruption={config.HISTORY_CORRUPTION_PROB:.0%}; '
               f'refinement=ornaments first + balanced 10%-100% independent edits (5% none); '
               f'pitch error range=+/-{config.PITCH_CORRUPTION_SEMITONES} semitones; '
               f'onset/duration jitter<= {config.TIMING_CORRUPTION_MS}ms (10ms ticks); '
               f'stem timing jitter<= {config.STEM_TIMING_JITTER_MS}ms; '
               f'accompaniment pitch errors={config.STEM_PITCH_SHIFT_MIN_PROB:.0%}..{config.STEM_PITCH_SHIFT_PROB:.0%} per example, '
               f'range=+/-{config.STEM_PITCH_SHIFT_SEMITONES} semitones; '
               f'harmonize masking>={config.HARMONIZE_MIN_MASK_RATIO:.0%}, fully masked={config.HARMONIZE_FULL_MASK_PROB:.0%}.')
    model = MelodyRefinementModel(**config.MODEL_KWARGS)
    metadata = checkpoint_metadata(model, config)
    data_policy_updated=False
    if checkpoint is not None:
        validate_checkpoint(checkpoint, metadata,allow_data_policy_update=True)
        data_policy_updated=checkpoint['refinement'].get('data_policy_revision',1)!=metadata['data_policy_revision']
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
            report('Loaded pretrained weights; initialized missing task/reference parameters; fresh optimizer.')
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
        if rank==0:
            report(f'Resumed checkpoint at step {step}; optimizer restored={resume["optimizer_restored"]}; '
                   f'scheduler restored={restored}; epoch={epoch}; next sample={consumed}.')
        if data_policy_updated:
            best=float('inf')
            if rank==0:
                report(f'Data policy upgraded to revision {metadata["data_policy_revision"]}: '
                       'resetting best-validation comparison for the changed sampling policy; '
                       'model/optimizer/scheduler and progress are retained.')
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
    data_stats=torch.zeros(len(DATA_STAT_NAMES),dtype=torch.long,device=device)
    exposures=torch.zeros(len(TASKS)*2+1,dtype=torch.long,device=device)
    corruption_summary=torch.zeros(6,dtype=torch.float64,device=device)

    def save(position, is_best=False):
        report(f'Saving checkpoint at step {step}.')
        rng = collect_rng_state_for_checkpoint(device)
        if rank == 0:
            save_checkpoint(model, optimizer, scheduler, step, best, *position,
                            micro_step, scaler=scaler, is_best=is_best,
                            rng_state=rng, config=config, metadata=metadata)
        if distributed:
            dist.barrier()
        report(f'Checkpoint saved at step {step}.')

    while step < n_steps and not stop:
        train_loader.dataset.set_epoch(epoch)
        train_loader.sampler.set_epoch(epoch)
        loader = resume_epoch_loader(train_loader, epoch=epoch, start_global_sample_idx=consumed)
        for batch_idx, batch in enumerate(loader):
            global_sample = consumed + batch_idx * config.BATCH_SIZE * world
            batch = {k: v.to(device) for k, v in batch.items()}
            reference_encoder.condition_batch(batch)
            data_stats+=batch['data_stats']
            exposures+=batch['exposure'].sum(0)
            corruption_summary+=batch['corruption_summary']
            accumulating = (micro_step + 1) % config.ACCUM_STEPS != 0
            sync = model.no_sync() if distributed and accumulating else contextlib.nullcontext()
            with sync:
                with autocast(device):
                    loss = loss_for_batch(model, batch, config.IGNORE_INDEX) / config.ACCUM_STEPS
                if not all_ranks_finite(loss, device):
                    report(f'Step {step}: skipping batch because loss is non-finite on at least one rank.')
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
                report(f'Step {step}: skipping update because gradients are non-finite on at least one rank.')
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
                report(f'Step {step:>6} | CE {mean_loss:.4f} | PPL {math.exp(min(mean_loss, 20)):.2f}')
                writer.add_scalar('Train/CE', mean_loss, step)
                writer.add_scalar('Hyperparameters/LR', optimizer.param_groups[0]['lr'], step)
            window_loss, window_batches = 0.0, 0
            is_best = False
            validation_due = step % config.VAL_EVERY == 0
            if validation_due:
                report(f'Validation starting at step {step}: {len(val_loader.dataset)} files, '
                       f'up to {min(100,len(val_loader))} batches per rank.')
                metrics = evaluate_validation(model, val_loader, device, config,report=report, reference_encoder=reference_encoder)
                is_best = metrics['loss'] < best
                best = min(best, metrics['loss'])
                if rank == 0:
                    report(f'Validation at {step}: {metrics}')
                    for name, value in metrics.items():
                        writer.add_scalar('Validation/' + name, value, step)
            stop = checkpoint_requested(device)
            if step%getattr(config,'DATA_LOG_EVERY',50)==0 or stop or step>=n_steps:
                data_metrics=report_data_stats(data_stats,'train',report)
                data_metrics.update(report_exposures(exposures,report))
                if dist.is_initialized():
                    dist.all_reduce(corruption_summary)
                if corruption_summary[0] > 0:
                    names=('requested_mean','effective_mean','changed_mean','changed_low_fraction','changed_high_fraction')
                    rates=(corruption_summary[1:]/corruption_summary[0]).tolist()
                    data_metrics.update({'corruption/'+name:value for name,value in zip(names,rates)})
                    report(f'Refinement corruption: requested mean={rates[0]:.1%}, '
                           f'effective mean={rates[1]:.1%}, changed notes={rates[2]:.1%}; '
                           f'low (<=20%) examples={rates[3]:.1%}, high (>=80%)={rates[4]:.1%}.')
                corruption_summary.zero_()
                exposures.zero_()
                data_stats.zero_()
                if rank==0:
                    for name,value in data_metrics.items():
                        writer.add_scalar('Train/'+name,value,step)
            if validation_due or step % config.CHECKPOINT_EVERY == 0 or stop or step >= n_steps:
                save(last_position, is_best)
            if stop or step >= n_steps:
                break
        consumed = 0
        epoch += 1
    if writer:
        writer.close()
    if logger:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    if distributed:
        dist.destroy_process_group()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=cfg.DATA_FILE)
    parser.add_argument('--save-dir', default=cfg.SAVE_DIR)
    parser.add_argument('--init-checkpoint', type=Path, default=cfg.INIT_CHECKPOINT,
                        help='Load AR weights only when starting a fresh run')
    parser.add_argument('--steps', type=int, default=cfg.N_STEPS,
                        help='Stop at this update; scheduler horizon remains config N_STEPS')
    args = parser.parse_args()
    settings = SimpleNamespace(**{k: v for k, v in vars(cfg).items() if k.isupper()})
    settings.DATA_FILE, settings.SAVE_DIR = args.data, args.save_dir
    settings.INIT_CHECKPOINT = args.init_checkpoint
    train(settings, args.steps)
