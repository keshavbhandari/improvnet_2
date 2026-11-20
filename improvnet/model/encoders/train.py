"""Shared bidirectional contrastive training for continuity, style and beat encoders."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import importlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from improvnet.model.encoders.model import ContinuityConfig, ContinuityEncoder
from improvnet.metrics.continuity.data import ContinuityDataset, PairBatchSampler, PairCollator
from improvnet.metrics.continuity.objective import contrastive_loss, summarize


def amp_context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else nullcontext()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def settings_dict(config):
    return {key: value for key, value in vars(config).items() if key.isupper()}


def validate_config(config):
    for key in ("BATCH_SIZE", "ACCUM_STEPS", "VALIDATION_BATCH_SIZE", "N_STEPS", "LOG_EVERY", "VALIDATE_EVERY", "SAVE_EVERY", "SAMPLES_PER_EPOCH", "MIN_LENGTH", "MAX_LENGTH", "EVAL_LENGTH"):
        value = getattr(config, key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("NUM_WORKERS", "WARMUP_STEPS", "MAX_VALIDATION_BATCHES"):
        value = getattr(config, key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if min(config.BATCH_SIZE, config.VALIDATION_BATCH_SIZE) < 2:
        raise ValueError("Contrastive batches require at least two pairs")
    if config.WARMUP_STEPS >= config.N_STEPS or config.MIN_LENGTH > config.MAX_LENGTH or config.EVAL_LENGTH > config.MAX_LENGTH:
        raise ValueError("Invalid warmup/crop lengths")
    if not math.isfinite(config.TEMPERATURE) or config.TEMPERATURE <= 0:
        raise ValueError("TEMPERATURE must be finite and positive")
    if not (math.isfinite(config.LR) and math.isfinite(config.MIN_LR) and 0 < config.MIN_LR <= config.LR and math.isfinite(config.GRAD_CLIP) and config.GRAD_CLIP > 0):
        raise ValueError("Invalid learning rate/gradient clip")
    if not math.isfinite(config.WEIGHT_DECAY) or config.WEIGHT_DECAY < 0 or len(config.BETAS) != 2 or any(not 0 <= x < 1 for x in config.BETAS):
        raise ValueError("Invalid AdamW settings")
    if not isinstance(config.PITCH_AUGMENTATION, bool) or not isinstance(config.MAX_SEMITONE_SHIFT, int) or isinstance(config.MAX_SEMITONE_SHIFT, bool) or config.MAX_SEMITONE_SHIFT < 0:
        raise ValueError("Invalid pitch augmentation")

    start = getattr(config, 'CONTINUATION_START_STEP', None)
    end_lr = getattr(config, 'CONTINUATION_END_LR', None)
    if (start is None) != (end_lr is None):
        raise ValueError('Continuation requires both a start step and end learning rate')
    if start is not None:
        if not isinstance(start, int) or isinstance(start, bool) or not config.WARMUP_STEPS < start < config.N_STEPS:
            raise ValueError('Continuation must extend a completed post-warmup schedule')
        if not math.isfinite(end_lr) or not 0 < end_lr <= config.MIN_LR:
            raise ValueError('Continuation end LR must be positive and at most the original final LR')

    if hasattr(config,'DECAY_STEPS'):
        if not config.WARMUP_STEPS < config.DECAY_STEPS < config.N_STEPS or not 0 < getattr(config,'TAIL_END_LR',0) <= config.MIN_LR:
            raise ValueError('Invalid initial decay horizon or final tail LR')


def lr_factor(step, config):
    start = getattr(config, 'CONTINUATION_START_STEP', getattr(config, 'DECAY_STEPS', None))
    if start is not None and step >= start:
        progress = min(1., (step-start)/(config.N_STEPS-start))
        final_lr = getattr(config, 'CONTINUATION_END_LR', getattr(config, 'TAIL_END_LR', config.MIN_LR))
        lr = final_lr + (config.MIN_LR-final_lr)*.5*(1+math.cos(math.pi*progress))
        return lr/config.LR
    if step < config.WARMUP_STEPS:
        return (step + 1) / max(1, config.WARMUP_STEPS)
    horizon = start if start is not None else config.N_STEPS
    progress = min(1., (step - config.WARMUP_STEPS) / max(1, horizon - config.WARMUP_STEPS))
    floor = config.MIN_LR / config.LR
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))


def validate_resume_settings(saved, config, trained_steps):
    """Permit an explicit extension only at the original completed horizon.

    Architecture, data, batch, augmentations and optimizer settings stay strict.
    Later resumes of this same continuation must match its saved schedule too.
    """
    current = settings_dict(config)
    saved = dict(saved)
    # Older style checkpoints named the same two-stage curve after the resume
    # that introduced it. Normalize metadata only; strict schedule checks remain.
    if 'CONTINUATION_START_STEP' not in current and 'DECAY_STEPS' in current:
        for old, new in (('CONTINUATION_START_STEP', 'DECAY_STEPS'),
                         ('CONTINUATION_END_LR', 'TAIL_END_LR')):
            if old in saved:
                if new in saved and saved[new] != saved[old]:
                    raise ValueError(f'Conflicting saved schedule settings: {old}, {new}')
                saved[new] = saved.pop(old)
    # Module moves change metadata names, never model parameters or sampling policy.
    aliases = {
        'improvnet.model.reference_encoder.data.StylePairDataset': 'improvnet.model.encoders.style.data.StylePairDataset',
        'improvnet.model.encoders.style.StylePairDataset': 'improvnet.model.encoders.style.data.StylePairDataset',
        'improvnet.model.encoders.beat.BeatPairDataset': 'improvnet.model.encoders.beat.data.BeatPairDataset',
        'improvnet.model.encoders.beat.BeatCollator': 'improvnet.model.encoders.beat.data.BeatCollator',
    }
    for key in ('DATASET_CLASS', 'COLLATOR_CLASS'):
        if saved.get(key) in aliases:
            saved[key] = aliases[saved[key]]
    if (saved.get('CHECKPOINT_FORMAT') == 'melody_reference_encoder_v6'
            and saved.get('DATASET_CLASS') == 'improvnet.model.encoders.style.data.StylePairDataset'
            and current.get('CHECKPOINT_FORMAT') == 'style_v1'):
        saved['CHECKPOINT_FORMAT'] = 'style_v1'
    flexible = {"SAVE_DIR", "NUM_WORKERS", "LOG_EVERY", "VALIDATE_EVERY", "SAVE_EVERY", "TENSORBOARD", "RESUME_TRAINING", "DEVICE"}
    if saved.get('N_STEPS') != config.N_STEPS:
        start = getattr(config, 'CONTINUATION_START_STEP', None)
        if saved.get('CONTINUATION_START_STEP') is not None or start != saved.get('N_STEPS') or trained_steps != start or config.N_STEPS <= start:
            raise ValueError('A continuation must start at the saved completed training horizon')
        flexible |= {'N_STEPS', 'CONTINUATION_START_STEP', 'CONTINUATION_END_LR'}
    changed = [key for key in set(saved) | set(current) if key not in flexible and saved.get(key) != current.get(key)]
    if changed:
        raise ValueError(f'Incompatible continuity resume; changed settings={sorted(changed)}')
    if trained_steps > config.N_STEPS:
        raise ValueError('Checkpoint is beyond the requested training horizon')


def prepare_directory(directory, resuming):
    directory.mkdir(parents=True, exist_ok=True)
    if resuming:
        return
    artifacts = [directory / name for name in ("metrics.jsonl", "tensorboard", "best_checkpoint.pt") if (directory / name).exists()]
    if artifacts:
        archive_root = directory / "previous_runs"
        archive_root.mkdir(exist_ok=True)
        archive = Path(tempfile.mkdtemp(prefix="interrupted-", dir=archive_root))
        for artifact in artifacts:
            artifact.rename(archive / artifact.name)
        print(f"Starting fresh; previous debug outputs preserved in {archive}", flush=True)


def encode_batch(model, batch):
    negative = {key:batch[key] for key in ('negative_a','negative_b','negative_a_mask','negative_b_mask') if key in batch}
    return model(batch['a'],batch['a_mask'],batch['b'],batch['b_mask'],**negative)


def batch_loss(embeddings,batch,*,temperature=.1,distributed=False,negative_gap_ms=1000):
    if len(embeddings)==4:
        from improvnet.model.encoders.objective import beat_loss
        return beat_loss(*embeddings,batch['piece_ids'],batch['bounds'],batch['negative_piece_ids'],batch['negative_bounds'],
                         temperature=temperature,distributed=distributed,gap_ms=negative_gap_ms,record_ids=batch.get('record_ids'),negative_record_ids=batch.get('negative_record_ids'))
    return contrastive_loss(*embeddings,batch['piece_ids'],temperature=temperature,distributed=distributed)


@torch.inference_mode()
def evaluate(model, loader, device, *, temperature=.1, max_batches=0, negative_gap_ms=1000):
    model = model.module if isinstance(model, DDP) else model
    training = model.training
    model.eval()
    totals = torch.zeros(7, device=device, dtype=torch.float64)
    try:
        for index, batch in enumerate(loader):
            if max_batches and index >= max_batches:
                break
            # Local validation pools avoid collectives in unequal DDP shards.
            # A singleton/duplicate-only final batch has no valid negatives.
            if "negative_a" not in batch and batch["piece_ids"].unique().numel() < 2:
                continue
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with amp_context(device):
                embeddings = encode_batch(model, batch)
            _, stats = batch_loss(embeddings, batch, temperature=temperature, negative_gap_ms=negative_gap_ms)
            totals += stats.double()
        if dist.is_initialized():
            dist.all_reduce(totals)
        return summarize(totals)
    finally:
        model.train(training)


def report_metrics(prefix, step, metrics, writer=None):
    print(f"{prefix} {step}: " + ", ".join((f"{key}={value:.3g}" if key == "lr" else f"{key}={value:.4f}") for key, value in metrics.items() if key != "pairs"), flush=True)
    if writer:
        for key, value in metrics.items():
            writer.add_scalar(prefix + "/" + key, value, step)


def train(config, *, model_class=ContinuityEncoder):
    validate_config(config)
    world, rank, local = [int(os.environ.get(key, default)) for key, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"))]
    device = torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        device = torch.device("cuda", local if world > 1 or device.index is None else device.index)
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    writer, log, train_data, val_data = None, None, None, None
    try:
        random.seed(config.SEED + rank)
        torch.manual_seed(config.SEED + rank)
        directory = Path(config.SAVE_DIR)
        latest = directory / "latest_checkpoint.pt"
        if latest.exists() and not config.RESUME_TRAINING:
            raise FileExistsError("Existing latest checkpoint: enable resume or use a new SAVE_DIR")
        state = torch.load(latest, map_location="cpu", weights_only=True, mmap=True) if latest.exists() else None
        if state is None and getattr(config, 'CONTINUATION_START_STEP', None) is not None:
            raise FileNotFoundError('Continuation requires the existing latest_checkpoint.pt')
        options = dict(min_length=config.MIN_LENGTH, max_length=config.MAX_LENGTH, eval_length=config.EVAL_LENGTH,
                       seed=config.SEED, pitch_augmentation=config.PITCH_AUGMENTATION, max_semitone_shift=config.MAX_SEMITONE_SHIFT)
        dataset_type = ContinuityDataset
        if getattr(config, 'DATASET_CLASS', None):
            module, name = config.DATASET_CLASS.rsplit('.', 1)
            dataset_type = getattr(importlib.import_module(module), name)
        options.update(getattr(config, 'DATASET_OPTIONS', {}))
        checkpoint_format = getattr(config, 'CHECKPOINT_FORMAT', 'continuity_encoder_v1')
        train_data = dataset_type(config.MANIFEST, "train", **options)
        val_data = dataset_type(config.MANIFEST, "validation", tokenizer=train_data.tokenizer, **options)
        model_config = ContinuityConfig.from_settings(config, vocab_size=train_data.tokenizer.vocab_size, pad_id=train_data.tokenizer.pad_id)
        model = model_class(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY, betas=config.BETAS)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, config))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
        sampler = PairBatchSampler(len(train_data), config.SAMPLES_PER_EPOCH, config.BATCH_SIZE,
                                   seed=config.SEED, rank=rank, world_size=world)
        step, epoch, next_batch, best = 0, 0, 0, float("inf")
        if state is not None:
            validate_resume_settings(state['train_config'], config, state['trained_steps'])
            if state.get("format") not in ({"style_v1", "melody_reference_encoder_v6"} if checkpoint_format == "style_v1" else {checkpoint_format}) or state["config"] != asdict(model_config) or state["data_fingerprint"] != train_data.fingerprint or state["vocabulary_digest"] != train_data.vocab_digest or state["world_size"] != world:
                raise ValueError('Incompatible continuity architecture, dataset, vocabulary or world size')
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            if getattr(config, 'CONTINUATION_START_STEP', None) == state['trained_steps']:
                expected_lr = config.LR*lr_factor(state['trained_steps'], config)
                if any(not math.isclose(group['lr'], expected_lr, rel_tol=1e-6, abs_tol=1e-12) for group in optimizer.param_groups):
                    raise ValueError('Continuation start LR does not match the saved optimizer LR')
            scaler.load_state_dict(state["scaler_state_dict"])
            step, epoch, next_batch, best = [state[key] for key in ("trained_steps", "epoch", "next_batch", "best_validation_loss")]
        if world > 1:
            model = DDP(model, device_ids=[local] if device.type == "cuda" else None)
        raw_model = model.module if isinstance(model, DDP) else model
        if state is not None:
            rng = state["rng_states"][rank]
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(rng["cuda"], device)
            del state
        collator = PairCollator
        if getattr(config, 'COLLATOR_CLASS', None):
            module, name = config.COLLATOR_CLASS.rsplit('.', 1)
            collator = getattr(importlib.import_module(module), name)
        loader_options = dict(num_workers=config.NUM_WORKERS, collate_fn=collator(model_config.pad_id), pin_memory=device.type == "cuda")
        loader = DataLoader(train_data, batch_sampler=sampler, generator=torch.Generator().manual_seed(config.SEED), **loader_options)
        val_loader = DataLoader(val_data, batch_size=config.VALIDATION_BATCH_SIZE, sampler=range(rank, len(val_data), world),
                                generator=torch.Generator().manual_seed(config.SEED + 1), **loader_options)
        if rank == 0:
            prepare_directory(directory, latest.exists())
            log = (directory / "metrics.jsonl").open("a" if latest.exists() else "x")
            if config.TENSORBOARD:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(directory / "tensorboard")
            if getattr(config, 'CONTINUATION_START_STEP', None) is not None:
                print(f"Continuation: target={config.N_STEPS}; current LR={optimizer.param_groups[0]['lr']:.3g}; final LR={config.CONTINUATION_END_LR:.3g}; optimizer/data/RNG/best validation restored", flush=True)
            label = "Beat" if checkpoint_format=="beat_v1" else "Style" if checkpoint_format=="style_v1" else "Continuity"
            print(f"{label} encoder: {model_config.embed_dim} wide/{model_config.n_layers} layers; global pairs per forward={config.BATCH_SIZE * world}; resumed step={step}", flush=True)

        def save(name, metrics):
            rng = {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None}
            rng_states = [rng]
            if world > 1:
                rng_states = [None] * world
                dist.all_gather_object(rng_states, rng)
            if rank != 0:
                return
            payload = {"format": checkpoint_format, "config": asdict(model_config), "train_config": settings_dict(config),
                       "model_state_dict": cpu_tree(raw_model.state_dict()), "optimizer_state_dict": cpu_tree(optimizer.state_dict()),
                       "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                       "trained_steps": step, "epoch": epoch, "next_batch": next_batch, "world_size": world, "rng_states": rng_states,
                       "best_validation_loss": best, "validation": metrics, "data_fingerprint": train_data.fingerprint,
                       "vocabulary_digest": train_data.vocab_digest, "tokenizer_config": train_data.tokenizer.config,
                       "augmentation": getattr(config, 'AUGMENTATION_DESCRIPTION', "independent per-segment bounded pitch transposition; training only")}
            temporary = directory / (name + f".tmp.{os.getpid()}")
            torch.save(payload, temporary)
            temporary.replace(directory / name)

        accumulated, totals, validation = 0, torch.zeros(7, device=device, dtype=torch.float64), None
        optimizer.zero_grad(set_to_none=True)
        model.train()
        while step < config.N_STEPS:
            if not 0 <= next_batch < len(sampler):
                raise ValueError("Invalid resumed data position")
            sampler.epoch, sampler.start_batch = epoch, next_batch
            train_data.set_epoch(epoch)
            for batch in loader:
                sync = accumulated + 1 == config.ACCUM_STEPS
                context = model.no_sync() if isinstance(model, DDP) and not sync else nullcontext()
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with context:
                    with amp_context(device):
                        embeddings = encode_batch(model, batch)
                    loss, stats = batch_loss(embeddings, batch, temperature=config.TEMPERATURE, distributed=world > 1, negative_gap_ms=getattr(config, 'NEGATIVE_GAP_MS', 1000))
                    finite = torch.isfinite(loss).to(torch.int32)
                    if world > 1:
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise FloatingPointError("Non-finite continuity loss; checkpoint not overwritten")
                    scaler.scale(loss / config.ACCUM_STEPS).backward()
                totals += stats.double()
                del batch, embeddings, loss, stats
                accumulated += 1
                next_batch += 1
                if next_batch == len(sampler):
                    epoch, next_batch = epoch + 1, 0
                if not sync:
                    continue
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                if not torch.isfinite(norm):
                    if not scaler.is_enabled():
                        raise FloatingPointError("Non-finite continuity gradients")
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    totals.zero_()
                    continue
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated, step = 0, step + 1
                if step == 1 or step % config.LOG_EVERY == 0 or step == config.N_STEPS:
                    if world > 1:
                        dist.all_reduce(totals)
                    if rank == 0:
                        metrics = {**summarize(totals), "lr": scheduler.get_last_lr()[0]}
                        report_metrics("train", step, metrics, writer)
                        log.write(json.dumps({"step": step, "train": metrics}) + "\n")
                        log.flush()
                    totals.zero_()
                improved = False
                if step % config.VALIDATE_EVERY == 0 or step == config.N_STEPS:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    validation = evaluate(model, val_loader, device, temperature=config.TEMPERATURE, max_batches=config.MAX_VALIDATION_BATCHES, negative_gap_ms=getattr(config, 'NEGATIVE_GAP_MS', 1000))
                    improved = validation["loss"] < best
                    best = min(best, validation["loss"])
                    if rank == 0:
                        report_metrics("validation", step, validation, writer)
                        log.write(json.dumps({"step": step, "validation": validation}) + "\n")
                        log.flush()
                request = os.environ.get("CHECKPOINT_REQUEST_FILE")
                requested = torch.tensor(int(bool(request and Path(request).exists())), device=device)
                if world > 1:
                    dist.all_reduce(requested, op=dist.ReduceOp.MAX)
                if improved or step % config.SAVE_EVERY == 0 or step == config.N_STEPS or requested.item():
                    save("latest_checkpoint.pt", validation)
                    if improved:
                        save("best_checkpoint.pt", validation)
                if requested.item():
                    return
                if step == config.N_STEPS:
                    break
    finally:
        if writer:
            writer.close()
        if log:
            log.close()
        for dataset in (train_data, val_data):
            if dataset is not None:
                dataset.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main(*, default_config="improvnet.metrics.continuity.continuity_config", model_class=ContinuityEncoder):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config)
    for flag in ("manifest", "save-dir", "device"):
        parser.add_argument("--" + flag)
    for flag in ("batch-size", "accum-steps", "steps", "workers", "max-validation-batches"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    module = importlib.import_module(args.config)
    config = SimpleNamespace(**settings_dict(module))
    for argument, key in (("manifest", "MANIFEST"), ("save_dir", "SAVE_DIR"), ("device", "DEVICE"),
                          ("batch_size", "BATCH_SIZE"), ("accum_steps", "ACCUM_STEPS"), ("steps", "N_STEPS"),
                          ("workers", "NUM_WORKERS"), ("max_validation_batches", "MAX_VALIDATION_BATCHES")):
        if getattr(args, argument) is not None:
            setattr(config, key, getattr(args, argument))
    if args.no_resume:
        config.RESUME_TRAINING = False
    if args.no_tensorboard:
        config.TENSORBOARD = False
    train(config, model_class=model_class)


if __name__ == "__main__":
    main()
