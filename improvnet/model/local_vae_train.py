"""Local VAE reconstruction training on indexed MIDI. python -m / torchrun.

--build-index prepares paths/genre metadata only; it does NOT start training.
Fresh runs need no AR checkpoint. Resumes include optimizer, RNG and data position.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import importlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sqlite3
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from improvnet.model.local_vae_model import LocalVAE, LocalVAEConfig, reconstruction_losses
from improvnet.utils.local_vae_data import LocalVAEDataset, build_midi_index, collate_bins, index_settings

LOSS_KEYS = ("activity", "onset", "velocity", "pedal", "consistency", "kl")
QUALITY_KEYS = ("activity_tp", "activity_fp", "activity_fn", "onset_tp", "onset_fp", "onset_fn",
                "velocity_error", "active_cells", "pedal_error", "pedal_cells", "invalid_onsets", "raw_onsets")
STAT_KEYS = ("loss", "reconstruction", *LOSS_KEYS, "denoising_reconstruction", "jitter_reconstruction",
             "robust_reconstruction", "examples", *QUALITY_KEYS,
             *(f"{branch}_{key}" for branch in ("denoising", "jitter") for key in QUALITY_KEYS),
             "corrupted_examples", "retries")


class ResumableBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size):
        super().__init__(sampler, batch_size, drop_last=True)
        self.start_batch = 0

    def __iter__(self):
        for index, batch in enumerate(super().__iter__()):
            if index >= self.start_batch:
                yield batch

    def __len__(self):
        return max(0, super().__len__() - self.start_batch)


def model_config(config):
    return LocalVAEConfig(frames_per_bin=config.COARSE_MS // config.FINE_MS,
                          base_channels=config.BASE_CHANNELS, channel_multipliers=tuple(config.CHANNEL_MULTIPLIERS),
                          latent_channels=config.LATENT_CHANNELS, res_blocks=config.RES_BLOCKS)


def lr_factor(step, config):
    if step < config.WARMUP_STEPS:
        return (step + 1) / max(1, config.WARMUP_STEPS)
    progress = min(1., (step - config.WARMUP_STEPS) / max(1, config.N_STEPS - config.WARMUP_STEPS))
    floor = config.MIN_LR / config.LR
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))


def amp_context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else nullcontext()


def to_device(batch, device):
    # Accompaniment/genre/instrument metadata are deliberately NOT model inputs.
    fields = ("roll", "pedal", "valid_mask", "pedal_enabled", "retries", "corrupted_roll")
    return {k: batch[k].to(device, non_blocking=True) for k in fields if k in batch}


def robustness_ramp(step, config):
    return min(1., (step + 1) / max(1, getattr(config, "ROBUSTNESS_WARMUP_STEPS", 0)))


def quality_stats(reconstruction, batch):
    r, roll = reconstruction, batch["roll"]
    mask = batch["valid_mask"][:, None].expand_as(roll[:, 0])
    active = (roll[:, 0] > 0) & mask
    predicted_active = (r.activity_logits.sigmoid() >= batch["activity_threshold"]) & mask
    raw_onset = (r.onset_logits.sigmoid() >= batch["onset_threshold"]) & mask
    predicted_onset, onset = raw_onset & predicted_active, (roll[:, 1] > 0) & mask
    p_mask = batch["valid_mask"][:, None] & batch["pedal_enabled"][:, None, None]
    return dict(activity_tp=(predicted_active & active).sum(), activity_fp=(predicted_active & ~active).sum(),
                activity_fn=(~predicted_active & active).sum(), onset_tp=(predicted_onset & onset).sum(),
                onset_fp=(predicted_onset & ~onset).sum(), onset_fn=(~predicted_onset & onset).sum(),
                velocity_error=(r.velocity.float() - roll[:, 0]).abs().masked_fill(~active, 0).sum() * 127,
                active_cells=active.sum(),
                pedal_error=(r.pedal.float() - batch["pedal"]).abs().masked_fill(~p_mask, 0).sum() * 127,
                pedal_cells=p_mask.sum(), invalid_onsets=(raw_onset & ~predicted_active).sum(), raw_onsets=raw_onset.sum())


def batch_objective(model, batch, config, step, *, sample=True, generator=None):
    ramp = robustness_ramp(step, config)
    semantic_weight = getattr(config, "SEMANTIC_WEIGHT", 0.) if getattr(config, "SEMANTIC_CORRUPTION", False) else 0.
    jitter_weight = getattr(config, "LATENT_JITTER_WEIGHT", 0.)
    corrupted = batch.get("corrupted_roll") if semantic_weight else None
    maximum = (getattr(config, "LATENT_JITTER_MAX_STD", 0.) * ramp if sample else
               getattr(config, "LATENT_JITTER_VALIDATION_STD", .1)) if jitter_weight else 0.
    output = model(batch["roll"], batch["pedal"], batch["valid_mask"], sample=sample,
                   corrupted_roll=corrupted, latent_jitter_std=maximum, fixed_jitter=not sample, generator=generator)

    def terms(branch):
        values = reconstruction_losses(branch, batch,
            activity_positive_fraction=getattr(config, "ACTIVITY_POSITIVE_FRACTION", .5),
            onset_positive_fraction=getattr(config, "ONSET_POSITIVE_FRACTION", .5))
        reconstruction = sum(values[k] * getattr(config, k.upper() + "_WEIGHT") for k in LOSS_KEYS if k != "kl")
        return values, reconstruction

    losses, reconstruction = terms(output)
    denoising, jitter = torch.zeros_like(reconstruction), torch.zeros_like(reconstruction)
    if output.denoising is not None:
        noisy_losses, denoising = terms(output.denoising)
        losses["kl"] = .5 * (losses["kl"] + noisy_losses["kl"])
    if output.jittered is not None:
        _, jitter = terms(output.jittered)
    beta = config.KL_WEIGHT * min(1., (step + 1) / max(1, config.KL_WARMUP_STEPS))
    total = reconstruction + ramp * (semantic_weight * denoising + jitter_weight * jitter) + beta * losses["kl"]
    # Fixed-weight, comparable clean+stress criterion for checkpoint selection.
    # Do not reward early checkpoints merely because auxiliary weights are small.
    active_semantic_weight = semantic_weight if output.denoising is not None else 0.
    active_jitter_weight = jitter_weight if output.jittered is not None else 0.
    robust = (reconstruction + active_semantic_weight * denoising + active_jitter_weight * jitter) / (
        1 + active_semantic_weight + active_jitter_weight)
    with torch.no_grad():
        quality_batch = dict(batch, activity_threshold=config.ACTIVITY_THRESHOLD, onset_threshold=config.ONSET_THRESHOLD)
        s = dict(loss=total.sum(), reconstruction=reconstruction.sum(), examples=total.new_tensor(len(batch["roll"])),
                 denoising_reconstruction=denoising.sum(), jitter_reconstruction=jitter.sum(), robust_reconstruction=robust.sum(),
                 **{k: losses[k].sum() for k in LOSS_KEYS},
                 **quality_stats(output.reconstruction, quality_batch), retries=batch["retries"].sum(),
                 corrupted_examples=total.new_zeros(()) if corrupted is None else (
                     (corrupted != batch["roll"]).masked_fill(~batch["valid_mask"][:, None, None], False).flatten(1).any(1).sum()))
        for name, branch in (("denoising", output.denoising), ("jitter", output.jittered)):
            values = quality_stats(branch.reconstruction, quality_batch) if branch is not None else {
                k: total.new_zeros(()) for k in QUALITY_KEYS}
            s.update({f"{name}_{k}": v for k, v in values.items()})
        stats = torch.stack([s[k].double() for k in STAT_KEYS])
    return total.mean(), stats


def summarize(stats):
    s = dict(zip(STAT_KEYS, stats.tolist()))
    n = max(1, s["examples"])
    values = {k: s[k] / n for k in ("loss", "reconstruction", *LOSS_KEYS,
                                    "denoising_reconstruction", "jitter_reconstruction", "robust_reconstruction")}
    for branch in ("", "denoising_", "jitter_"):
        for kind in ("activity", "onset"):
            prefix = branch + kind
            tp, fp, fn = (s[prefix + "_" + k] for k in ("tp", "fp", "fn"))
            values.update({prefix + "_precision": tp / max(1, tp + fp), prefix + "_recall": tp / max(1, tp + fn),
                           prefix + "_f1": 2 * tp / max(1, 2 * tp + fp + fn)})
    values.update(velocity_mae_midi=s["velocity_error"] / max(1, s["active_cells"]),
                  pedal_mae_midi=s["pedal_error"] / max(1, s["pedal_cells"]),
                  onset_without_activity_rate=s["invalid_onsets"] / max(1, s["raw_onsets"]),
                  retries_per_example=s["retries"] / n, corrupted_example_fraction=s["corrupted_examples"] / n,
                  examples=s["examples"])
    return values


@torch.inference_mode()
def evaluate(model, loader, config, device, step):
    raw = model.module if isinstance(model, DDP) else model
    was_training = raw.training
    raw.eval()
    totals = torch.zeros(len(STAT_KEYS), device=device, dtype=torch.float64)
    generator = torch.Generator(device=device).manual_seed(config.SEED + 7919 + int(os.environ.get("RANK", 0)))
    try:
        # Fixed stress noise, without consuming the posterior RNG used to train.
        for index, batch in enumerate(loader):
            if index >= config.MAX_VALIDATION_BATCHES:
                break
            with amp_context(device):
                _, stats = batch_objective(raw, to_device(batch, device), config, step, sample=False, generator=generator)
            totals += stats
        if dist.is_initialized():
            dist.all_reduce(totals)
        if not bool(torch.isfinite(totals).all()) or totals[STAT_KEYS.index("examples")] == 0:
            raise RuntimeError("Validation produced no examples or non-finite values")
        return summarize(totals)
    finally:
        raw.train(was_training)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def index_identity(path):
    """Cheap guard against replacing a frozen index while resuming a run."""
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def save_state(model, optimizer, scheduler, scaler, config, *, step, epoch, next_batch, best_val, device, index_id, best=False):
    local_rng = dict(python=random.getstate(), torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None)
    world = dist.get_world_size() if dist.is_initialized() else 1
    rng = [None] * world
    if world > 1:
        dist.all_gather_object(rng, local_rng)
    else:
        rng[0] = local_rng
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    if index_identity(config.INDEX_PATH) != index_id:
        raise ValueError("MIDI index changed during training; previous checkpoint preserved")
    raw = model.module if isinstance(model, DDP) else model
    state = dict(format="local_vae_training_v1", model_config=asdict(raw.config), train_config=vars(config),
                 model_state_dict=cpu_tree(raw.state_dict()), optimizer_state_dict=cpu_tree(optimizer.state_dict()),
                 scheduler_state_dict=scheduler.state_dict(), scaler_state_dict=scaler.state_dict(),
                 step=step, epoch=epoch, next_batch=next_batch, best_val=best_val, world_size=world, rng=rng,
                 index_identity=index_id)
    directory = Path(config.SAVE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    for name in (["latest_checkpoint.pt", "best_model.pt"] if best else ["latest_checkpoint.pt"]):
        path = directory / name
        pending = path.with_name(name + f".tmp.{os.getpid()}")
        torch.save(state, pending)
        os.replace(pending, path)


def validate_config(config):
    for name in ("BATCH_SIZE", "VALIDATION_BATCH_SIZE", "ACCUM_STEPS", "N_STEPS", "SAMPLES_PER_EPOCH",
                 "VALIDATION_SAMPLES", "MAX_VALIDATION_BATCHES", "LOG_EVERY", "VALIDATE_EVERY", "SAVE_EVERY", "MAX_DATA_RETRIES"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if config.COARSE_MS < 1 or config.FINE_MS < 1 or config.COARSE_MS % config.FINE_MS:
        raise ValueError("Coarse time must be divisible by fine time")
    if config.NUM_WORKERS < 0 or min(config.WARMUP_STEPS, config.KL_WARMUP_STEPS, config.CACHE_MB_PER_WORKER) < 0:
        raise ValueError("Workers/warmups/cache must be nonnegative")
    if not 0 < config.MIN_LR <= config.LR or not 0 <= config.ACTIVE_BIN_PROB <= 1 or config.GRAD_CLIP <= 0:
        raise ValueError("Invalid LR, active-bin probability or gradient clip")
    for name in ("KL_WEIGHT", "ACTIVITY_WEIGHT", "ONSET_WEIGHT", "VELOCITY_WEIGHT", "PEDAL_WEIGHT", "CONSISTENCY_WEIGHT"):
        if not math.isfinite(getattr(config, name)) or getattr(config, name) < 0:
            raise ValueError(f"Invalid {name}")
    for name in ("ACTIVITY_THRESHOLD", "ONSET_THRESHOLD"):
        if not 0 < getattr(config, name) < 1:
            raise ValueError(f"{name} must be in (0,1)")
    if not config.PITCH_SHIFT_RANGE[0] <= 0 <= config.PITCH_SHIFT_RANGE[1]:
        raise ValueError("Pitch shift range must include zero")
    fraction = getattr(config, "GIGAMIDI_DRUM_ONLY_FRACTION", .05)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("GIGAMIDI_DRUM_ONLY_FRACTION must be between 0 and 1")
    for name in ("ACTIVITY_POSITIVE_FRACTION", "ONSET_POSITIVE_FRACTION"):
        value = getattr(config, name, .5)
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be in (0,1)")
    for name in ("SEMANTIC_WEIGHT", "LATENT_JITTER_WEIGHT", "LATENT_JITTER_MAX_STD", "LATENT_JITTER_VALIDATION_STD"):
        value = getattr(config, name, 0.)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    warmup = getattr(config, "ROBUSTNESS_WARMUP_STEPS", 0)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("ROBUSTNESS_WARMUP_STEPS must be a nonnegative integer")
    if getattr(config, "LATENT_JITTER_WEIGHT", 0.) and (getattr(config, "LATENT_JITTER_MAX_STD", 0.) <= 0 or
                                                       getattr(config, "LATENT_JITTER_VALIDATION_STD", 0.) <= 0):
        raise ValueError("Positive LATENT_JITTER_WEIGHT needs positive training/validation noise levels")
    if getattr(config, "SEMANTIC_CORRUPTION", False):
        if not math.isfinite(config.SEMANTIC_NOTE_PROB) or not 0 <= config.SEMANTIC_NOTE_PROB <= 1:
            raise ValueError("SEMANTIC_NOTE_PROB must be in [0,1]")
        for name in ("SEMANTIC_PITCH_SHIFT_MAX", "SEMANTIC_TIME_SHIFT_MS", "SEMANTIC_MIN_NOTES_FOR_DELETION"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if config.SEMANTIC_MIN_NOTES_FOR_DELETION < 2:
            raise ValueError("Deletion must retain evidence from other notes")
        operations = config.SEMANTIC_OPERATION_WEIGHTS
        if (not operations or set(operations) - {"pitch", "addition", "timing", "deletion"} or
                any(not math.isfinite(v) or v < 0 for v in operations.values()) or sum(operations.values()) <= 0):
            raise ValueError("Invalid SEMANTIC_OPERATION_WEIGHTS")
    model_config(config)


def train(config, *, force_cpu=False, allow_resume_changes=False):
    validate_config(config)
    if not Path(config.INDEX_PATH).exists():
        raise FileNotFoundError(f"Build the MIDI index first: python -m improvnet.model.local_vae_train --build-index --index {config.INDEX_PATH}")
    index_id = index_identity(config.INDEX_PATH)
    rank, world, local_rank = (int(os.environ.get(k, default)) for k, default in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() and not force_cpu else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    writer = None
    handlers = {}
    stop_requested = [False]
    try:
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *_: stop_requested.__setitem__(0, True))
        random.seed(config.SEED + rank)
        torch.manual_seed(config.SEED + rank)
        directory = Path(config.SAVE_DIR)
        resume = directory / "latest_checkpoint.pt"
        state = None
        for path in (resume, directory / "best_model.pt"):
            if path.exists():
                candidate = torch.load(path, map_location="cpu", weights_only=False)
                if candidate.get("format") != "local_vae_training_v1":
                    raise ValueError("SAVE_DIR contains an unrelated checkpoint; it will not be overwritten")
                if not config.RESUME_TRAINING:
                    raise FileExistsError("Existing checkpoints: use RESUME_TRAINING or a fresh SAVE_DIR")
                if state is None:
                    state = candidate
        model = LocalVAE(model_config(config)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, betas=config.BETAS, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, config))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
        step, epoch, next_batch, best_val = 0, 0, 0, float("inf")
        if state is not None:
            if state.get("index_identity") != index_id:
                raise ValueError("MIDI index changed since checkpoint; choose a fresh SAVE_DIR")
            ignored = {"SAVE_DIR", "RESUME_TRAINING", "NUM_WORKERS", "LOG_EVERY", "VALIDATE_EVERY", "SAVE_EVERY", "CACHE_MB_PER_WORKER", "LOG_DATA_ERRORS",
                       "ACTIVITY_THRESHOLD", "ONSET_THRESHOLD"}  # Metrics only; not part of the objective.
            changed = [k for k, v in vars(config).items() if k not in ignored and state["train_config"].get(k) != v]
            compatible = {"BATCH_SIZE", "ACCUM_STEPS", "GIGAMIDI_DRUM_ONLY_FRACTION",
                          "VALIDATION_BATCH_SIZE", "MAX_VALIDATION_BATCHES", "VALIDATION_SAMPLES",
                          "ACTIVITY_WEIGHT", "ONSET_WEIGHT", "VELOCITY_WEIGHT", "PEDAL_WEIGHT",
                          "CONSISTENCY_WEIGHT", "KL_WEIGHT", "KL_WARMUP_STEPS", "GRAD_CLIP",
                          "ACTIVITY_POSITIVE_FRACTION", "ONSET_POSITIVE_FRACTION",
                          "SEMANTIC_CORRUPTION", "SEMANTIC_NOTE_PROB", "SEMANTIC_OPERATION_WEIGHTS",
                          "SEMANTIC_PITCH_SHIFT_MAX", "SEMANTIC_TIME_SHIFT_MS", "SEMANTIC_MIN_NOTES_FOR_DELETION",
                          "SEMANTIC_WEIGHT", "LATENT_JITTER_WEIGHT", "LATENT_JITTER_MAX_STD",
                          "LATENT_JITTER_VALIDATION_STD", "ROBUSTNESS_WARMUP_STEPS",
                          "ACTIVE_BIN_PROB", "PITCH_AUGMENTATION", "PITCH_SHIFT_RANGE",
                          "COLLECTION_WEIGHTS", "SAMPLES_PER_EPOCH", "MAX_DATA_RETRIES"}
            blocked = [k for k in changed if not allow_resume_changes or k not in compatible]
            if state["model_config"] != asdict(model.config) or state["world_size"] != world or blocked:
                raise ValueError(f"Exact resume needs the same architecture/world/data/optimizer settings; changed={changed}. "
                                 "Use --allow-resume-changes for loss weights, KL warmup, gradient clipping, "
                                 "sampling/augmentation and batch/validation settings; "
                                 "architecture, world size, index and optimizer schedule remain protected.")
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            scaler.load_state_dict(state["scaler_state_dict"])
            rng = state["rng"][rank]
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"].cpu())
            if device.type == "cuda" and rng["cuda"] is not None:
                torch.cuda.set_rng_state(rng["cuda"], device)
            step, epoch, next_batch, best_val = (state[k] for k in ("step", "epoch", "next_batch", "best_val"))
            if changed:
                # The old batch index no longer identifies the same examples
                # with a new batch size/pool. Resume at a virtual epoch boundary,
                # retaining optimizer/scheduler progress and posterior RNG.
                epoch += int(next_batch > 0)
                next_batch, best_val = 0, float("inf")
                if rank == 0:
                    print(f"WARNING: non-exact continuation; changed={changed}. Keeping weights, optimizer, "
                          f"scheduler and step {step}; restarting data at epoch {epoch}, batch 0. "
                          "Validation best-score tracking reset.", flush=True)
            del state
        if world > 1:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        train_data = LocalVAEDataset(config, "train", error_dir=directory / "data_errors")
        val_data = LocalVAEDataset(config, "validation", error_dir=directory / "data_errors")
        sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=True, seed=config.SEED, drop_last=True)
        batches = ResumableBatchSampler(sampler, config.BATCH_SIZE)
        if not len(batches):
            raise ValueError("SAMPLES_PER_EPOCH cannot supply one distributed batch")
        options = dict(num_workers=config.NUM_WORKERS, pin_memory=device.type == "cuda", collate_fn=collate_bins)
        train_loader = DataLoader(train_data, batch_sampler=batches, generator=torch.Generator().manual_seed(config.SEED), **options)
        val_loader = DataLoader(val_data, batch_size=config.VALIDATION_BATCH_SIZE,
                                sampler=range(rank, len(val_data), world), generator=torch.Generator().manual_seed(config.SEED + 1), **options)
        if rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(directory / "tensorboard"))
            raw = model.module if isinstance(model, DDP) else model
            print(f"Local VAE: {sum(p.numel() for p in raw.parameters()):,} parameters; latent={raw.config.latent_shape}; "
                  f"effective batch={config.BATCH_SIZE * world * config.ACCUM_STEPS}; resumed step={step}", flush=True)
            print(f"Denoising={getattr(config, 'SEMANTIC_CORRUPTION', False)}; "
                  f"latent jitter=0..{getattr(config, 'LATENT_JITTER_MAX_STD', 0.):.0%} channel std; "
                  f"robustness warmup={getattr(config, 'ROBUSTNESS_WARMUP_STEPS', 0)} steps", flush=True)
            if train_data.report["limited"]:
                print("WARNING: partial/debug MIDI index, not the full corpus", flush=True)
            for name, data in (("train", train_data), ("validation", val_data)):
                if data.drum_subset is not None:
                    subset = data.drum_subset
                    print(f"GigaMIDI {name} drums-only: retained {subset['retained']:,}/{subset['total']:,} "
                          f"files ({subset['fraction']:.0%}); other files unchanged", flush=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        metrics = torch.zeros(len(STAT_KEYS), device=device, dtype=torch.float64)
        while step < config.N_STEPS:
            sampler.set_epoch(epoch)
            train_data.set_epoch(epoch)
            batches.start_batch = next_batch
            full_batches = len(sampler) // config.BATCH_SIZE
            if next_batch >= full_batches:
                raise ValueError("Invalid checkpoint data position")
            for batch in train_loader:
                sync = accumulated + 1 == config.ACCUM_STEPS
                with model.no_sync() if isinstance(model, DDP) and not sync else nullcontext():
                    with amp_context(device):
                        loss, stats = batch_objective(model, to_device(batch, device), config, step)
                    finite = torch.isfinite(loss).to(torch.int32)
                    if world > 1:
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise FloatingPointError("Non-finite loss; last checkpoint preserved")
                    scaler.scale(loss / config.ACCUM_STEPS).backward()
                metrics += stats
                accumulated += 1
                next_batch += 1
                del loss, stats, batch
                if next_batch == full_batches:
                    epoch, next_batch = epoch + 1, 0
                if not sync:
                    continue
                scaler.unscale_(optimizer)
                norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                if not bool(torch.isfinite(norm_value)):
                    raise FloatingPointError("Non-finite gradients; last checkpoint preserved")
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                step += 1
                if step % config.LOG_EVERY == 0 or step == config.N_STEPS:
                    if world > 1:
                        dist.all_reduce(metrics)
                    values = summarize(metrics)
                    if rank == 0:
                        print(f"step {step}: loss={values['loss']:.4f}, recon={values['reconstruction']:.4f}, "
                              f"KL={values['kl']:.4f}, activity_F1={values['activity_f1']:.3f}, onset_F1={values['onset_f1']:.3f}, "
                              f"onset_P={values['onset_precision']:.3f}, denoise={values['denoising_reconstruction']:.4f}, "
                              f"jitter={values['jitter_reconstruction']:.4f}, jitter_onset_F1={values['jitter_onset_f1']:.3f}, "
                              f"velocity_MAE={values['velocity_mae_midi']:.2f}, retries={values['retries_per_example']:.2f}", flush=True)
                        for k, v in values.items():
                            writer.add_scalar("train/" + k, v, step)
                        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                        writer.add_scalar("train/kl_weight", config.KL_WEIGHT * min(1., step / max(1, config.KL_WARMUP_STEPS)), step)
                        writer.add_scalar("train/robustness_ramp", robustness_ramp(step - 1, config), step)
                    metrics.zero_()
                best = False
                if step % config.VALIDATE_EVERY == 0 or step == config.N_STEPS:
                    values = evaluate(model, val_loader, config, device, step - 1)
                    best = values["robust_reconstruction"] < best_val
                    best_val = min(best_val, values["robust_reconstruction"])
                    if rank == 0:
                        print(f"validation {step}: clean={values['reconstruction']:.4f}, "
                              f"denoise={values['denoising_reconstruction']:.4f}, jitter={values['jitter_reconstruction']:.4f}, "
                              f"robust={values['robust_reconstruction']:.4f}, onset_F1/P="
                              f"{values['onset_f1']:.3f}/{values['onset_precision']:.3f}, jitter_onset_F1/P="
                              f"{values['jitter_onset_f1']:.3f}/{values['jitter_onset_precision']:.3f}", flush=True)
                        for k, v in values.items():
                            writer.add_scalar("validation/" + k, v, step)
                request = os.environ.get("CHECKPOINT_REQUEST_FILE")
                requested = torch.tensor(int(stop_requested[0] or bool(request and Path(request).exists())), device=device)
                if world > 1:
                    dist.all_reduce(requested, op=dist.ReduceOp.MAX)
                if best or step % config.SAVE_EVERY == 0 or step == config.N_STEPS or requested.item():
                    save_state(model, optimizer, scheduler, scaler, config, step=step, epoch=epoch,
                               next_batch=next_batch, best_val=best_val, device=device, index_id=index_id, best=best)
                if requested.item() or step == config.N_STEPS:
                    return
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="improvnet.model.local_vae_config")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--max-index-files", type=int, help="Per collection, debug index only")
    parser.add_argument("--index")
    parser.add_argument("--save-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accum-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-resume-changes", action="store_true",
                        help="Continue weights/optimizer with changed loss weights, KL warmup, clipping, sampling/augmentation or batch/validation settings; not an exact replay")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.no_resume and args.allow_resume_changes:
        parser.error("--allow-resume-changes cannot be combined with --no-resume")
    module = importlib.import_module(args.config)
    config = SimpleNamespace(**{k: v for k, v in vars(module).items() if k.isupper()})
    for arg, key in (("index", "INDEX_PATH"), ("save_dir", "SAVE_DIR"), ("batch_size", "BATCH_SIZE"),
                     ("accum_steps", "ACCUM_STEPS"), ("num_workers", "NUM_WORKERS"), ("steps", "N_STEPS"),
                     ("samples_per_epoch", "SAMPLES_PER_EPOCH")):
        if getattr(args, arg) is not None:
            setattr(config, key, getattr(args, arg))
    if args.no_resume:
        config.RESUME_TRAINING = False
    if args.build_index:
        if int(os.environ.get("RANK", 0)) != 0:
            return
        if Path(config.INDEX_PATH).exists():
            with sqlite3.connect(Path(config.INDEX_PATH).resolve().as_uri() + "?mode=ro", uri=True) as db:
                report = json.loads(db.execute("SELECT value FROM metadata WHERE key='report'").fetchone()[0])
            if report["settings"] != index_settings(config):
                raise ValueError("Existing index has different settings; choose a new --index")
            print(f"Index already ready: {report['total']:,} MIDI files; limited={report['limited']}")
        else:
            build_midi_index(config, max_files_per_collection=args.max_index_files)
    else:
        if args.max_index_files is not None:
            parser.error("--max-index-files requires --build-index")
        train(config, force_cpu=args.cpu, allow_resume_changes=args.allow_resume_changes)


if __name__ == "__main__":
    main()
