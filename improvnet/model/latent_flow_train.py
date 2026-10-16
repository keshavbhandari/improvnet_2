"""Stage-2 MIDI latent flow matching training. No AR changes and no VAE weight updates.

--check performs one inference-only model/data check; it never starts training.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import importlib
import math
import os
from pathlib import Path
import random
import signal
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from improvnet.inference.local_vae_inference import load_model as load_vae
from improvnet.model.local_vae_train import ResumableBatchSampler, amp_context, cpu_tree, index_identity, lr_factor
from improvnet.model.latent_flow_model import LatentFlowConfig, LatentFlowModel
from improvnet.utils.latent_flow import (
    LatentNormalizer, checkpoint_digest, clean_from_velocity, encode_windows, flow_target, weighted_loss,
)
from improvnet.utils.latent_flow_data import LatentFlowDataset, collate_windows, genre_labels

FORMAT = "latent_flow_training_v1"
STAT_KEYS = ("loss", "latent_squared_error", "examples", "low_squared_error", "low_examples",
             "mid_squared_error", "mid_examples", "high_squared_error", "high_examples", "retries")


def model_config(config, vae, labels):
    channels, pitch, time = vae.config.latent_shape
    return LatentFlowConfig(latent_channels=channels, latent_pitch=pitch, latent_time=time,
        frames_per_bin=vae.config.frames_per_bin, history_bins=config.HISTORY_BINS, target_bins=config.TARGET_BINS,
        patch_pitch=config.PATCH_PITCH, patch_time=config.PATCH_TIME, d_model=config.D_MODEL,
        n_layers=config.N_LAYERS, n_heads=config.N_HEADS, ff_multiplier=config.FF_MULTIPLIER,
        num_genres=len(labels), max_tracks=config.MAX_ACCOMPANIMENT_TRACKS,
        max_events=config.MAX_ACCOMPANIMENT_EVENTS, rhythm_frames_per_token=config.RHYTHM_FRAMES_PER_TOKEN,
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING)


def validate_config(c):
    for name in ("TARGET_BINS", "MAX_ACCOMPANIMENT_EVENTS", "LATENT_CALIBRATION_SAMPLES", "VAE_BATCH_SIZE",
                 "SAMPLES_PER_EPOCH", "VALIDATION_SAMPLES", "BATCH_SIZE", "VALIDATION_BATCH_SIZE", "ACCUM_STEPS",
                 "N_STEPS", "LOG_EVERY", "VALIDATE_EVERY", "SAVE_EVERY", "MAX_VALIDATION_BATCHES", "MAX_DATA_RETRIES"):
        if isinstance(getattr(c, name), bool) or not isinstance(getattr(c, name), int) or getattr(c, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if c.HISTORY_BINS < 0 or c.NUM_WORKERS < 0 or c.WARMUP_STEPS < 0:
        raise ValueError("History/workers/warmup must be nonnegative")
    for name in ("PROMPTLESS_PROB", "PITCH_DROP_PROB", "PITCH_NOTE_DROP_PROB", "GENRE_DROP_PROB", "ACTIVE_BIN_PROB", "GIGAMIDI_DRUM_ONLY_FRACTION"):
        if not math.isfinite(getattr(c, name)) or not 0 <= getattr(c, name) <= 1:
            raise ValueError(f"Invalid {name}")
    p = c.ACCOMPANIMENT_MODE_PROBS
    if set(p) != {"none", "full", "rhythm"} or any(not math.isfinite(v) or v < 0 for v in p.values()) or not math.isclose(sum(p.values()), 1):
        raise ValueError("Accompaniment mode probabilities must sum to one")
    if not 0 < c.MIN_LR <= c.LR or c.GRAD_CLIP <= 0 or not 0 <= c.EMA_DECAY < 1:
        raise ValueError("Invalid optimizer/EMA settings")
    if not c.PITCH_SHIFT_RANGE[0] <= 0 <= c.PITCH_SHIFT_RANGE[1]:
        raise ValueError("Pitch shifts must include zero")
    save, checkpoint = Path(c.SAVE_DIR).resolve(), Path(c.VAE_CHECKPOINT).resolve()
    if save == checkpoint.parent or checkpoint.is_relative_to(save):
        raise ValueError("Stage-2 SAVE_DIR must be separate from VAE checkpoints")


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


@torch.inference_mode()
def calibrate(vae, dataset, c, device):
    loader = DataLoader(dataset, batch_size=c.VALIDATION_BATCH_SIZE,
        sampler=range(min(c.LATENT_CALIBRATION_SAMPLES, len(dataset))), num_workers=0, collate_fn=collate_windows)
    sums = squares = None
    count = 0.
    for batch in loader:
        batch = move_batch(batch, device)
        with amp_context(device):
            z, weights = encode_windows(vae, batch, c.VAE_BATCH_SIZE)
        w = weights.double() * (~batch["known"])[..., None, None, None]
        total = (z.double() * w).sum((0, 1, 3, 4)).cpu()
        square = (z.double().square() * w).sum((0, 1, 3, 4)).cpu()
        sums, squares = (total, square) if sums is None else (sums + total, squares + square)
        count += float(w.sum()) * z.shape[-2]
    if not count:
        raise ValueError("No training target latents for calibration")
    mean = sums / count
    std = (squares / count - mean.square()).clamp_min(1e-6).sqrt()
    return LatentNormalizer(mean, std)


def batch_objective(model, vae, normalizer, batch, config, *, generator=None):
    with torch.no_grad():
        clean, weights = encode_windows(vae, batch, config.VAE_BATCH_SIZE)
        clean = normalizer.normalize(clean, weights)
        t = torch.rand(len(clean), device=clean.device, generator=generator).clamp_min(1e-4)
        epsilon = torch.randn(clean.shape, device=clean.device, generator=generator)
        noisy, target, levels = flow_target(clean, batch["known"], weights, t, epsilon)
    prediction = model(noisy, levels, batch["known"], weights, batch)
    losses = weighted_loss(prediction, target, weights, batch["known"])
    with torch.no_grad():
        estimate = clean_from_velocity(noisy, prediction, levels)
        errors = weighted_loss(estimate, clean, weights, batch["known"])
        stats = dict(loss=losses.sum(), latent_squared_error=errors.sum(), examples=losses.new_tensor(len(losses)),
                     retries=batch["retries"].sum())
        for prefix, selection in (("low", t >= .75), ("mid", (t >= .25) & (t < .75)), ("high", t < .25)):
            stats[prefix + "_squared_error"], stats[prefix + "_examples"] = errors[selection].sum(), selection.sum()
        vector = torch.stack([stats[k].double() for k in STAT_KEYS])
    return losses.mean(), vector


def summarize(stats):
    s = dict(zip(STAT_KEYS, stats.tolist()))
    n = max(1, s["examples"])
    result = dict(loss=s["loss"] / n, latent_nrmse=math.sqrt(s["latent_squared_error"] / n), retries=s["retries"] / n)
    for prefix in ("low", "mid", "high"):
        if s[prefix + "_examples"]:
            result["latent_nrmse_" + prefix] = math.sqrt(s[prefix + "_squared_error"] / s[prefix + "_examples"])
    return result


@torch.no_grad()
def update_ema(ema, model, decay):
    raw = model.module if isinstance(model, DDP) else model
    for dest, source in zip(ema.parameters(), raw.parameters()):
        dest.lerp_(source.detach(), 1 - decay)


@torch.inference_mode()
def evaluate(model, vae, normalizer, loader, c, device):
    training = model.training
    model.eval()
    stats = torch.zeros(len(STAT_KEYS), device=device, dtype=torch.float64)
    generator = torch.Generator(device=device).manual_seed(c.SEED + int(os.environ.get("RANK", 0)) + 9001)
    try:
        for index, batch in enumerate(loader):
            if index >= c.MAX_VALIDATION_BATCHES:
                break
            with amp_context(device):
                _, values = batch_objective(model, vae, normalizer, move_batch(batch, device), c, generator=generator)
            stats += values
        if dist.is_initialized():
            dist.all_reduce(stats)
        if not bool(torch.isfinite(stats).all()) or stats[STAT_KEYS.index("examples")] == 0:
            raise RuntimeError("Empty/nonfinite validation")
        return summarize(stats)
    finally:
        model.train(training)


def save_state(model, ema, optimizer, scheduler, scaler, normalizer, c, labels, vae_hash, index_id,
               *, step, epoch, next_batch, best_val, device, best=False):
    rng = dict(python=random.getstate(), torch=torch.get_rng_state(),
               cuda=torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None)
    world = dist.get_world_size() if dist.is_initialized() else 1
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    if index_identity(c.INDEX_PATH) != index_id:
        raise ValueError("MIDI index changed; previous checkpoint preserved")
    raw = model.module if isinstance(model, DDP) else model
    state = dict(format=FORMAT, model_config=asdict(raw.config), train_config=vars(c), genres=list(labels),
        model_state_dict=cpu_tree(raw.state_dict()), ema_state_dict=cpu_tree(ema.state_dict()),
        optimizer_state_dict=cpu_tree(optimizer.state_dict()), scheduler_state_dict=scheduler.state_dict(),
        scaler_state_dict=scaler.state_dict(), normalizer=normalizer.state_dict(), vae_sha256=vae_hash,
        index_identity=index_id, step=step, epoch=epoch, next_batch=next_batch, best_val=best_val,
        world_size=world, rng=states)
    directory = Path(c.SAVE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    for name in (["latest_checkpoint.pt", "best_model.pt"] if best else ["latest_checkpoint.pt"]):
        path = directory / name
        pending = path.with_name(name + f".tmp.{os.getpid()}")
        torch.save(state, pending)
        os.replace(pending, path)


def prepare(c, device):
    validate_config(c)
    if not Path(c.INDEX_PATH).is_file():
        raise FileNotFoundError("Build the MIDI index with local_vae_train --build-index first")
    vae, tokenizer, _, _ = load_vae(c.VAE_CHECKPOINT, device)
    vae.requires_grad_(False)
    if (tokenizer.config.coarse_ms, tokenizer.config.fine_ms, tokenizer.config.piano_programs) != (c.COARSE_MS, c.FINE_MS, tuple(c.PIANO_PROGRAMS)):
        raise ValueError("Stage-2 time grid/instrument selection differs from the frozen VAE")
    return vae, genre_labels(c.INDEX_PATH)


@torch.inference_mode()
def check(c, device):
    """Read-only forward check; no optimizer, backward, checkpoints or calibration artifacts."""
    vae, labels = prepare(c, device)
    dataset = LatentFlowDataset(c, "validation", labels=labels)
    batch = move_batch(collate_windows([dataset[0]]), device)
    model = LatentFlowModel(model_config(c, vae, labels)).to(device).eval()
    norm = LatentNormalizer(torch.zeros(vae.config.latent_shape[0]), torch.ones(vae.config.latent_shape[0]))
    with amp_context(device):
        loss, stats = batch_objective(model, vae, norm, batch, c)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Forward check failed")
    print(f"Forward check only: {sum(p.numel() for p in model.parameters()):,} parameters; "
          f"{c.HISTORY_BINS} history + {c.TARGET_BINS} target bins; genres={len(labels)-1}; {summarize(stats)}")


def resume_setting_changes(state, c, current_model_config, world, *, allow_batch_size_change=False):
    """Permit only an explicitly requested batch-size change, not other incompatibilities."""
    ignored = {"SAVE_DIR", "RESUME_TRAINING", "NUM_WORKERS", "CACHE_MB_PER_WORKER", "LOG_DATA_ERRORS",
               "LOG_EVERY", "VALIDATE_EVERY", "SAVE_EVERY", "EVALUATION_DIR", "ACTIVITY_THRESHOLD", "ONSET_THRESHOLD"}
    changed = [k for k, value in vars(c).items() if k not in ignored and state["train_config"].get(k) != value]
    blocked = [k for k in changed if not (allow_batch_size_change and k == "BATCH_SIZE")]
    if blocked or state["world_size"] != world or state["model_config"] != current_model_config:
        raise ValueError(f"Exact stage-2 resume settings differ: {changed}. "
                         "Use --allow-batch-size-change to continue with a different BATCH_SIZE; "
                         "architecture, world size, data, normalization and optimizer settings remain protected.")
    return changed


def train(c, *, force_cpu=False, allow_batch_size_change=False):
    validate_config(c)
    rank, world, local_rank = (int(os.environ.get(k, default)) for k, default in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() and not force_cpu else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    writer, handlers, stopped = None, {}, [False]
    try:
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
        random.seed(c.SEED + rank); torch.manual_seed(c.SEED + rank)
        vae, labels = prepare(c, device)
        index_id, vae_hash = index_identity(c.INDEX_PATH), checkpoint_digest(c.VAE_CHECKPOINT)
        directory = Path(c.SAVE_DIR)
        state = None
        for name in ("latest_checkpoint.pt", "best_model.pt"):
            path = directory / name
            if path.exists():
                candidate = torch.load(path, map_location="cpu", weights_only=False)
                if candidate.get("format") != FORMAT:
                    raise ValueError("Unrelated checkpoint in SAVE_DIR; it will not be overwritten")
                if not c.RESUME_TRAINING:
                    raise FileExistsError("Use a fresh SAVE_DIR or resume existing stage-2 training")
                if state is None:
                    state = candidate
        model = LatentFlowModel(model_config(c, vae, labels)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c.LR, betas=c.BETAS, weight_decay=c.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, c))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
        step, epoch, next_batch, best_val = 0, 0, 0, float("inf")
        train_data = LatentFlowDataset(c, "train", labels=labels, error_dir=directory / "data_errors")
        val_data = LatentFlowDataset(c, "validation", labels=labels, error_dir=directory / "data_errors")
        if state:
            changed = resume_setting_changes(state, c, asdict(model.config), world,
                                             allow_batch_size_change=allow_batch_size_change)
            if state["vae_sha256"] != vae_hash or state["index_identity"] != index_id or state["genres"] != list(labels):
                raise ValueError("Frozen VAE, index or genre vocabulary changed since checkpoint")
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            scaler.load_state_dict(state["scaler_state_dict"])
            normalizer = LatentNormalizer(**state["normalizer"])
            step, epoch, next_batch, best_val = (state[k] for k in ("step", "epoch", "next_batch", "best_val"))
            if changed:
                # Batch indices depend on batch size. Do not reuse an old index
                # against the new loader or pretend this continuation is exact.
                epoch += int(next_batch > 0)
                next_batch = 0
                if rank == 0:
                    print(f"WARNING: non-exact batch-size continuation; changed={changed}. "
                          f"Keeping model/EMA, optimizer, scheduler, normalization and step {step}; "
                          f"restarting data at epoch {epoch}, batch 0. Effective batch size changes; "
                          "learning rate is not automatically rescaled.", flush=True)
        else:
            # Same rank-independent training examples; broadcast rank-0 statistics.
            if rank == 0:
                print("Calibrating frozen VAE latents on training targets only", flush=True)
                norm_state = calibrate(vae, train_data, c, device).state_dict()
            else:
                norm_state = None
            if world > 1:
                package = [norm_state]; dist.broadcast_object_list(package, src=0); norm_state = package[0]
            normalizer = LatentNormalizer(**norm_state)
        if world > 1:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        raw = model.module if isinstance(model, DDP) else model
        ema = deepcopy(raw).eval().requires_grad_(False)
        if state:
            ema.load_state_dict(state["ema_state_dict"])
            rng = state["rng"][rank]
            random.setstate(rng["python"]); torch.set_rng_state(rng["torch"].cpu())
            if device.type == "cuda" and rng["cuda"] is not None:
                torch.cuda.set_rng_state(rng["cuda"], device)
            del state
        sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=True, seed=c.SEED, drop_last=True)
        batches = ResumableBatchSampler(sampler, c.BATCH_SIZE)
        if not len(batches):
            raise ValueError("Not enough samples for a distributed batch")
        options = dict(num_workers=c.NUM_WORKERS, pin_memory=device.type == "cuda", collate_fn=collate_windows)
        loader = DataLoader(train_data, batch_sampler=batches, generator=torch.Generator().manual_seed(c.SEED), **options)
        val_loader = DataLoader(val_data, batch_size=c.VALIDATION_BATCH_SIZE, sampler=range(rank, len(val_data), world),
                               generator=torch.Generator().manual_seed(c.SEED + 1), **options)
        if rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(directory / "tensorboard"))
            print(f"Stage 2: {sum(p.numel() for p in raw.parameters()):,} parameters; frozen VAE; "
                  f"effective window batch={c.BATCH_SIZE*world*c.ACCUM_STEPS}; genres={len(labels)-1}; step={step}", flush=True)
            print(f"Genre labels: {', '.join(labels[1:])}; unlabelled/dropout examples use unconditioned ID 0", flush=True)
        model.train(); optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        stats = torch.zeros(len(STAT_KEYS), device=device, dtype=torch.float64)
        while step < c.N_STEPS:
            sampler.set_epoch(epoch); train_data.set_epoch(epoch); batches.start_batch = next_batch
            full_batches = len(sampler) // c.BATCH_SIZE
            if next_batch >= full_batches:
                raise ValueError("Invalid saved data position")
            for batch in loader:
                sync = accumulated + 1 == c.ACCUM_STEPS
                with model.no_sync() if isinstance(model, DDP) and not sync else nullcontext():
                    with amp_context(device):
                        loss, values = batch_objective(model, vae, normalizer, move_batch(batch, device), c)
                    finite = torch.isfinite(loss).int()
                    if world > 1:
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise FloatingPointError("Nonfinite loss; previous checkpoint preserved")
                    scaler.scale(loss / c.ACCUM_STEPS).backward()
                stats += values; accumulated += 1; next_batch += 1
                del loss, values, batch
                if next_batch == full_batches:
                    epoch, next_batch = epoch + 1, 0
                if not sync:
                    continue
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError("Nonfinite gradients; previous checkpoint preserved")
                scaler.step(optimizer); scaler.update(); scheduler.step()
                update_ema(ema, model, c.EMA_DECAY)
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0; step += 1
                if step % c.LOG_EVERY == 0 or step == c.N_STEPS:
                    if world > 1:
                        dist.all_reduce(stats)
                    values = summarize(stats)
                    if rank == 0:
                        print(f"step {step}: loss={values['loss']:.4f}, latent_NRMSE={values['latent_nrmse']:.3f}, "
                              f"low_noise={values.get('latent_nrmse_low', float('nan')):.3f}, retries={values['retries']:.2f}", flush=True)
                        for key, value in values.items():
                            writer.add_scalar("train/" + key, value, step)
                        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                    stats.zero_()
                best = False
                if step % c.VALIDATE_EVERY == 0 or step == c.N_STEPS:
                    values = evaluate(ema, vae, normalizer, val_loader, c, device)
                    best = values["loss"] < best_val; best_val = min(best_val, values["loss"])
                    if rank == 0:
                        print(f"validation {step}: {values}", flush=True)
                        for key, value in values.items():
                            writer.add_scalar("validation/" + key, value, step)
                request = os.environ.get("CHECKPOINT_REQUEST_FILE")
                requested = torch.tensor(int(stopped[0] or bool(request and Path(request).exists())), device=device)
                if world > 1:
                    dist.all_reduce(requested, op=dist.ReduceOp.MAX)
                if best or step % c.SAVE_EVERY == 0 or step == c.N_STEPS or requested.item():
                    save_state(model, ema, optimizer, scheduler, scaler, normalizer, c, labels, vae_hash, index_id,
                        step=step, epoch=epoch, next_batch=next_batch, best_val=best_val, device=device, best=best)
                if requested.item() or step == c.N_STEPS:
                    return
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if writer:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="improvnet.model.latent_flow_config")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--allow-batch-size-change", action="store_true",
                        help="Resume with changed BATCH_SIZE, preserving training state but restarting data at an epoch boundary")
    for name in ("save-dir", "vae-checkpoint", "index"):
        parser.add_argument("--" + name)
    for name in ("batch-size", "num-workers", "steps", "accum-steps"):
        parser.add_argument("--" + name, type=int)
    args = parser.parse_args(argv)
    module = importlib.import_module(args.config)
    c = SimpleNamespace(**{k: value for k, value in vars(module).items() if k.isupper()})
    for name, key in (("save_dir", "SAVE_DIR"), ("vae_checkpoint", "VAE_CHECKPOINT"), ("index", "INDEX_PATH"),
                      ("batch_size", "BATCH_SIZE"), ("num_workers", "NUM_WORKERS"), ("steps", "N_STEPS"), ("accum_steps", "ACCUM_STEPS")):
        if getattr(args, name) is not None:
            setattr(c, key, getattr(args, name))
    if args.check:
        check(c, torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda"))
    else:
        train(c, force_cpu=args.cpu, allow_batch_size_change=args.allow_batch_size_change)


if __name__ == "__main__":
    main()
