"""Fresh masked diffusion training, following the AR optimizer/DDP/checkpoint layout."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import importlib
import math
import os
from pathlib import Path
import random
import signal

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from improvnet.model.ar_train import build_scheduler
from improvnet.model.masked_diffusion_model import MaskedDiffusionModel, model_config
from improvnet.utils.masked_diffusion import batch_objective
from improvnet.utils.masked_diffusion_data import MaskedDiffusionDataset, GenreSampler, collate_sequences

FORMAT = "masked_diffusion_v1"


def amp_context(device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


def validate_config(c):
    for key in ("BATCH_SIZE", "ACCUM_STEPS", "N_STEPS", "SEQ_LEN", "LOG_EVERY", "VAL_EVERY", "CHECKPOINT_EVERY"):
        if not isinstance(getattr(c, key), int) or getattr(c, key) < 1:
            raise ValueError(f"{key} must be positive")
    if c.NUM_WORKERS < 0 or not math.isfinite(c.JAZZ_BLUES_WEIGHT) or c.JAZZ_BLUES_WEIGHT <= 0:
        raise ValueError("Invalid workers/genre weight")
    if not 0 <= c.ELASTICITY_MAX <= 1 or not 0 <= c.RHYTHM_DROP_PROB <= 1:
        raise ValueError("Invalid elasticity/rhythm dropout")
    if not 0 < c.MIN_LR <= c.LR or c.GRAD_CLIP <= 0:
        raise ValueError("Invalid optimizer settings")
    if c.SEQ_LEN < 8 or c.PAD_ID != 2 or c.MASK_ID != 5 or c.BLANK_ID != 6:
        raise ValueError("Expected the AR tokenizer's special IDs and room for full notes")


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def summarize(stats):
    n = max(1, float(stats[1]))
    return dict(loss=float(stats[0]) / n,
                masked_accuracy=float(stats[2]) / max(1, float(stats[3])))


def data_identity(c):
    return {path: (os.stat(path).st_size, os.stat(path).st_mtime_ns,
        os.stat(path.replace('.jsonl', '_index.pkl')).st_mtime_ns) for path in c.JSONL_FILES}


def resume_plan(state, c, current_model_config, world, batches_per_epoch=None):
    """Keep optimizer progress when GPU/batch layout changes, as in the AR trainer.

    Match the saved global data position to a complete batch in the new layout.
    This can repeat fewer than one new global batch and is not an exact replay.
    """
    ignored = {"NUM_WORKERS", "LOG_EVERY", "VAL_EVERY", "CHECKPOINT_EVERY",
               "RESUME_TRAINING", "SAVE_DIR", "BATCH_SIZE", "ACCUM_STEPS"}
    settings = {k: v for k, v in vars(c).items() if k.isupper() and k not in ignored}
    saved = {k: v for k, v in state.get("train_config", {}).items() if k not in ignored}
    problems = []
    if state.get("format") != FORMAT:
        problems.append("checkpoint format")
    if state.get("model_config") != current_model_config:
        problems.append("model architecture")
    problems.extend(k for k in sorted(settings.keys() | saved.keys()) if settings.get(k) != saved.get(k))
    if state.get("data_identity") != data_identity(c):
        problems.append("data files/index identity")
    if problems:
        raise ValueError("Incompatible masked-diffusion checkpoint/settings/data: " +
                         ", ".join(problems) + ". Use a fresh SAVE_DIR for these changes.")
    old_world = state["world_size"]
    old_batch = state["train_config"]["BATCH_SIZE"]
    old_accum = state["train_config"]["ACCUM_STEPS"]
    changed = (old_world, old_batch, old_accum) != (world, c.BATCH_SIZE, c.ACCUM_STEPS)
    epoch, next_batch = state["epoch"], state["next_batch"]
    repeated = 0
    if changed:
        consumed = next_batch * old_batch * old_world
        next_batch = consumed // (c.BATCH_SIZE * world)
        repeated = consumed - next_batch * c.BATCH_SIZE * world
        if batches_per_epoch is not None and next_batch >= batches_per_epoch:
            epoch, next_batch = epoch + 1, 0
    return dict(changed=changed, epoch=epoch, next_batch=next_batch,
        repeated_examples=repeated, old_world=old_world, new_world=world,
        old_effective_batch=old_world * old_batch * old_accum,
        new_effective_batch=world * c.BATCH_SIZE * c.ACCUM_STEPS)


def restore_checkpoint_rng(state, c, rank, device):
    rank_states = state.get("rng", [])
    if rank < len(rank_states):
        rng = rank_states[rank]
        random.setstate(rng["python"]); torch.set_rng_state(rng["torch"].cpu())
        if device.type == "cuda" and rng.get("cuda") is not None:
            torch.cuda.set_rng_state(rng["cuda"], device)
    else:
        # Additional ranks must not clone rank zero's random corruption stream.
        seed = c.SEED + 1000003 * state["step"] + rank
        random.seed(seed); torch.manual_seed(seed)


@torch.inference_mode()
def evaluate(model, loader, device, seed):
    training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    stats = torch.zeros(4, device=device, dtype=torch.float64)
    try:
        # Full held-out dataset, no genre oversampling and fixed corruption draws.
        for batch in loader:
            with amp_context(device):
                raw = model.module if isinstance(model, DDP) else model
                _, values = batch_objective(raw, move_batch(batch, device), generator=generator)
            stats += values
        if dist.is_initialized():
            dist.all_reduce(stats)
        if not bool(torch.isfinite(stats).all()) or stats[1] == 0:
            raise ValueError("Empty/nonfinite validation")
        return summarize(stats)
    finally:
        model.train(training)


def _rng(device):
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None)


def save_state(model, optimizer, scheduler, scaler, c, *, step, epoch, next_batch, best_val, device, best=False):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    rng = [None] * world
    if world > 1:
        dist.all_gather_object(rng, _rng(device))
    else:
        rng[0] = _rng(device)
    if rank:
        return
    raw = model.module if isinstance(model, DDP) else model
    state = dict(format=FORMAT, model_config=asdict(raw.config),
        train_config={k: v for k, v in vars(c).items() if k.isupper()},
        model_state_dict=raw.state_dict(), optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(), scaler_state_dict=scaler.state_dict(),
        step=step, epoch=epoch, next_batch=next_batch, best_val=best_val, world_size=world, rng=rng,
        data_identity=data_identity(c))
    directory = Path(c.SAVE_DIR); directory.mkdir(parents=True, exist_ok=True)
    for name in (["latest_checkpoint.pt", "best_model.pt"] if best else ["latest_checkpoint.pt"]):
        pending = directory / (name + f".tmp.{os.getpid()}")
        try:
            torch.save(state, pending); os.replace(pending, directory / name)
        finally:
            pending.unlink(missing_ok=True)


def train(c, *, force_cpu=False):
    validate_config(c)
    rank, world, local_rank = (int(os.environ.get(k, default)) for k, default in
                             (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() and not force_cpu else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    writer, handlers, stopped = None, {}, [False]
    data, validation = None, None
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            handlers[sig] = signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
        random.seed(c.SEED + rank); torch.manual_seed(c.SEED + rank)
        data = MaskedDiffusionDataset(c, "train")
        validation = MaskedDiffusionDataset(c, "validation")
        sampler = GenreSampler(data.weights, c.BATCH_SIZE, c.SEED, rank, world)
        model = MaskedDiffusionModel(model_config(c)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=c.LR, betas=c.BETAS, weight_decay=c.WEIGHT_DECAY)
        scheduler = build_scheduler(optimizer, c.WARMUP_STEPS, c.N_STEPS, c.MIN_LR, c.DECAY_STEPS)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
        step, epoch, next_batch, best_val = 0, 0, 0, float("inf")
        directory = Path(c.SAVE_DIR)
        latest = directory / "latest_checkpoint.pt"
        if latest.exists():
            if not c.RESUME_TRAINING:
                raise FileExistsError("Use a fresh SAVE_DIR or enable resume")
            state = torch.load(latest, map_location="cpu", weights_only=False)
            plan = resume_plan(state, c, asdict(model.config), world, sampler.size // c.BATCH_SIZE)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"]); scaler.load_state_dict(state["scaler_state_dict"])
            step, best_val = state["step"], state["best_val"]
            epoch, next_batch = plan["epoch"], plan["next_batch"]
            restore_checkpoint_rng(state, c, rank, device)
            if rank == 0 and plan["changed"]:
                print(f"Continuing checkpoint step {step}: GPUs {plan['old_world']} -> {world}; "
                      f"effective batch {plan['old_effective_batch']} -> {plan['new_effective_batch']}. "
                      "Model, optimizer and scheduler restored; learning rate unchanged. "
                      f"Data continuation is approximate; up to {plan['repeated_examples']} examples "
                      "are replayed at the new batch boundary. New ranks use separate random seeds.", flush=True)
            del state
        elif (directory / "best_model.pt").exists():
            raise FileExistsError("Best checkpoint exists without latest checkpoint; use a fresh directory")
        if world > 1:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        raw = model.module if isinstance(model, DDP) else model
        options = dict(num_workers=c.NUM_WORKERS, pin_memory=device.type == "cuda", collate_fn=collate_sequences)
        loader = DataLoader(data, batch_size=c.BATCH_SIZE, sampler=sampler, drop_last=True,
                            generator=torch.Generator().manual_seed(c.SEED), **options)
        val_loader = DataLoader(validation, batch_size=c.BATCH_SIZE, sampler=range(rank, len(validation), world),
                                generator=torch.Generator().manual_seed(c.SEED + 1), **options)
        if rank == 0:
            writer = SummaryWriter(str(directory / "tensorboard"))
            print(f"Masked diffusion: {sum(p.numel() for p in raw.parameters()):,} parameters; "
                  f"effective batch={c.BATCH_SIZE * world * c.ACCUM_STEPS}; step={step}; jazz/blues weight={c.JAZZ_BLUES_WEIGHT}", flush=True)
        model.train(); optimizer.zero_grad(set_to_none=True)
        accumulated, stats = 0, torch.zeros(4, device=device, dtype=torch.float64)
        while step < c.N_STEPS:
            data.set_epoch(epoch); sampler.epoch, sampler.start = epoch, next_batch * c.BATCH_SIZE
            full_batches = sampler.size // c.BATCH_SIZE
            if not 0 <= next_batch < full_batches:
                raise ValueError("Invalid saved data position")
            for batch in loader:
                sync = accumulated + 1 == c.ACCUM_STEPS
                with model.no_sync() if isinstance(model, DDP) and not sync else nullcontext():
                    with amp_context(device):
                        loss, values = batch_objective(model, move_batch(batch, device))
                    finite = torch.isfinite(loss).int()
                    if world > 1:
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise FloatingPointError("Nonfinite masked diffusion loss")
                    scaler.scale(loss / c.ACCUM_STEPS).backward()
                stats += values; accumulated += 1; next_batch += 1
                if next_batch == full_batches:
                    epoch, next_batch = epoch + 1, 0
                if not sync:
                    continue
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("Nonfinite gradients")
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                accumulated = 0; step += 1
                if step % c.LOG_EVERY == 0:
                    if world > 1:
                        dist.all_reduce(stats)
                    metrics = summarize(stats)
                    if rank == 0:
                        print(f"step {step}: {metrics}", flush=True)
                        for key, value in metrics.items():
                            writer.add_scalar("train/" + key, value, step)
                        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                    stats.zero_()
                best = False
                if step % c.VAL_EVERY == 0 or step == c.N_STEPS:
                    metrics = evaluate(model, val_loader, device, c.SEED + rank + 9001)
                    best = metrics["loss"] < best_val; best_val = min(best_val, metrics["loss"])
                    if rank == 0:
                        print(f"validation {step}: {metrics}", flush=True)
                        for key, value in metrics.items():
                            writer.add_scalar("validation/" + key, value, step)
                request = os.environ.get("CHECKPOINT_REQUEST_FILE")
                requested = torch.tensor(int(stopped[0] or bool(request and Path(request).exists())), device=device)
                if world > 1:
                    dist.all_reduce(requested, op=dist.ReduceOp.MAX)
                if best or step % c.CHECKPOINT_EVERY == 0 or step == c.N_STEPS or requested.item():
                    save_state(model, optimizer, scheduler, scaler, c, step=step, epoch=epoch,
                        next_batch=next_batch, best_val=best_val, device=device, best=best)
                if requested.item() or step == c.N_STEPS:
                    return
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if writer is not None:
            writer.close()
        for dataset in (data, validation):
            if dataset is not None:
                dataset.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="improvnet.model.masked_diffusion_config")
    parser.add_argument("--check", action="store_true", help="Check config/data paths without creating a model or starting training")
    args = parser.parse_args(argv)
    c = importlib.import_module(args.config)
    validate_config(c)
    if args.check:
        for path in c.JSONL_FILES:
            if not Path(path).exists() or not Path(path.replace('.jsonl', '_index.pkl')).exists():
                raise FileNotFoundError(f"Missing tokenized AR data or byte index: {path}")
        print("Masked diffusion config and data paths checked; training was not started.")
        return
    train(c)


if __name__ == "__main__":
    main()
