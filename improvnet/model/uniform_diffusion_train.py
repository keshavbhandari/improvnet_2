"""Shared-backbone uniform diffusion SFT. Run with python -m or torchrun.

Fresh runs copy the AR backbone; resumes restore this model AND optimizer.
Nothing in this module starts training until main() is explicitly invoked.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import importlib
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from improvnet.model.uniform_diffusion_model import UniformDiffusionModel
from improvnet.utils.ar_utils import collect_rng_state_for_checkpoint
from improvnet.utils.uniform_diffusion_corruption import reconstruction_losses, uniform_corrupt
from improvnet.utils.uniform_diffusion_data import UniformDiffusionCollator, UniformDiffusionDataset


class ResumableBatchSampler(BatchSampler):
    """Skip index batches, not loaded examples (no augmentation/RNG replay)."""
    def __init__(self, sampler, batch_size, start_batch=0):
        super().__init__(sampler, batch_size, drop_last=True)
        self.start_batch = start_batch

    def __iter__(self):
        for index, batch in enumerate(super().__iter__()):
            if index >= self.start_batch:
                yield batch

    def __len__(self):
        return max(0, super().__len__() - self.start_batch)


def lr_factor(step, config):
    if step < config.WARMUP_STEPS:
        return (step + 1) / max(1, config.WARMUP_STEPS)
    progress = min(1., (step - config.WARMUP_STEPS) / max(1, config.N_STEPS - config.WARMUP_STEPS))
    floor = config.MIN_LR / config.LR
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def batch_objective(model, batch, config, *, generator=None):
    corruption = uniform_corrupt(
        batch["target_ids"], batch["target_mask"], config.MODEL.vocab_size, generator=generator,
    )
    sc_mask = torch.rand(batch["genre"].shape, device=batch["genre"].device, generator=generator) < config.SELF_CONDITIONING_PROB
    output = model(
        batch["clean_ids"], corruption.tokens,
        genre=batch["genre"], multi_hot=batch["multi_hot"], elasticity=batch["elasticity"],
        context_attention_mask=batch["clean_mask"], canvas_attention_mask=batch["target_mask"],
        prefix_lengths=batch["prefix_lengths"], return_context_logits=config.AR_LOSS_WEIGHT > 0,
        bootstrap_self_conditioning=True, self_conditioning_mask=sc_mask,
        self_conditioning_temperature=config.SELF_CONDITIONING_TEMPERATURE,
    )
    denoising = reconstruction_losses(output.logits, batch["target_ids"], batch["target_mask"], config.CE_CHUNK_SIZE)
    ar = torch.zeros_like(denoising)
    if config.AR_LOSS_WEIGHT > 0:
        # Musical position j predicts j+1. Never train a pad or the final
        # position without a next-token label. Controls have no target labels.
        ar = reconstruction_losses(
            output.context.logits[:, :-1], batch["clean_ids"][:, 1:],
            batch["clean_mask"][:, :-1] & batch["clean_mask"][:, 1:], config.CE_CHUNK_SIZE,
        )
    total = denoising + config.AR_LOSS_WEIGHT * ar
    with torch.no_grad():
        mask = batch["target_mask"]
        correct = output.logits.argmax(-1).eq(batch["target_ids"]) & mask
        changed = corruption.changed
        # Sum/count statistics, reducible across ranks and unequal final val batches.
        stats = torch.stack((
            total.detach().sum(), denoising.detach().sum(), ar.detach().sum(),
            total.new_tensor(total.numel()), correct.sum(), mask.sum(),
            (correct & changed).sum(), changed.sum(), corruption.timesteps.sum(),
        )).double()
    return total.mean(), stats


def summarize(stats):
    s = stats.tolist()
    return dict(loss=s[0] / max(s[3], 1), denoising=s[1] / max(s[3], 1),
                ar=s[2] / max(s[3], 1), accuracy=s[4] / max(s[5], 1),
                changed_accuracy=s[6] / max(s[7], 1), mean_t=s[8] / max(s[3], 1))


def amp_context(device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def evaluate(model, loader, config, device, rank=0):
    # No DDP forward collectives: validation shards can have unequal lengths.
    raw_model = model.module if isinstance(model, DDP) else model
    was_training = raw_model.training
    raw_model.eval()
    totals = torch.zeros(9, device=device, dtype=torch.float64)
    generator = torch.Generator(device=device).manual_seed(config.SEED + 500003 + rank)
    try:
        for index, batch in enumerate(loader):
            if index >= config.MAX_VALIDATION_BATCHES:
                break
            with amp_context(device):
                _, stats = batch_objective(raw_model, to_device(batch, device), config, generator=generator)
            totals += stats
        if dist.is_initialized():
            dist.all_reduce(totals)
        if not bool(torch.isfinite(totals).all()) or totals[3] == 0:
            raise RuntimeError("Validation produced non-finite results or no examples")
        return summarize(totals)
    finally:
        raw_model.train(was_training)


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def save_training_state(model, optimizer, scheduler, scaler, config, *, step,
                        epoch, next_batch, best_val, device, is_best=False):
    rng_state = collect_rng_state_for_checkpoint(device)  # ALL ranks participate.
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    raw_model = model.module if isinstance(model, DDP) else model
    state = {
        "format": "uniform_diffusion_sft_v1", "model_config": asdict(config.MODEL),
        "train_config": {k: v for k, v in vars(config).items() if k != "MODEL"},
        "model_state_dict": _cpu_tree(raw_model.state_dict()),
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
        "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
        "step": step, "epoch": epoch, "next_batch": next_batch, "best_val": best_val,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "rng_state": rng_state,
    }
    directory = Path(config.SAVE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    for name in (["latest_checkpoint.pt", "best_model.pt"] if is_best else ["latest_checkpoint.pt"]):
        path = directory / name
        temporary = directory / f"{name}.tmp.{os.getpid()}"
        torch.save(state, temporary)
        os.replace(temporary, path)


def restore_training_state(state, model, optimizer, scheduler, scaler, config, device, rank, world):
    if state.get("format") != "uniform_diffusion_sft_v1" or state["model_config"] != asdict(config.MODEL):
        raise ValueError("Resume checkpoint is not a compatible uniform diffusion model")
    # Exact data position and scheduler/RNG continuation require this run shape.
    keys = ("BATCH_SIZE", "ACCUM_STEPS", "SEED", "JSONL_FILES", "DATA_SPLIT_SEED",
            "DATA_SPLIT_RATIOS", "MIDI_FILEPATH_EXCLUDE_SUBSTRINGS", "PROMPT_MAX",
            "PROMPTLESS_PROB", "END_ANCHORED_TARGET_PROB", "PITCH_AUGMENTATION",
            "SELF_CONDITIONING_PROB", "SELF_CONDITIONING_TEMPERATURE", "AR_LOSS_WEIGHT", "LR", "MIN_LR", "WARMUP_STEPS",
            "N_STEPS", "BETAS", "WEIGHT_DECAY", "GRAD_CLIP")
    saved_config = dict(state["train_config"])
    # Earlier checkpoints used an implicit fixed temperature of one.
    saved_config.setdefault("SELF_CONDITIONING_TEMPERATURE", 1.0)
    changed = [key for key in keys if saved_config.get(key) != getattr(config, key)]
    if state["world_size"] != world or changed:
        raise ValueError(f"Exact resume requires unchanged world size and settings; changed={changed}")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    scaler.load_state_dict(state["scaler_state_dict"])
    rng = state["rng_state"]["rank_states"][rank]
    random.setstate(rng["python_random_state"])
    torch.set_rng_state(rng["torch_rng_state"].cpu())
    if device.type == "cuda" and rng["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(rng["cuda_rng_state"].cpu(), device)
    return state["step"], state["epoch"], state["next_batch"], state["best_val"]


def validate_config(config):
    for key in ("BATCH_SIZE", "ACCUM_STEPS", "VALIDATION_BATCH_SIZE", "MAX_VALIDATION_BATCHES",
                "N_STEPS", "CE_CHUNK_SIZE", "LOG_EVERY", "SAVE_EVERY", "VALIDATE_EVERY"):
        if getattr(config, key) < 1:
            raise ValueError(f"{key} must be positive")
    if config.NUM_WORKERS < 0 or config.PROMPT_MAX < 0 or config.WARMUP_STEPS < 0:
        raise ValueError("Worker count, prompt cap and warmup must be nonnegative")
    for key in ("PROMPTLESS_PROB", "END_ANCHORED_TARGET_PROB", "SELF_CONDITIONING_PROB"):
        if not 0 <= getattr(config, key) <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if not math.isfinite(config.SELF_CONDITIONING_TEMPERATURE) or config.SELF_CONDITIONING_TEMPERATURE <= 0:
        raise ValueError("SELF_CONDITIONING_TEMPERATURE must be finite and positive")
    if not (0 < config.MIN_LR <= config.LR) or config.AR_LOSS_WEIGHT < 0 or config.GRAD_CLIP <= 0:
        raise ValueError("Invalid learning rates, AR loss weight or gradient clip")
    if Path(config.SAVE_DIR).resolve() == Path(config.AR_MODEL_PATH).resolve().parent:
        raise ValueError("Diffusion checkpoints must not overwrite the source AR directory")


def train(config):
    validate_config(config)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    writer = None
    try:
        random.seed(config.SEED + rank)
        torch.manual_seed(config.SEED + rank)
        resume_path = Path(config.SAVE_DIR) / "latest_checkpoint.pt"
        if resume_path.exists() and not config.RESUME_TRAINING:
            raise FileExistsError("Choose a fresh SAVE_DIR for a non-resuming run")
        state = torch.load(resume_path, map_location="cpu", weights_only=False, mmap=True) if resume_path.exists() else None
        model = UniformDiffusionModel(config.MODEL)
        if state is None:
            ar_state = torch.load(config.AR_MODEL_PATH, map_location="cpu", weights_only=False, mmap=True)
            model.initialize_from_ar(ar_state)
            del ar_state
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, betas=config.BETAS, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, config))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
        step, epoch, next_batch, best_val = 0, 0, 0, float("inf")
        if state is not None:
            step, epoch, next_batch, best_val = restore_training_state(
                state, model, optimizer, scheduler, scaler, config, device, rank, world,
            )
            del state
        if world > 1:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        train_data = UniformDiffusionDataset(config, "train")
        val_data = UniformDiffusionDataset(config, "validation", processor=train_data.processor)
        processor = train_data.processor
        if config.MODEL.vocab_size != processor.tokenizer.vocab_size or config.MODEL.num_genres != len(processor.genres) or config.MODEL.num_instruments != len(processor.INSTRUMENT_CLASSES):
            raise ValueError("Model vocabulary/conditioning sizes must match the music tokenizer")
        collator = UniformDiffusionCollator(config.MODEL.canvas_size, processor.tokenizer.tok_to_id["<P>"])
        sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=True, seed=config.SEED, drop_last=True)
        batches = ResumableBatchSampler(sampler, config.BATCH_SIZE)
        if len(batches) == 0:
            raise ValueError("Dataset is too small for one distributed training batch")
        # Dedicated loader RNGs prevent iterator construction from disturbing
        # model/noise RNG state, including when resuming mid-epoch.
        loader_options = dict(num_workers=config.NUM_WORKERS, pin_memory=device.type == "cuda", collate_fn=collator)
        train_loader = DataLoader(train_data, batch_sampler=batches, generator=torch.Generator().manual_seed(config.SEED), **loader_options)
        val_loader = DataLoader(val_data, batch_size=config.VALIDATION_BATCH_SIZE,
                                sampler=range(rank, len(val_data), world),
                                generator=torch.Generator().manual_seed(config.SEED + 1), **loader_options)
        if rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(Path(config.SAVE_DIR) / "tensorboard"))
            print(f"Uniform diffusion SFT: shared trainable backbone; effective batch={config.BATCH_SIZE * config.ACCUM_STEPS * world}; resumed step={step}", flush=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        metrics = torch.zeros(9, device=device, dtype=torch.float64)
        while step < config.N_STEPS:
            sampler.set_epoch(epoch)
            train_data.set_epoch(epoch)
            batches.start_batch = next_batch
            full_epoch_batches = len(sampler) // config.BATCH_SIZE
            if next_batch >= full_epoch_batches:
                raise ValueError("Invalid checkpoint data position")
            for batch in train_loader:
                sync = accumulated + 1 == config.ACCUM_STEPS
                sync_context = model.no_sync() if isinstance(model, DDP) and not sync else nullcontext()
                with sync_context:
                    with amp_context(device):
                        loss, stats = batch_objective(model, to_device(batch, device), config)
                    finite = torch.isfinite(loss).to(torch.int32)
                    if world > 1:
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise FloatingPointError("Non-finite loss; stopping without overwriting the last checkpoint")
                    scaler.scale(loss / config.ACCUM_STEPS).backward()
                metrics += stats
                del loss, stats, batch
                accumulated += 1
                next_batch += 1
                if next_batch == full_epoch_batches:
                    epoch, next_batch = epoch + 1, 0
                if not sync:
                    continue
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                if not bool(torch.isfinite(norm)):
                    if scaler.is_enabled():
                        # GradScaler skips this overflowing update. Do not
                        # advance the optimizer-step schedule or save bad state.
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        accumulated = 0
                        metrics.zero_()
                        continue
                    raise FloatingPointError("Non-finite gradients; checkpoint not overwritten")
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
                        print(f"step {step}: loss={values['loss']:.4f} denoise={values['denoising']:.4f} AR={values['ar']:.4f} changed_acc={values['changed_accuracy']:.3f}", flush=True)
                        for key, value in values.items():
                            writer.add_scalar(f"train/{key}", value, step)
                        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                    metrics.zero_()
                is_best = False
                if step % config.VALIDATE_EVERY == 0 or step == config.N_STEPS:
                    # Gradients and forward references have been released before
                    # validation. Validation has its own smaller batch and RNG.
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    values = evaluate(model, val_loader, config, device, rank)
                    is_best = values["denoising"] < best_val
                    best_val = min(best_val, values["denoising"])
                    if rank == 0:
                        print(f"validation {step}: {values}", flush=True)
                        for key, value in values.items():
                            writer.add_scalar(f"validation/{key}", value, step)
                request = os.environ.get("CHECKPOINT_REQUEST_FILE")
                requested = torch.tensor(int(bool(request and Path(request).exists())), device=device)
                if world > 1:
                    dist.all_reduce(requested, op=dist.ReduceOp.MAX)
                if is_best or step % config.SAVE_EVERY == 0 or step == config.N_STEPS or requested.item():
                    save_training_state(model, optimizer, scheduler, scaler, config, step=step,
                                        epoch=epoch, next_batch=next_batch, best_val=best_val,
                                        device=device, is_best=is_best)
                if requested.item() or step == config.N_STEPS:
                    if rank == 0 and requested.item():
                        print("Checkpoint request handled at an optimizer boundary; exiting.", flush=True)
                    return
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="improvnet.model.uniform_diffusion_train_config")
    parser.add_argument("--ar-model")
    parser.add_argument("--save-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accum-steps", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    module = importlib.import_module(args.config)
    config = SimpleNamespace(**{k: v for k, v in vars(module).items() if k.isupper()})
    for argument, key in (("ar_model", "AR_MODEL_PATH"), ("save_dir", "SAVE_DIR"),
                          ("batch_size", "BATCH_SIZE"), ("accum_steps", "ACCUM_STEPS"), ("steps", "N_STEPS")):
        if getattr(args, argument) is not None:
            setattr(config, key, getattr(args, argument))
    if args.no_resume:
        config.RESUME_TRAINING = False
    train(config)


if __name__ == "__main__":
    main()
