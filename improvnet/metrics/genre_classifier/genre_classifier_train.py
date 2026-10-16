"""Standalone genre-encoder training: python -m or torchrun, never on import.

Matches the stage-2 trainer's config/DDP/AMP/checkpoint organization, but
uses independent parameters, class-balanced music excerpts and classification
CE. It does not load AR weights or alter shared data/preprocessing.
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
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader

from improvnet.metrics.genre_classifier.data import BalancedGenreSampler, GenreCollator, GenreDataset
from improvnet.metrics.genre_classifier.evaluation import class_weights, classification_metrics, evaluate, fit_temperature
from improvnet.metrics.genre_classifier.genre_classifier_model import GenreClassifier, GenreClassifierConfig


class ResumableBatchSampler(BatchSampler):
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
        return (step + 1) / max(config.WARMUP_STEPS, 1)
    progress = min(1., (step - config.WARMUP_STEPS) / max(1, config.N_STEPS - config.WARMUP_STEPS))
    floor = config.MIN_LR / config.LR
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress))


def amp_context(device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(item) for item in value)
    return value


def validate_config(config):
    for key in ("BATCH_SIZE", "ACCUM_STEPS", "VALIDATION_BATCH_SIZE", "N_STEPS", "LOG_EVERY", "VALIDATE_EVERY", "SAVE_EVERY", "SAMPLES_PER_EPOCH", "MIN_LENGTH", "MAX_LENGTH", "EVAL_LENGTH"):
        value = getattr(config, key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("NUM_WORKERS", "WARMUP_STEPS", "MAX_VALIDATION_BATCHES"):
        value = getattr(config, key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config.WARMUP_STEPS >= config.N_STEPS:
        raise ValueError("Warmup must be shorter than training")
    if config.MIN_LENGTH > config.MAX_LENGTH or config.EVAL_LENGTH > config.MAX_LENGTH:
        raise ValueError("Invalid crop lengths")
    if not (math.isfinite(config.LR) and 0 < config.MIN_LR <= config.LR and math.isfinite(config.GRAD_CLIP) and config.GRAD_CLIP > 0):
        raise ValueError("Invalid learning rate/gradient clip")
    if not math.isfinite(config.WEIGHT_DECAY) or config.WEIGHT_DECAY < 0 or len(config.BETAS) != 2 or any(not 0 <= x < 1 for x in config.BETAS):
        raise ValueError("Invalid AdamW options")
    if not isinstance(config.PITCH_AUGMENTATION, bool) or not isinstance(config.MAX_SEMITONE_SHIFT, int) or isinstance(config.MAX_SEMITONE_SHIFT, bool) or config.MAX_SEMITONE_SHIFT < 0:
        raise ValueError("Invalid pitch augmentation settings")
    if not math.isfinite(config.FOCUS_SAMPLING_WEIGHT) or config.FOCUS_SAMPLING_WEIGHT <= 0:
        raise ValueError("Invalid focus sampling weight")


def _config_dict(config):
    return {key: value for key, value in vars(config).items() if key.isupper()}


def format_classification_metrics(metrics):
    """Compact console scores; unsupported genres/groups are explicitly N/A."""
    def number(value):
        return "N/A" if value is None else f"{value:.3f}"

    groups = "; ".join(
        f"{name}: accuracy={number(values['accuracy'])}, macro_f1={number(values['macro_f1'])}"
        for name, values in metrics["groups"].items()
    )
    genres = "; ".join(
        f"{genre}: accuracy={number(values['accuracy'])}, f1={number(values['f1']) if values['support'] else 'N/A'}"
        for genre, values in metrics["per_genre"].items()
    )
    return groups + "\n  per_genre: " + genres


def log_classification_metrics(writer, prefix, metrics, step):
    """Log defined group and individual-genre scores without replacing old tags."""
    for group, values in metrics["groups"].items():
        for key, value in values.items():
            if value is not None:
                writer.add_scalar(f"{prefix}/groups/{group}/{key}", value, step)
    for genre, values in metrics["per_genre"].items():
        for key, value in values.items():
            if value is not None:
                writer.add_scalar(f"{prefix}/{key}/{genre}", value, step)


def prepare_run_directory(directory, *, resuming):
    """Reuse interrupted debug directories without mixing logs or losing artifacts.

    Called on rank zero only. A latest checkpoint is handled by the caller's
    strict resume checks; orphaned outputs are preserved for fresh starts.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if resuming:
        return None
    artifacts = [directory / name for name in (
        "metrics.jsonl", "tensorboard", "best_checkpoint.pt", "calibrated_checkpoint.pt",
    ) if (directory / name).exists()]
    if not artifacts:
        return None
    archive_root = directory / "previous_runs"
    archive_root.mkdir(exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix="interrupted-", dir=archive_root))
    for artifact in artifacts:
        artifact.rename(archive / artifact.name)
    return archive


def train(config):
    validate_config(config)
    world, rank, local_rank = [int(os.environ.get(key, default)) for key, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"))]
    device = torch.device(getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        device = torch.device("cuda", local_rank if world > 1 or device.index is None else device.index)
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    writer, train_data, val_data, log = None, None, None, None
    try:
        random.seed(config.SEED + rank)
        torch.manual_seed(config.SEED + rank)
        directory = Path(config.SAVE_DIR)
        resume_path = directory / "latest_checkpoint.pt"
        if not config.RESUME_TRAINING and resume_path.exists():
            raise FileExistsError("Choose a fresh SAVE_DIR for a non-resuming run")
        state = torch.load(resume_path, map_location="cpu", weights_only=True, mmap=True) if resume_path.exists() else None
        train_data = GenreDataset(config.MANIFEST, "train", min_length=config.MIN_LENGTH,
                                  max_length=config.MAX_LENGTH, eval_length=config.EVAL_LENGTH,
                                  seed=config.SEED, class_caps=config.TRAIN_CLASS_CAPS,
                                  pitch_augmentation=config.PITCH_AUGMENTATION, max_semitone_shift=config.MAX_SEMITONE_SHIFT)
        val_data = GenreDataset(config.MANIFEST, "validation", tokenizer=train_data.tokenizer,
                                min_length=config.MIN_LENGTH, max_length=config.MAX_LENGTH,
                                eval_length=config.EVAL_LENGTH, seed=config.SEED)
        labels = train_data.labels
        model_config = GenreClassifierConfig.from_settings(
            config, vocab_size=train_data.tokenizer.vocab_size,
            num_genres=len(labels), pad_id=train_data.tokenizer.pad_id,
        )
        sampler = BalancedGenreSampler(train_data, config.SAMPLES_PER_EPOCH, focus_genres=config.FOCUS_GENRES,
                                       focus_weight=config.FOCUS_SAMPLING_WEIGHT, seed=config.SEED, rank=rank, world_size=world)
        batches = ResumableBatchSampler(sampler, config.BATCH_SIZE)
        full_epoch_batches = len(sampler) // config.BATCH_SIZE
        if not full_epoch_batches:
            raise ValueError("Sample budget is too small for a batch on every rank")
        model = GenreClassifier(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY, betas=config.BETAS)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, config))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported())
        # Correct balanced sampling back to the known-label held-out priors.
        # Unknown records are not a genre and never enter the benchmark.
        reference = torch.tensor([val_data.class_counts[label] for label in labels], dtype=torch.float32)
        if bool((reference == 0).any()):
            raise ValueError("Every genre needs held-out support for probability calibration")
        reference /= reference.sum()
        weights = class_weights([train_data.class_counts[label] for label in labels], getattr(config, "CLASS_BALANCE", "none")).to(device)
        prior_correction = (sampler.class_probabilities.float().log() + weights.cpu().log() - reference.log()).to(device)
        step, epoch, next_batch, best = 0, 0, 0, -1.
        if state is not None:
            saved_config, current_config = dict(state.get("train_config", {})), _config_dict(config)
            # Older checkpoints kept architecture only in the model-config dict.
            # Normalize that layout without relaxing any resume compatibility checks.
            for key, field in (("EMBED_DIM", "embed_dim"), ("N_HEADS", "n_heads"),
                               ("N_KV_HEADS", "n_kv_heads"), ("N_LAYERS", "n_layers"),
                               ("FFN_MULT", "ffn_mult"), ("DROPOUT", "dropout"),
                               ("GRADIENT_CHECKPOINTING", "gradient_checkpointing"),
                               ("MAX_LENGTH", "max_length")):
                if key not in saved_config and field in state.get("config", {}):
                    saved_config[key] = state["config"][field]
            flexible = {"SAVE_DIR", "NUM_WORKERS", "LOG_EVERY", "VALIDATE_EVERY", "SAVE_EVERY", "TENSORBOARD", "CALIBRATE", "RESUME_TRAINING", "DEVICE", "JSONL_FILES"}
            changed = sorted(key for key in set(saved_config) | set(current_config) if key not in flexible and saved_config.get(key) != current_config.get(key))
            if state.get("format") != "genre_encoder_v2" or state.get("config") != asdict(model_config) or state.get("labels") != labels or state.get("vocabulary_digest") != train_data.vocab_digest or state.get("data_fingerprints") != [train_data.data_fingerprint, val_data.data_fingerprint] or state.get("world_size") != world or changed:
                raise ValueError(f"Resume requires compatible model/data/sampling/world size/settings; changed={changed}")
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            scaler.load_state_dict(state["scaler_state_dict"])
            step, epoch, next_batch, best = [state[key] for key in ("trained_steps", "epoch", "next_batch", "best_focus_f1")]
        if world > 1:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        raw_model = model.module if isinstance(model, DDP) else model
        if state is not None:
            rng = state["rng_states"][rank]
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(rng["cuda"], device)
            del state
        loader_options = dict(num_workers=config.NUM_WORKERS, collate_fn=GenreCollator(model_config.pad_id), pin_memory=device.type == "cuda")
        loader = DataLoader(train_data, batch_sampler=batches, generator=torch.Generator().manual_seed(config.SEED), **loader_options)
        val_loader = DataLoader(val_data, batch_size=config.VALIDATION_BATCH_SIZE, sampler=range(rank, len(val_data), world),
                                generator=torch.Generator().manual_seed(config.SEED + 1), **loader_options)
        if rank == 0:
            archived = prepare_run_directory(directory, resuming=resume_path.exists())
            if archived is not None:
                print(f"No latest checkpoint; starting fresh. Previous debug outputs preserved in {archived}", flush=True)
            log = (directory / "metrics.jsonl").open("a" if resume_path.exists() else "x")
            if config.TENSORBOARD:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(directory / "tensorboard")
            print(f"Genre encoder: {model_config.embed_dim} wide / {model_config.n_layers} layers; effective batch={config.BATCH_SIZE * config.ACCUM_STEPS * world}; resumed step={step}", flush=True)
            print(f"Class pools: {dict(train_data.class_counts)}; sampling probabilities={dict(zip(labels, sampler.class_probabilities.tolist()))}", flush=True)

        def save(name, metrics):
            rng = {"python": random.getstate(), "torch": torch.get_rng_state(),
                   "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None}
            rng_states = [rng]
            if world > 1:
                rng_states = [None] * world
                dist.all_gather_object(rng_states, rng)
            if rank != 0:
                return
            payload = {"format": "genre_encoder_v2", "model_state_dict": _cpu_tree(raw_model.state_dict()),
                       "optimizer_state_dict": _cpu_tree(optimizer.state_dict()), "scheduler_state_dict": scheduler.state_dict(),
                       "scaler_state_dict": scaler.state_dict(), "config": asdict(model_config), "labels": labels,
                       "vocabulary_digest": train_data.vocab_digest, "tokenizer_config": train_data.tokenizer.config,
                       "data_fingerprints": [train_data.data_fingerprint, val_data.data_fingerprint],
                       "trained_steps": step, "epoch": epoch, "next_batch": next_batch, "best_focus_f1": best,
                       "world_size": world, "rng_states": rng_states, "train_config": _config_dict(config),
                       "temperature": 1., "prior_correction": prior_correction.cpu().tolist(),
                       "sampling_probabilities": sampler.class_probabilities.tolist(), "reference_priors": reference.tolist(),
                       "class_pool_counts": dict(train_data.class_counts), "manifest": str(Path(config.MANIFEST).resolve()), "validation": metrics}
            temporary = directory / (name + f".tmp.{os.getpid()}")
            torch.save(payload, temporary)
            temporary.replace(directory / name)

        accumulated, stats = 0, torch.zeros(3, device=device, dtype=torch.float64)
        train_confusion = torch.zeros(len(labels), len(labels), device=device, dtype=torch.long)
        validation = None
        model.train()
        optimizer.zero_grad(set_to_none=True)
        while step < config.N_STEPS:
            sampler.set_epoch(epoch)
            train_data.set_epoch(epoch)
            batches.start_batch = next_batch
            if not 0 <= next_batch < full_epoch_batches:
                raise ValueError("Invalid resume data position")
            for batch in loader:
                sync = accumulated + 1 == config.ACCUM_STEPS
                context = model.no_sync() if isinstance(model, DDP) and not sync else nullcontext()
                with context:
                    ids, mask, targets = [batch[key].to(device, non_blocking=True) for key in ("ids", "attention_mask", "labels")]
                    with amp_context(device):
                        logits = model(ids, mask)
                        loss = F.cross_entropy(logits.float(), targets, weight=weights)
                    finite = torch.isfinite(loss).to(torch.int32)
                    if world > 1:
                        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise FloatingPointError("Non-finite classifier loss; checkpoint not overwritten")
                    scaler.scale(loss / config.ACCUM_STEPS).backward()
                with torch.no_grad():
                    stats += torch.stack((loss.detach().double() * len(targets), (logits.argmax(-1) == targets).sum().double(), loss.new_tensor(len(targets), dtype=torch.float64)))
                    train_confusion += torch.bincount(targets * len(labels) + logits.argmax(-1), minlength=len(labels)**2).view(len(labels), len(labels))
                del batch, logits, loss, ids, mask, targets
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
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        accumulated = 0
                        stats.zero_()
                        train_confusion.zero_()
                        continue
                    raise FloatingPointError("Non-finite classifier gradients; checkpoint not overwritten")
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                step += 1
                if step % config.LOG_EVERY == 0 or step == config.N_STEPS:
                    if world > 1:
                        dist.all_reduce(stats)
                        dist.all_reduce(train_confusion)
                    if rank == 0:
                        scores = classification_metrics(train_confusion.cpu(), labels, focus_genres=config.FOCUS_GENRES)
                        values = {"loss": float(stats[0] / stats[2]), "accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"], "lr": scheduler.get_last_lr()[0]}
                        print(f"step {step}: train_loss={values['loss']:.4f}; {format_classification_metrics(scores)}", flush=True)
                        if writer:
                            for key, value in values.items():
                                writer.add_scalar("train/" + key, value, step)
                            log_classification_metrics(writer, "train", scores, step)
                    stats.zero_()
                    train_confusion.zero_()
                improved = False
                if step % config.VALIDATE_EVERY == 0 or step == config.N_STEPS:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    validation = evaluate(model, val_loader, labels, device, prior_correction=prior_correction, max_batches=config.MAX_VALIDATION_BATCHES, focus_genres=config.FOCUS_GENRES)
                    focus = [validation["per_genre"][label]["f1"] for label in config.FOCUS_GENRES if label in labels and validation["per_genre"][label]["support"]]
                    validation["focus_f1"] = sum(focus) / len(focus) if focus else validation["macro_f1"]
                    improved = validation["focus_f1"] > best
                    best = max(best, validation["focus_f1"])
                    if rank == 0:
                        log.write(json.dumps({"step": step, "validation": validation}) + "\n")
                        log.flush()
                        print(f"validation {step}: nll={validation['nll']:.4f}, focus_f1={validation['focus_f1']:.3f}; {format_classification_metrics(validation)}", flush=True)
                        if writer:
                            for key in ("accuracy", "macro_f1", "focus_f1", "nll"):
                                writer.add_scalar("validation/" + key, validation[key], step)
                            log_classification_metrics(writer, "validation", validation, step)
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
        if config.CALIBRATE:
            # Calibration must accompany the selected best model, not the
            # latest optimizer state. Never overwrite a resumable checkpoint.
            if world > 1:
                dist.barrier()
            state = torch.load(directory / "best_checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
            raw_model.load_state_dict(state["model_state_dict"])
            _, logits, targets = evaluate(raw_model, val_loader, labels, device, prior_correction=prior_correction,
                                         max_batches=config.MAX_VALIDATION_BATCHES, collect_logits=True)
            if rank == 0:
                calibration = fit_temperature(logits, targets)
                state.update(temperature=calibration["temperature"], calibration=calibration)
                temporary = directory / "calibrated_checkpoint.pt.tmp"
                torch.save(state, temporary)
                temporary.replace(directory / "calibrated_checkpoint.pt")
                print("Calibration: " + json.dumps(calibration), flush=True)
    finally:
        if writer:
            writer.close()
        if log:
            log.close()
        for dataset in (train_data, val_data):
            if dataset:
                dataset.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="improvnet.metrics.genre_classifier.genre_classifier_config")
    for flag in ("manifest", "save-dir", "device"):
        parser.add_argument("--" + flag)
    for flag in ("batch-size", "accum-steps", "steps", "workers", "max-validation-batches"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    module = importlib.import_module(args.config)
    config = SimpleNamespace(**{key: value for key, value in vars(module).items() if key.isupper()})
    for argument, key in (("manifest", "MANIFEST"), ("save_dir", "SAVE_DIR"), ("device", "DEVICE"),
                          ("batch_size", "BATCH_SIZE"), ("accum_steps", "ACCUM_STEPS"), ("steps", "N_STEPS"),
                          ("workers", "NUM_WORKERS"), ("max_validation_batches", "MAX_VALIDATION_BATCHES")):
        if getattr(args, argument) is not None:
            setattr(config, key, getattr(args, argument))
    for argument, key in (("no_resume", "RESUME_TRAINING"), ("no_calibrate", "CALIBRATE"), ("no_tensorboard", "TENSORBOARD")):
        if getattr(args, argument):
            setattr(config, key, False)
    train(config)


if __name__ == "__main__":
    main()
