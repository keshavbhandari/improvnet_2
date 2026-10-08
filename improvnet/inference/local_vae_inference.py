"""Reconstruct a MIDI's selected instrument with a trained local VAE."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from improvnet.model import local_vae_config as defaults
from improvnet.model.local_vae_model import LocalVAE, LocalVAEConfig
from improvnet.tokenizer.hierarchical import GridConfig, HierarchicalTokenizer, roll_notes


def load_model(path, device):
    # Only load trusted, locally trained checkpoints (they include Python RNG state).
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "local_vae_training_v1":
        raise ValueError("Expected a local VAE training checkpoint")
    model = LocalVAE(LocalVAEConfig(**state["model_config"]))
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.decoder_calibration = state.get("decoder_calibration")
    config = state["train_config"]
    grid = GridConfig(coarse_ms=config["COARSE_MS"], fine_ms=config["FINE_MS"],
                      piano_programs=tuple(config["PIANO_PROGRAMS"]), max_frames=config["MAX_FRAMES"])
    if grid.frames_per_bin != model.config.frames_per_bin:
        raise ValueError("Checkpoint model and tokenizer frame counts disagree")
    return model.to(device).eval(), HierarchicalTokenizer(grid), state["step"], config


def resolve_thresholds(args, model):
    """CLI overrides > checkpoint calibration > legacy .5/.99 defaults."""
    calibration = getattr(model, "decoder_calibration", None) or {}
    for name, fallback in (("activity_threshold", .5), ("onset_threshold", .99)):
        value = getattr(args, name)
        if value is None:
            value = calibration.get(name, fallback)
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError("Thresholds must be finite and strictly between zero and one")
        setattr(args, name, float(value))


def safe_ratio(a, b):
    return float(a / b) if b else None


class Scores:
    """Micro counts over valid cells, never over padded frames or accompaniment."""
    def __init__(self):
        self.counts = {k: 0. for k in ("activity_tp", "activity_fp", "activity_fn", "onset_tp", "onset_fp",
            "onset_fn", "velocity_error", "active", "pedal_error", "pedal_cells", "notes_tp", "notes_fp",
            "notes_fn", "onset_ms", "offset_ms", "boundary_tp", "boundary_fp", "boundary_fn")}

    def add(self, other):
        for key in self.counts:
            self.counts[key] += other.counts[key]

    def arrays(self, reference, prediction, indices=None):
        ids = slice(None) if indices is None else indices
        mask = reference.valid_mask[ids, None, :]
        a, b = (reference.velocity[ids] > 0) & mask, (prediction.velocity[ids] > 0) & mask
        for name, x, y in (("activity", a, b), ("onset", reference.onset[ids].astype(bool) & mask,
                                                 prediction.onset[ids].astype(bool) & mask)):
            self.counts[name + "_tp"] += int((x & y).sum())
            self.counts[name + "_fp"] += int((~x & y).sum())
            self.counts[name + "_fn"] += int((x & ~y).sum())
        boundary = mask.copy()
        # First 50 ms after every patch boundary, where carry errors can appear.
        boundary[:, :, max(1, round(50 / reference.config.fine_ms)):] = False
        self.counts["boundary_tp"] += int((a & b & boundary).sum())
        self.counts["boundary_fp"] += int((~a & b & boundary).sum())
        self.counts["boundary_fn"] += int((a & ~b & boundary).sum())
        self.counts["velocity_error"] += float(np.abs(reference.velocity[ids].astype(float) - prediction.velocity[ids])[a].sum())
        self.counts["active"] += int(a.sum())
        if reference.metadata["pedal_enabled"]:
            self.counts["pedal_error"] += float(np.abs(reference.pedal[ids].astype(float) - prediction.pedal[ids])[reference.valid_mask[ids]].sum())
            self.counts["pedal_cells"] += int(reference.valid_mask[ids].sum())

    def notes(self, reference, prediction, tolerance_ms=50, indices=None):
        # Same-pitch one-to-one, chronological onset matching; no velocity constraint.
        def notes(piece):
            ids = slice(None) if indices is None else indices
            return roll_notes(piece.velocity[ids], piece.onset[ids], int(piece.valid_mask[ids].sum()), 0, 0, 0, strict=False)
        left, right = notes(reference), notes(prediction)
        matched = 0
        for pitch in range(128):
            a = sorted((n[5], n[6]) for n in left if n[3] == pitch)
            b = sorted((n[5], n[6]) for n in right if n[3] == pitch)
            i = j = 0
            frame_ms = reference.config.fine_ms
            while i < len(a) and j < len(b):
                delta = (b[j][0] - a[i][0]) * frame_ms
                if abs(delta) <= tolerance_ms:
                    matched += 1
                    self.counts["onset_ms"] += abs(delta)
                    self.counts["offset_ms"] += abs(b[j][1] - a[i][1]) * frame_ms
                    i += 1; j += 1
                elif delta < 0:
                    j += 1
                else:
                    i += 1
        self.counts["notes_tp"] += matched
        self.counts["notes_fn"] += len(left) - matched
        self.counts["notes_fp"] += len(right) - matched

    def report(self):
        c = self.counts
        result = {}
        for name in ("activity", "onset", "notes", "boundary"):
            tp, fp, fn = (c[name + "_" + k] for k in ("tp", "fp", "fn"))
            result[name + "_f1"] = safe_ratio(2 * tp, 2 * tp + fp + fn)
        result.update(missed_notes=int(c["notes_fn"]), extra_notes=int(c["notes_fp"]),
                      onset_mae_ms=safe_ratio(c["onset_ms"], c["notes_tp"]),
                      offset_mae_ms=safe_ratio(c["offset_ms"], c["notes_tp"]),
                      velocity_mae=safe_ratio(c["velocity_error"], c["active"]),
                      pedal_mae=safe_ratio(c["pedal_error"], c["pedal_cells"]))
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}


class LatentStats:
    def __init__(self):
        self.weight = 0.
        self.total = self.squares = self.noise = None
        self.kl_sum = 0.
        self.bins = 0

    def update(self, posterior):
        mean, var = posterior.mean.double(), posterior.logvar.double().exp()
        w = posterior.weights.double().expand_as(mean)
        n = float(w[:, 0].sum())
        sums = [(x * w).sum((0, 2, 3)).cpu().numpy() for x in (mean, mean.square(), var)]
        if self.total is None:
            self.total, self.squares, self.noise = [np.zeros_like(x) for x in sums]
        self.total += sums[0]; self.squares += sums[1]; self.noise += sums[2]
        self.weight += n
        self.kl_sum += float(posterior.kl().sum())
        self.bins += len(mean)

    def report(self):
        mean = self.total / self.weight
        var = np.maximum(0, self.squares / self.weight - mean**2)
        return dict(mean=np.round(mean, 4).tolist(), std=np.round(np.sqrt(var), 4).tolist(),
                    posterior_std=np.round(np.sqrt(self.noise / self.weight), 4).tolist(),
                    kl_per_element=round(self.kl_sum / self.bins, 4),
                    active_channel_fraction=round(float((var > .01).mean()), 4))


def inputs(piece, indices, device):
    # Avoid constructing/collating unused accompaniment tensors for inference.
    return (torch.from_numpy(np.stack((piece.velocity[indices] / 127., piece.onset[indices]), axis=1)).float().to(device),
            torch.from_numpy(piece.pedal[indices, None] / 127.).float().to(device),
            torch.from_numpy(piece.valid_mask[indices]).to(device),
            torch.full((len(indices),), bool(piece.metadata["pedal_enabled"]), device=device))


@torch.inference_mode()
def reconstruct(model, piece, *, batch_size=16, activity_threshold=.5, onset_threshold=.5,
                sample=False, generator=None, indices=None, diagnostics=False):
    if batch_size < 1 or not 0 < activity_threshold < 1 or not 0 < onset_threshold < 1:
        raise ValueError("Positive batch size and thresholds strictly between zero and one required")
    piece.validate()
    indices = np.arange(len(piece.velocity)) if indices is None else np.asarray(indices)
    if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
            or indices.min() < 0 or indices.max() >= len(piece.velocity) or len(np.unique(indices)) != len(indices)):
        raise ValueError("Select unique valid integer bin indices")
    output = deepcopy(piece)
    # Diagnostics compare the same bins, including genuine silence.
    sampled, zero = deepcopy(piece) if diagnostics else None, deepcopy(piece) if diagnostics else None
    stats = LatentStats()
    device = next(model.parameters()).device
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        roll, pedal, mask, enabled = inputs(piece, ids, device)
        posterior = model.encode(roll, pedal, mask)
        if not bool(torch.isfinite(posterior.mean).all() & torch.isfinite(posterior.logvar).all()):
            raise FloatingPointError("Non-finite posterior")
        stats.update(posterior)
        z = posterior.sample(generator) if sample else posterior.mode()
        variants = [(output, model.decode(z))]
        if diagnostics:
            variants += [(sampled, model.decode(posterior.sample(generator))),
                         (zero, model.decode(torch.zeros_like(z)))]
        for target, reconstruction in variants:
            if not all(bool(torch.isfinite(t).all()) for t in (reconstruction.activity_logits,
                reconstruction.onset_logits, reconstruction.velocity, reconstruction.pedal)):
                raise FloatingPointError("Non-finite reconstruction")
            arrays = reconstruction.tensors(mask, enabled, activity_threshold=activity_threshold,
                                             onset_threshold=onset_threshold)
            for key, tensor in arrays.items():
                getattr(target, key)[ids] = tensor.cpu().numpy()
    output.refresh_coarse().validate()
    return output, stats, sampled, zero


def output_directory(value, prefix):
    if value is None:
        root = Path(defaults.SAVE_DIR).parent / "local_vae_evaluation"
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    path = Path(value)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Use a new/empty output directory; existing artifacts are never overwritten")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_report(path, report):
    with Path(path).open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


def common_args(parser):
    parser.add_argument("--checkpoint", type=Path, default=Path(defaults.SAVE_DIR) / "best_model.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activity-threshold", type=float, help="Override saved calibration (legacy fallback .5)")
    parser.add_argument("--onset-threshold", type=float, help="Override saved calibration (legacy fallback .99)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common_args(parser)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--target-channel", type=int)
    parser.add_argument("--target-program", type=int)
    parser.add_argument("--sample", action="store_true", help="Sample posterior instead of deterministic mean")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    model, tokenizer, step, _ = load_model(args.checkpoint, device)
    resolve_thresholds(args, model)
    piece = tokenizer.encode(args.input, recover=True, seed=args.seed,
                             target_channel=args.target_channel, target_program=args.target_program)
    if not piece.metadata["target_present"]:
        raise ValueError("No valid target notes")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    predicted, _, _, _ = reconstruct(model, piece, batch_size=args.batch_size,
        activity_threshold=args.activity_threshold, onset_threshold=args.onset_threshold,
        sample=args.sample, generator=generator)
    scores = Scores(); scores.arrays(piece, predicted); scores.notes(piece, predicted)
    output = output_directory(args.output_dir, "single-")
    tokenizer.decode(piece, target_only=True).save(output / "reference_target.mid")
    tokenizer.decode(predicted, target_only=True).save(output / "reconstructed_target.mid")
    tokenizer.decode(predicted).save(output / "reconstructed.mid")
    report = dict(input=str(args.input.resolve()), checkpoint=str(args.checkpoint.resolve()), step=step, bins=len(piece.velocity),
                  target=dict(channel=piece.metadata["target_channel"], program=piece.metadata["target_program"]),
                  posterior="sample" if args.sample else "mean", seed=args.seed,
                  thresholds=dict(activity=args.activity_threshold, onset=args.onset_threshold), errors=scores.report())
    write_report(output / "report.json", report)
    print(json.dumps(report["errors"]))
    print(f"Saved: {output}")
    return report


if __name__ == "__main__":
    main()
