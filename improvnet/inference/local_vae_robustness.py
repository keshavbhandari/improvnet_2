"""Frozen-VAE latent noise sweep; training-only scale, held-out scores."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import mido
import numpy as np
import torch

from improvnet.inference.local_vae_evaluate import candidates
from improvnet.inference.local_vae_inference import (
    LatentStats, Scores, common_args, inputs, load_model, note_decoding_options, output_directory, resolve_thresholds, write_report,
)
from improvnet.model.local_vae_model import Posterior
from improvnet.utils.note_decoding import NoteDecoding, apply_note_decoding, confidence_arrays
from improvnet.utils.local_vae_data import LocalVAEDataset, seed_for


def channel_scale(stats):
    """Use unrounded, valid-weighted training posterior-mean standard deviation."""
    if stats.weight <= 0:
        raise ValueError("No training latent observations")
    variance = stats.squares / stats.weight - (stats.total / stats.weight)**2
    if not np.isfinite(variance).all():
        raise FloatingPointError("Non-finite training latent scale")
    return torch.tensor(np.sqrt(np.maximum(variance, 0)), dtype=torch.float32)


def perturb(mean, scale, epsilon, alpha, weights):
    """Noise std = alpha * training channel std; do not disturb padded columns."""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Noise alpha must be finite and nonnegative")
    if scale.shape != (mean.shape[1],) or epsilon.shape != mean.shape:
        raise ValueError("Noise/scale shape mismatch")
    if not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
        raise ValueError("Invalid channel scale")
    return (mean + alpha * scale[None, :, None, None] * epsilon) * (weights > 0)


def merge_stats(total, update):
    if total.total is None:
        return update
    total.total += update.total; total.squares += update.squares; total.noise += update.noise
    total.weight += update.weight; total.bins += update.bins; total.kl_sum += update.kl_sum
    return total


@torch.inference_mode()
def collect(model, tokenizer, config, split, count, bins_per_file, seed, batch_size, manifest):
    dataset = LocalVAEDataset(config, split)
    windows, stats, skipped = [], LatentStats(), Counter()
    device = next(model.parameters()).device
    try:
        for attempt, (row_id, collection, path) in enumerate(candidates(dataset, seed)):
            if len(windows) >= count or attempt >= count * 5:
                break
            try:
                piece = tokenizer.encode(path, recover=True, seed=seed_for(config.SEED, 0, row_id))
                if not piece.metadata["target_present"]:
                    raise ValueError("No valid target notes")
            except (ValueError, OSError, EOFError, IndexError, KeyError, mido.KeySignatureError) as error:
                skipped[str(error)[:100]] += 1
                continue
            length = min(bins_per_file, len(piece.velocity))
            start = random.Random(seed_for(seed, split, row_id, "window")).randrange(len(piece.velocity) - length + 1)
            ids = np.arange(start, start + length)
            # Only cache compact windows, not complete pieces/accompaniment.
            reference = SimpleNamespace(config=piece.config, metadata={"pedal_enabled": piece.metadata["pedal_enabled"]},
                velocity=piece.velocity[ids].copy(), onset=piece.onset[ids].copy(),
                pedal=piece.pedal[ids].copy(), valid_mask=piece.valid_mask[ids].copy())
            parts = []
            for offset in range(0, length, batch_size):
                roll, pedal, mask, _ = inputs(reference, np.arange(offset, min(offset + batch_size, length)), device)
                posterior = model.encode(roll, pedal, mask)
                if not all(bool(torch.isfinite(x).all()) for x in (posterior.mean, posterior.logvar)):
                    raise FloatingPointError("Non-finite posterior")
                update = LatentStats(); update.update(posterior); stats = merge_stats(stats, update)
                parts.append(Posterior(*(x.cpu() for x in (posterior.mean, posterior.logvar, posterior.weights))))
            posterior = Posterior(*(torch.cat([getattr(p, k) for p in parts]) for k in ("mean", "logvar", "weights")))
            windows.append(dict(reference=reference, posterior=posterior, collection=collection,
                                kind=piece.metadata["target_kind"], row_id=row_id))
            manifest.writerow([split, row_id, collection, start, length])
            if len(windows) % 20 == 0 or len(windows) == count:
                print(f"Encoded {split}: {len(windows)}/{count} files", flush=True)
    finally:
        if dataset.connection is not None:
            dataset.connection.close()
    if not windows:
        raise RuntimeError(f"No usable {split} files; skips={dict(skipped)}")
    return windows, stats, skipped


def raw_metrics(score):
    """Unrounded metrics; note count is normalized by fixed reference notes."""
    c = score.counts
    result = {}
    for name in ("activity", "onset", "notes"):
        tp, fp, fn = (c[name + "_" + k] for k in ("tp", "fp", "fn"))
        result[name + "_precision"] = tp / (tp + fp) if tp + fp else None
        result[name + "_recall"] = tp / (tp + fn) if tp + fn else None
        result[name + "_f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    for name, num, den in (("onset_precision", "onset_tp", None), ("onset_recall", "onset_tp", None),
                           ("velocity_mae", "velocity_error", "active"), ("pedal_mae", "pedal_error", "pedal_cells"),
                           ("offset_mae_ms", "offset_ms", "notes_tp")):
        divisor = c[den] if den else c["onset_tp"] + c["onset_fp" if name == "onset_precision" else "onset_fn"]
        result[name] = c[num] / divisor if divisor else None
    n = c["notes_tp"] + c["notes_fn"]
    result["extra_per_100_notes"] = 100 * c["notes_fp"] / n if n else None
    result["missed_per_100_notes"] = 100 * c["notes_fn"] / n if n else None
    return result


def average_scores(scores):
    values = [raw_metrics(s) for s in scores]
    result = {}
    for key in values[0]:
        valid = [v[key] for v in values if v[key] is not None]
        result[key] = round(float(np.mean(valid)), 5) if valid else None
    f1 = [v["notes_f1"] for v in values if v["notes_f1"] is not None]
    result["notes_f1_draw_sd"] = round(float(np.std(f1)), 5) if f1 else None
    return result


@torch.inference_mode()
def sweep(model, windows, scale, alphas, draws, seed, batch_size, activity_threshold, onset_threshold,
          note_decoding=None):
    device = next(model.parameters()).device
    options = note_decoding or getattr(model, "note_decoding", NoteDecoding())
    options = options if isinstance(options, NoteDecoding) else NoteDecoding(**options)
    modes = [*alphas, "posterior"]
    totals = {mode: [Scores() for _ in range(draws)] for mode in modes}
    groups = {kind: {mode: [Scores() for _ in range(draws)] for mode in modes} for kind in {w["kind"] for w in windows}}
    scale = scale.to(device)
    for index, window in enumerate(windows):
        reference, p = window["reference"], window["posterior"]
        mean, logvar, weights = (t.to(device) for t in (p.mean, p.logvar, p.weights))
        mask = torch.from_numpy(reference.valid_mask).to(device)
        enabled = torch.full((len(mean),), reference.metadata["pedal_enabled"], device=device)
        for draw in range(draws):
            # Same random direction across strengths AND posterior control,
            # independently seeded per file/draw; independent of batch size.
            gen = torch.Generator().manual_seed(seed_for(seed, "latent-noise", window["row_id"], draw))
            epsilon = torch.randn(p.mean.shape, generator=gen).to(device)
            for mode in modes:
                z = ((mean + (.5 * logvar).exp() * epsilon) * (weights > 0) if mode == "posterior"
                     else perturb(mean, scale, epsilon, mode, weights))
                arrays = {k: [] for k in ("velocity", "onset", "pedal")}
                confidence = {k: [] for k in ("activity", "onset", "velocity")}
                for offset in range(0, len(mean), batch_size):
                    sl = slice(offset, offset + batch_size)
                    r = model.decode(z[sl])
                    if not all(bool(torch.isfinite(t).all()) for t in (r.velocity, r.activity_logits, r.onset_logits, r.pedal)):
                        raise FloatingPointError("Non-finite decoder output")
                    for key, tensor in r.tensors(mask[sl], enabled[sl], activity_threshold=activity_threshold,
                                                 onset_threshold=onset_threshold).items():
                        arrays[key].append(tensor.cpu().numpy())
                    if options.mode == "held":
                        for key, value in confidence_arrays(r).items():
                            confidence[key].append(value)
                prediction = SimpleNamespace(**{k: np.concatenate(v) for k, v in arrays.items()},
                                             config=reference.config, valid_mask=reference.valid_mask)
                if options.mode == "held":
                    apply_note_decoding(prediction, np.arange(len(mean)),
                        {k: np.concatenate(v) for k, v in confidence.items()}, options,
                        activity_threshold=activity_threshold, onset_threshold=onset_threshold)
                score = Scores(); score.arrays(reference, prediction); score.notes(reference, prediction)
                totals[mode][draw].add(score); groups[window["kind"]][mode][draw].add(score)
        if (index + 1) % 20 == 0 or index + 1 == len(windows):
            print(f"Noise sweep: {index + 1}/{len(windows)} held-out files", flush=True)
    return totals, groups


def plot_report(report, output):
    try:
        import matplotlib
    except ModuleNotFoundError as error:
        raise RuntimeError("Metrics are saved. Plotting requires matplotlib; install it and call plot_report on report.json.") from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = report["curve"]
    x = np.array([r["alpha"] * 100 for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = {"activity_f1": "#2563eb", "onset_f1": "#db2777", "notes_f1": "#059669"}
    for key, color in colors.items():
        y = np.array([r[key] if r[key] is not None else np.nan for r in rows])
        axes[0, 0].plot(x, 100 * y, "o-", label=key.replace("_f1", "").title(), color=color)
        axes[0, 1].plot(x, 100 * (y[0] - y), "o-", label=key.replace("_f1", "").title(), color=color)
    sd = np.array([r["notes_f1_draw_sd"] or 0 for r in rows]) * 100
    y = np.array([r["notes_f1"] if r["notes_f1"] is not None else np.nan for r in rows]) * 100
    axes[0, 0].fill_between(x, y - sd, y + sd, color=colors["notes_f1"], alpha=.12, label="Note F1 ± noise-draw SD")
    control = report["posterior_control"]["notes_f1"]
    if control is not None:
        axes[0, 0].axhline(control * 100, linestyle="--", color="#6b7280", label="Posterior-sample note F1")
    axes[0, 0].set(title="Reconstruction fidelity", ylabel="F1 (%)", ylim=(0, 101))
    axes[0, 1].set(title="Degradation from clean latent", ylabel="F1 loss (percentage points)")
    axes[1, 0].plot(x, [r["velocity_mae"] for r in rows], "o-", color="#7c3aed", label="Velocity")
    if rows[0]["pedal_mae"] is not None:
        axes[1, 0].plot(x, [r["pedal_mae"] for r in rows], "o-", label="Pedal", color="#d97706")
    axes[1, 0].set(title="Velocity / pedal error", ylabel="MAE (0–127 MIDI units)")
    for key, label in (("extra_per_100_notes", "Extra"), ("missed_per_100_notes", "Missed")):
        axes[1, 1].plot(x, [r[key] for r in rows], "o-", label=label)
    axes[1, 1].set(title="Note errors", ylabel="Errors per 100 reference notes")
    for ax in axes.flat:
        ax.set_xlabel("Noise standard deviation (% of training latent channel std)")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    files = report.get("evaluation_files", report.get("validation_files"))
    split = report.get("evaluation_split", "validation")
    fig.suptitle(f"Frozen local VAE · {files} {split} files · onset threshold {report['thresholds']['onset']}\n"
                 "Mean-latent perturbation; shading is noise-draw variability, not a confidence interval", fontsize=12)
    fig.savefig(output / "robustness.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for kind, values in report["by_target"].items():
        ax.plot(x, [100 * r["notes_f1"] if r["notes_f1"] is not None else np.nan for r in values],
                "o-", label=f"{kind} ({report['target_files'][kind]} files)")
    ax.set(xlabel="Noise std (% of training latent channel std)", ylabel="Note F1 (%)",
           title="Robustness by target instrument type", ylim=(0, 101))
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(output / "by_instrument.png", dpi=160); plt.close(fig)


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common_args(parser)
    parser.set_defaults(batch_size=8)
    parser.add_argument("--num-files", type=int, default=200)
    parser.add_argument("--calibration-files", type=int, default=64)
    parser.add_argument("--bins-per-file", type=int, default=8)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--split", choices=("validation", "test"), default="validation",
                        help="Use test after selecting decoding thresholds on validation")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0., .005, .01, .02, .05, .1, .2])
    args = parser.parse_args(argv)
    if min(args.num_files, args.calibration_files, args.bins_per_file, args.draws, args.batch_size) < 1:
        parser.error("All counts must be positive")
    if not all(np.isfinite(a) and a >= 0 for a in args.alphas):
        parser.error("Alphas must be finite and nonnegative")
    alphas = sorted(set([0., *args.alphas]))
    if len(alphas) < 2:
        parser.error("Include at least one nonzero noise level")
    device = torch.device(args.device)
    # Guard the exact checkpoint used; never overwrite or fine-tune it.
    digest = checkpoint_digest(args.checkpoint)
    model, tokenizer, step, saved = load_model(args.checkpoint, device)
    resolve_thresholds(args, model)
    if checkpoint_digest(args.checkpoint) != digest:
        raise RuntimeError("Checkpoint changed while loading; rerun with a stable checkpoint")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = SimpleNamespace(**saved)
    output = output_directory(args.output_dir, "latent-robustness-")
    with (output / "selection.csv").open("x", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["split", "row_id", "collection", "start_bin", "bins"])
        calibration, train_stats, train_skips = collect(model, tokenizer, config, "train", args.calibration_files,
            args.bins_per_file, args.seed, args.batch_size, writer)
        scale = channel_scale(train_stats)
        calibration_count = len(calibration); calibration_bins = train_stats.bins
        del calibration
        windows, _, validation_skips = collect(model, tokenizer, config, args.split, args.num_files,
            args.bins_per_file, args.seed, args.batch_size, writer)
    totals, groups = sweep(model, windows, scale, alphas, args.draws, args.seed, args.batch_size,
                          args.activity_threshold, args.onset_threshold, note_decoding_options(args))
    collections = Counter(w["collection"] for w in windows)
    kinds = Counter(w["kind"] for w in windows)
    curve = [dict(alpha=alpha, **average_scores(totals[alpha])) for alpha in alphas]
    report = dict(checkpoint_step=step, checkpoint_sha256=digest, seed=args.seed,
                  evaluation_split=args.split, evaluation_files=len(windows),
                  evaluation_bins=sum(len(w["reference"].velocity) for w in windows),
                  calibration_files=calibration_count, calibration_bins=calibration_bins, noise_draws=args.draws,
                  thresholds=dict(activity=args.activity_threshold, onset=args.onset_threshold),
                  note_decoding=vars(note_decoding_options(args)),
                  training_channel_std=scale.tolist(), training_posterior_std=train_stats.report()["posterior_std"],
                  evaluation_collections=dict(collections), target_files=dict(kinds),
                  curve=curve, posterior_control=average_scores(totals["posterior"]),
                  by_target={kind: [dict(alpha=a, notes_f1=average_scores(values[a])["notes_f1"],
                                        onset_f1=average_scores(values[a])["onset_f1"]) for a in alphas]
                             for kind, values in groups.items()},
                  skipped={"train": sum(train_skips.values()), args.split: sum(validation_skips.values())})
    if args.split == "validation":
        # Retain the original report fields for existing validation analyses.
        report.update(validation_files=report["evaluation_files"], validation_bins=report["evaluation_bins"],
                      validation_collections=report["evaluation_collections"])
    write_report(output / "report.json", report)
    print("alpha | activity F1 | onset F1 | note F1 | velocity MAE", flush=True)
    for row in curve:
        print(f"{row['alpha']:5.3f} | {row['activity_f1']} | {row['onset_f1']} | {row['notes_f1']} | {row['velocity_mae']}", flush=True)
    print(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB" if device.type == "cuda" else "CPU evaluation")
    print(f"Saved: {output}")
    plot_report(report, output)
    return report


if __name__ == "__main__":
    main()
