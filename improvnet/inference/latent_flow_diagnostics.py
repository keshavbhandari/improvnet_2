"""Read-only held-note, flow-step and frozen-decoder diagnostics.

Writes caches/reports/listening MIDIs only beneath --output-dir. No training,
checkpoint edits, test-set threshold selection, or reference-assisted decoding.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sqlite3
import time

import mido
import numpy as np
import torch

from improvnet.inference.latent_flow_inference import load_model, resize_piece
from improvnet.inference.local_vae_evaluate import candidates
from improvnet.inference.local_vae_inference import Scores, inputs
from improvnet.model import latent_flow_config as defaults
from improvnet.model.local_vae_train import amp_context
from improvnet.tokenizer.hierarchical import roll_notes
from improvnet.utils.latent_flow import checkpoint_digest, encode_windows, sample
from improvnet.utils.latent_flow_data import collate_windows, window_tensors
from improvnet.utils.local_vae_data import LocalVAEDataset, seed_for
from improvnet.utils.note_decoding import NoteDecoding, apply_note_decoding, confidence_arrays


def write_report(path, data):
    """Atomically replace only this evaluation's own reports, allowing restart."""
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(pending, path)


def crop_piece(piece, start, bins):
    """Keep exact target frames; reset the output MIDI clock to clip time zero."""
    result = resize_piece(piece, bins)
    for key in ("velocity", "onset", "pedal", "valid_mask"):
        setattr(result, key, getattr(piece, key)[start:start + bins].copy())
    result.events = np.zeros((0, 11), dtype=np.int64)
    result.event_ptr = np.zeros(bins + 1, dtype=np.int64)
    result.metadata = dict(piece.metadata, ppq=1000, tempo_map=[(0, 500000)], tracks=1, target_track=0)
    result.metadata["messages"] = [
        dict(track=0, tick=0, order=0, is_meta=True, message=mido.MetaMessage("set_tempo", tempo=500000).dict()),
        dict(track=0, tick=0, order=1, is_meta=False,
             message=mido.Message("program_change", channel=piece.metadata["target_channel"],
                                  program=piece.metadata["target_program"]).dict()),
    ]
    return result.refresh_coarse().validate()


def collect_test(tokenizer, config, count, bins, seed):
    dataset = LocalVAEDataset(config, "test")
    windows, skipped = [], Counter()
    try:
        for attempt, (row_id, collection, path) in enumerate(candidates(dataset, seed)):
            if len(windows) >= count or attempt >= count * 20:
                break
            try:
                piece = tokenizer.encode(path, recover=True, seed=seed_for(config.SEED, 0, row_id))
                if not piece.metadata["target_present"] or piece.metadata["target_kind"] != "piano":
                    skipped["non-piano/empty target"] += 1
                    continue
                length = min(bins, len(piece.velocity))
                start = random.Random(seed_for(seed, "diagnostic-window", row_id)).randrange(len(piece.velocity) - length + 1)
                clipped = crop_piece(piece, start, length)
            except (ValueError, OSError, EOFError, IndexError, KeyError, mido.KeySignatureError) as error:
                skipped[str(error)[:100]] += 1
                continue
            windows.append(dict(name=f"test-{row_id}", row_id=row_id, collection=collection,
                                path=str(path), start_bin=start, bins=length, piece=clipped))
        if len(windows) < count:
            raise RuntimeError(f"Only {len(windows)}/{count} usable test piano windows; {dict(skipped)}")
        return windows, dict(skipped)
    finally:
        if dataset.connection is not None:
            dataset.connection.close()


def decode_prediction(reference, confidence, pedal, activity, onset, options):
    prediction = deepcopy(reference)
    active = (confidence["activity"] >= activity) & reference.valid_mask[:, None]
    prediction.velocity = np.clip(np.rint(confidence["velocity"] * 127), 1, 127).astype(np.uint8) * active
    prediction.onset = ((confidence["onset"] >= onset) & active).astype(np.uint8)
    prediction.pedal = pedal.copy()
    if options.mode == "held":
        apply_note_decoding(prediction, np.arange(len(reference.velocity)), confidence, options,
                            activity_threshold=activity, onset_threshold=onset)
    return prediction.refresh_coarse().validate()


def extra_note_categories(reference, prediction, tolerance_ms=50):
    """Partition unmatched attacks; pitch categories describe source mismatch, not jazz quality."""
    left = roll_notes(reference.velocity, reference.onset, reference.n_frames, 0, 0, strict=False)
    right = roll_notes(prediction.velocity, prediction.onset, prediction.n_frames, 0, 0, strict=False)
    tolerance = tolerance_ms // reference.config.fine_ms
    categories = Counter(activity_only=0, same_pitch_extra_strike=0, off_reference_pitch=0,
                         between_reference_attacks=0, short_extra_50ms=0, near_bin_boundary=0)
    reference_activity = reference.velocity.transpose(1, 0, 2).reshape(128, -1)[:, :reference.n_frames]
    predicted_onsets = prediction.onset.transpose(1, 0, 2).reshape(128, -1)
    all_attacks = np.array([n[5] for n in left])
    for pitch in range(128):
        a = sorted([n for n in left if n[3] == pitch], key=lambda n: n[5])
        b = sorted([n for n in right if n[3] == pitch], key=lambda n: n[5])
        i = j = 0
        unmatched = []
        while i < len(a) and j < len(b):
            delta = b[j][5] - a[i][5]
            if abs(delta) <= tolerance:
                i += 1; j += 1
            elif delta < 0:
                unmatched.append(b[j]); j += 1
            else:
                i += 1
        unmatched.extend(b[j:])
        for note in unmatched:
            start, end = note[5:7]
            lo, hi = max(0, start - tolerance), min(reference.n_frames, start + tolerance + 1)
            if start > 0 and not predicted_onsets[pitch, start]:
                categories["activity_only"] += 1
            elif reference_activity[pitch, lo:hi].any():
                categories["same_pitch_extra_strike"] += 1
            else:
                categories["off_reference_pitch"] += 1
            categories["between_reference_attacks"] += int(not np.any(np.abs(all_attacks - start) <= tolerance))
            categories["short_extra_50ms"] += int((end - start) * reference.config.fine_ms <= 50)
            phase = start % reference.config.frames_per_bin
            categories["near_bin_boundary"] += int(min(phase, reference.config.frames_per_bin - phase) * reference.config.fine_ms <= 50)
    return dict(categories)


def measure(reference, prediction):
    score = Scores(); score.arrays(reference, prediction); score.notes(reference, prediction)
    metrics = score.report()
    n = metrics["reference_notes"]
    metrics.update(extra_per_100=100 * metrics["extra_notes"] / n if n else None,
                   missed_per_100=100 * metrics["missed_notes"] / n if n else None,
                   extra_categories=extra_note_categories(reference, prediction))
    return score, metrics


@torch.inference_mode()
def decode_latents(vae, z, reference, device, batch_size=8):
    confidence = {key: [] for key in ("activity", "onset", "velocity")}
    pedal = []
    for start in range(0, len(z), batch_size):
        with amp_context(device):
            result = vae.decode(z[start:start + batch_size].to(device))
        for key, value in confidence_arrays(result).items():
            confidence[key].append(value)
        pedal.append((result.pedal[:, 0].float().clamp(0, 1) * 127).round().to(torch.uint8).cpu().numpy())
    pedal = np.concatenate(pedal) * reference.valid_mask * reference.metadata["pedal_enabled"]
    return {k: np.concatenate(v) for k, v in confidence.items()}, pedal


@torch.inference_mode()
def encode_piece(vae, piece, device, batch_size=8):
    means, logvars, weights = [], [], []
    for offset in range(0, len(piece.velocity), batch_size):
        ids = np.arange(offset, min(offset + batch_size, len(piece.velocity)))
        roll, pedal, mask, _ = inputs(piece, ids, device)
        with amp_context(device):
            posterior = vae.encode(roll, pedal, mask)
        for output, value in ((means, posterior.mean), (logvars, posterior.logvar), (weights, posterior.weights)):
            output.append(value.float().cpu())
    return tuple(torch.cat(parts) for parts in (means, logvars, weights))


class Aggregates:
    def __init__(self):
        self.scores, self.categories, self.n = {}, {}, Counter()

    def add(self, key, score, metrics):
        self.scores.setdefault(key, Scores()).add(score)
        self.categories.setdefault(key, Counter()).update(metrics["extra_categories"])
        self.n[key] += 1

    def rows(self, names):
        rows = []
        for key, score in self.scores.items():
            result = score.report(); n = result["reference_notes"]
            rows.append(dict(zip(names, key), observations=self.n[key], **result,
                extra_per_100=100 * result["extra_notes"] / n if n else None,
                missed_per_100=100 * result["missed_notes"] / n if n else None,
                extra_categories=dict(self.categories[key])))
        return rows


def paired_noise(shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def conditioning_genre(tags, vocabulary):
    """Match available source labels; never impose a different style in reconstruction."""
    return next((tag for tag in tags if tag in vocabulary[1:]), "none")


@torch.inference_mode()
def run(args):
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.25)
    torch.set_num_threads(4)
    digest = checkpoint_digest(args.checkpoint)
    model, vae, tokenizer, normalizer, config, info = load_model(args.checkpoint, device)
    model.requires_grad_(False)
    activity = vae.decoder_calibration["activity_threshold"]
    onset = vae.decoder_calibration["onset_threshold"]
    decoders = dict(legacy=NoteDecoding(), held=vae.note_decoding)
    if decoders["held"].mode != "held":
        raise ValueError("Install a validated held-note decoding profile before evaluating")
    windows, skipped = collect_test(tokenizer, config, args.test_files, 30, args.seed)
    manifest = [{k: v for k, v in w.items() if k != "piece"} for w in windows]
    write_report(output / "test_selection.json", dict(split="test", piano_only=True, windows=manifest, skipped=skipped))
    protocol = dict(checkpoint_sha256=digest, seed=args.seed, clips=manifest, flow_files=args.flow_files,
                    activity=activity, onset=onset, note_decoding=vars(vae.note_decoding))
    protocol_path = output / "cache_identity.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output cache belongs to different weights, clips, seed or decoding settings")
    write_report(protocol_path, protocol)
    records = output / "records.jsonl"
    records.write_text("")
    aggregates = Aggregates()
    noise_rows = []
    scale = normalizer.std.float().cpu()[None, :, None, None]
    alphas = (0., .05, .1, .2, .3, .4)
    print(f"Checkpoint step={info['step']}; held={vars(vae.note_decoding)}; test clips={len(windows)}", flush=True)

    def record(row):
        with records.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    for index, window in enumerate(windows):
        reference = window["piece"]
        mean, logvar, weights = encode_piece(vae, reference, device)
        for draw in range(3):
            eps = paired_noise(mean.shape, seed_for(args.seed, "decoder-jitter", window["row_id"], draw))
            for alpha in (*alphas, "posterior"):
                if alpha == 0 and draw > 0:
                    continue
                z = mean + (logvar.mul(.5).exp() if alpha == "posterior" else alpha * scale) * eps
                confidence, pedal = decode_latents(vae, z * (weights > 0), reference, device)
                for name, options in decoders.items():
                    prediction = decode_prediction(reference, confidence, pedal, activity, onset, options)
                    score, metrics = measure(reference, prediction)
                    aggregates.add((str(alpha), name), score, metrics)
                    record(dict(experiment="decoder_sensitivity", clip=window["name"], draw=draw,
                                alpha=alpha, decoder=name, metrics=metrics))
        print(f"Decoder sweep {index + 1}/{len(windows)}", flush=True)
    noise_rows = aggregates.rows(("alpha", "decoder"))
    write_report(output / "decoder_sensitivity.json", dict(clips=len(windows), noise_draws=3,
        scale="Frozen flow checkpoint's training-only per-coordinate latent std", rows=noise_rows))

    # Six held-out piano clips plus the user's listening example, kept separate.
    flow_windows = [w for w in windows if int(w["piece"].onset.sum()) >= 8][:args.flow_files]
    if len(flow_windows) != args.flow_files:
        raise ValueError("Not enough active test piano clips for the flow comparison")
    clair_path = Path(__file__).with_name("examples") / "debussy-clair-de-lune_original.mid"
    clair = tokenizer.encode(clair_path, recover=True)
    clair = crop_piece(clair, 0, min(30, len(clair.velocity)))
    flow_windows.append(dict(name="clair-de-lune", piece=clair, path=str(clair_path), collection="listening-example"))
    with sqlite3.connect(Path(config.INDEX_PATH).resolve().as_uri() + "?mode=ro", uri=True) as db:
        for window in flow_windows[:-1]:
            tags = json.loads(db.execute("SELECT genres FROM files WHERE id=?", (window["row_id"],)).fetchone()[0])
            window["source_genres"] = tags
            window["conditioning_genre"] = conditioning_genre(tags, info["genres"])
    flow_totals = Aggregates()
    timings, latent_errors = defaultdict(list), defaultdict(list)
    listening = output / "listening"; listening.mkdir(exist_ok=True)
    cache = output / "flow_confidence"; cache.mkdir(exist_ok=True)
    flow_count = 0
    for window in flow_windows:
        reference = window["piece"]; n = len(reference.velocity); h = config.HISTORY_BINS
        item = window_tensors(reference, 0, config, mode="none")
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in collate_windows([item]).items()}
        with amp_context(device):
            original, weights = encode_windows(vae, batch, config.VAE_BATCH_SIZE)
        source = normalizer.normalize(original, weights)
        clean = source[0, h:h + n].float().cpu()
        is_clair = window["name"] == "clair-de-lune"
        genres = ("classical", "jazz") if is_clair else (window["conditioning_genre"],)
        if is_clair:
            tokenizer.decode(reference, target_only=True).save(listening / "clair_source.mid")
            confidence, pedal = decode_latents(vae, original[0, h:h + n], reference, device)
            for decoder, options in decoders.items():
                baseline = decode_prediction(reference, confidence, pedal, activity, onset, options)
                tokenizer.decode(baseline, target_only=True).save(listening / f"clair_vae_{decoder}.mid")
        for genre in genres:
            genre_id = 0 if genre == "none" else info["genres"].index(genre)
            batch["genre"] = torch.tensor([genre_id], device=device)
            for draw in range(2):
                seed = seed_for(args.seed, "flow", window["name"], draw)
                for strength in (.1, .3, .5, .7):
                    for steps in (48, 96, 192):
                        stem = f"{window['name']}-{genre}-d{draw}-s{strength}-n{steps}"
                        cached = cache / f"{stem}.npz"
                        if cached.exists():
                            with np.load(cached, allow_pickle=False) as saved:
                                confidence = {k: saved[k] for k in ("activity", "onset", "velocity")}
                                pedal = saved["pedal"]; residual = float(saved["nrmse"]); duration = float(saved["seconds"])
                        else:
                            torch.cuda.synchronize() if device.type == "cuda" else None
                            start = time.perf_counter()
                            with amp_context(device):
                                predicted = sample(model, source, batch["known"], weights, batch, steps=steps,
                                    strength=strength, generator=torch.Generator(device=device).manual_seed(seed))
                            torch.cuda.synchronize() if device.type == "cuda" else None
                            duration = time.perf_counter() - start
                            final = predicted[0, h:h + n].float().cpu()
                            residual = float((final - clean).square().mean().sqrt())
                            z = normalizer.denormalize(predicted, weights)[0, h:h + n]
                            confidence, pedal = decode_latents(vae, z, reference, device)
                            np.savez_compressed(cached, **confidence, pedal=pedal, nrmse=residual,
                                                seconds=duration, normalized_residual=(final - clean).numpy())
                        cohort = "clair-de-lune" if is_clair else "test"
                        group_genre = genre if is_clair else "source-conditioned"
                        timings[(cohort, group_genre, strength, steps)].append(duration)
                        latent_errors[(cohort, group_genre, strength, steps)].append(residual)
                        for decoder, options in decoders.items():
                            prediction = decode_prediction(reference, confidence, pedal, activity, onset, options)
                            score, metrics = measure(reference, prediction)
                            flow_totals.add((cohort, group_genre, strength, steps, decoder, "flow"), score, metrics)
                            record(dict(experiment="flow_steps", clip=window["name"], genre=genre, draw=draw,
                                        strength=strength, steps=steps, decoder=decoder, nrmse=residual,
                                        seconds=duration, metrics=metrics))
                            if is_clair and draw == 0 and steps in (48, 192) and strength in (.3, .5, .7):
                                tokenizer.decode(prediction, target_only=True).save(listening / f"{stem}-{decoder}.mid")
                        # Exact same RMS error, random direction: test structured flow errors versus isotropic jitter.
                        eps = paired_noise(clean.shape, seed_for(args.seed, "matched-rms", window["name"], draw))
                        eps = eps / eps.square().mean().sqrt().clamp_min(1e-8)
                        jitter = clean + residual * eps
                        jitter_raw = jitter * scale + normalizer.mean.float().cpu()[None, :, None, None]
                        jitter_confidence, jitter_pedal = decode_latents(vae, jitter_raw, reference, device)
                        for decoder, options in decoders.items():
                            jitter_prediction = decode_prediction(reference, jitter_confidence, jitter_pedal, activity, onset, options)
                            score, metrics = measure(reference, jitter_prediction)
                            flow_totals.add((cohort, group_genre, strength, steps, decoder, "matched_rms_jitter"), score, metrics)
                        flow_count += 1
                        print(f"Flow {flow_count}: {stem}; {duration:.2f}s; residual={residual:.3f}", flush=True)
        del batch, source, original, weights
    rows = flow_totals.rows(("cohort", "genre", "strength", "steps", "decoder", "perturbation"))
    for row in rows:
        key = tuple(row[k] for k in ("cohort", "genre", "strength", "steps"))
        row.update(mean_sampling_seconds=float(np.mean(timings[key])), mean_latent_nrmse=float(np.mean(latent_errors[key])))
    write_report(output / "flow_steps.json", dict(test_clips=len(flow_windows) - 1, noise_draws=2, rows=rows))
    if checkpoint_digest(args.checkpoint) != digest:
        raise RuntimeError("Flow checkpoint changed during diagnostics; results must not mix training states")
    write_report(output / "manifest.json", dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest,
        **info, model_config=asdict(model.config), test_files=len(windows), test_bins=sum(w["bins"] for w in windows),
        flow_files=[{k: v for k, v in w.items() if k != "piece"} for w in flow_windows],
        seed=args.seed, activity_threshold=activity, onset_threshold=onset, note_decoding=vars(vae.note_decoding),
        accompaniment="none", thresholds_selected_on="64 separate validation clips; no test-set tuning",
        caveat="Jazz source mismatches include intended edits. Clair de Lune is a listening example, not certified held-out data."))
    print(f"Complete: {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(defaults.SAVE_DIR) / "latest_checkpoint.pt")
    parser.add_argument("--test-files", type=int, default=32)
    parser.add_argument("--flow-files", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args(argv)
    if not 1 <= args.flow_files <= args.test_files:
        parser.error("Require 1 <= flow-files <= test-files")
    run(args)


if __name__ == "__main__":
    main()
