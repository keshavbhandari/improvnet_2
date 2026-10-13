"""Stage-2 transfer/reconstruction/continuation with concise, separated error diagnostics."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from improvnet.inference.local_vae_inference import Scores, inputs, load_model as load_vae, note_decoding_options, resolve_thresholds, write_report
from improvnet.model import latent_diffusion_config as defaults
from improvnet.model.latent_diffusion_model import LatentDiffusionConfig, LatentDiffusionModel
from improvnet.model.local_vae_train import amp_context
from improvnet.tokenizer.hierarchical import TempoMap, read_performance
from improvnet.utils.latent_diffusion import LatentNormalizer, checkpoint_digest, encode_windows, sample
from improvnet.utils.latent_diffusion_data import collate_windows, window_tensors
from improvnet.utils.note_decoding import NoteDecoding, apply_note_decoding, confidence_arrays


def validate_inference_config(model_config, config):
    """Window construction and attention must describe the same saved architecture."""
    fields = {
        "HISTORY_BINS": "history_bins", "TARGET_BINS": "target_bins",
        "PATCH_PITCH": "patch_pitch", "PATCH_TIME": "patch_time",
        "D_MODEL": "d_model", "N_LAYERS": "n_layers", "N_HEADS": "n_heads",
        "FF_MULTIPLIER": "ff_multiplier", "MAX_ACCOMPANIMENT_TRACKS": "max_tracks",
        "MAX_ACCOMPANIMENT_EVENTS": "max_events",
        "RHYTHM_FRAMES_PER_TOKEN": "rhythm_frames_per_token",
    }
    conflicts = [name for name, field in fields.items()
                 if getattr(config, name, None) != getattr(model_config, field)]
    if conflicts:
        raise ValueError("Inference settings differ from the diffusion checkpoint architecture: "
                         + ", ".join(conflicts))


def load_model(path, device, *, vae_checkpoint=None, use_ema=True):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "latent_diffusion_training_v1":
        raise ValueError("Expected a stage-2 latent diffusion checkpoint")
    config = SimpleNamespace(**state["train_config"])
    architecture = LatentDiffusionConfig(**state["model_config"])
    validate_inference_config(architecture, config)
    vae_path = Path(vae_checkpoint or config.VAE_CHECKPOINT)
    if checkpoint_digest(vae_path) != state["vae_sha256"]:
        raise ValueError("This stage-2 checkpoint belongs to a different VAE; decoder/latents cannot be mixed")
    vae, tokenizer, vae_step, _ = load_vae(vae_path, device)
    vae.requires_grad_(False)
    model = LatentDiffusionModel(architecture)
    c = model.config
    if tuple(vae.config.latent_shape) != (c.latent_channels, c.latent_pitch, c.latent_time) or vae.config.frames_per_bin != c.frames_per_bin:
        raise ValueError("Stage-2 latent shape differs from its frozen VAE")
    model.load_state_dict(state["ema_state_dict"] if use_ema else state["model_state_dict"], strict=True)
    info = dict(step=state["step"], vae_step=vae_step, genres=state["genres"], weights="ema" if use_ema else "raw")
    return model.to(device).eval(), vae, tokenizer, LatentNormalizer(**state["normalizer"]), config, info


def resize_piece(piece, bins, *, extend=False):
    """Trim all arrays/events/metadata consistently; optionally extend a prefix canvas."""
    if bins < 1 or bins * piece.config.frames_per_bin > piece.config.max_frames:
        raise ValueError("Invalid output canvas length")
    result = deepcopy(piece)
    old, f = len(piece.velocity), piece.config.frames_per_bin
    for name in ("velocity", "onset", "pedal", "valid_mask"):
        value = getattr(piece, name)
        array = np.zeros((bins, *value.shape[1:]), dtype=value.dtype)
        array[:min(old, bins)] = value[:min(old, bins)]
        if name == "valid_mask" and extend:
            array[:] = True
        setattr(result, name, array)
    length = result.n_frames
    rows = piece.events.copy()
    starts, ends = rows[:, 6] * f + rows[:, 7], rows[:, 8] * f + rows[:, 9]
    rows = rows[starts < length]
    end = np.minimum(ends[starts < length], length)
    rows[:, 8], rows[:, 9] = end // f, end % f
    result.events = rows
    result.event_ptr = np.r_[0, np.cumsum(np.bincount(rows[:, 6], minlength=bins))].astype(np.int64)
    tempo = TempoMap(result.metadata["ppq"], result.metadata["tempo_map"])
    duration = length * piece.config.fine_ms / 1000
    result.metadata["messages"] = [m for m in result.metadata["messages"] if tempo.to_seconds(m["tick"]) <= duration]
    return result.refresh_coarse().validate()


def reference_events(path, grid, *, time_scale=1., offset_seconds=0.):
    """All tracks of an external MIDI, including its piano; no silent beat alignment."""
    if not np.isfinite(time_scale) or time_scale <= 0 or not np.isfinite(offset_seconds):
        raise ValueError("Invalid reference time scale/offset")
    rows = []
    for n in read_performance(path, recover=True)["notes"]:
        start, end = n.start * time_scale + offset_seconds, n.end * time_scale + offset_seconds
        if end <= 0:
            continue
        start_frame, end_frame = grid.frame(max(0., start)), grid.frame(end)
        end_frame = max(start_frame + 1, end_frame)
        f = grid.frames_per_bin
        rows.append((n.track, n.channel, n.program, int(n.channel == 9), n.pitch, n.velocity,
                     start_frame // f, start_frame % f, end_frame // f, end_frame % f, n.order))
    rows.sort(key=lambda e: (e[6], e[7], e[10]))
    return np.array(rows, dtype=np.int64).reshape(-1, 11)


def score(reference, prediction, ids):
    scores = Scores()
    scores.arrays(reference, prediction, ids); scores.notes(reference, prediction, indices=ids)
    return scores.report()


def diagnostics(reference, vae_baseline, prediction, ids, latent_squared_sum, latent_cells, task):
    baseline, generated = score(reference, vae_baseline, ids), score(reference, prediction, ids)
    nrmse = float(np.sqrt(latent_squared_sum / latent_cells)) if latent_cells else None
    note = ("Reconstruction task: latent distance measures stage-2 prediction error, normalized by training channel std. "
            "VAE-only errors measure the stage-1 floor. Large output errors do not by themselves imply the VAE needs retraining.")
    if task != "reconstruct":
        note = ("Transfer/continuation: source distances include intentional musical changes, not only denoising errors. "
                "Use --task reconstruct at several noise strengths to diagnose stage 1 versus stage 2.")
    return dict(vae_only=baseline, generated_vs_source=generated,
                latent_nrmse_vs_source=round(nrmse, 5) if nrmse is not None else None, interpretation=note)


@torch.inference_mode()
def generate(model, vae, normalizer, piece, config, *, genre=0, task="transfer", prompt_bins=0,
             max_bins=32, steps=48, strength=1., mode="full", events=None, seed=42,
             activity_threshold=None, onset_threshold=None, progress=True, note_decoding=None):
    validate_inference_config(model.config, config)
    calibration = getattr(vae, 'decoder_calibration', None) or {}
    activity_threshold = calibration.get('activity_threshold', .5) if activity_threshold is None else activity_threshold
    onset_threshold = calibration.get('onset_threshold', .99) if onset_threshold is None else onset_threshold
    if task not in ("transfer", "reconstruct", "continue") or max_bins < 1 or prompt_bins < 0:
        raise ValueError("Invalid task/output/prompt length")
    if prompt_bins > len(piece.velocity) or (prompt_bins == len(piece.velocity) and task != "continue"):
        raise ValueError("Prompt leaves no target bins in the source")
    if task == "continue" and (prompt_bins < 1 or strength != 1 or piece.n_frames < prompt_bins * piece.config.frames_per_bin):
        raise ValueError("Continuation requires a complete-bin prompt and noise strength 1")
    if not 0 < activity_threshold < 1 or not 0 < onset_threshold < 1:
        raise ValueError("Invalid decoder thresholds")
    if tuple(vae.config.latent_shape) != (model.config.latent_channels, model.config.latent_pitch, model.config.latent_time):
        raise ValueError("VAE/model shape mismatch")
    length = prompt_bins + max_bins if task == "continue" else min(len(piece.velocity), prompt_bins + max_bins)
    reference = resize_piece(piece, length, extend=task == "continue")
    output, baseline = deepcopy(reference), deepcopy(reference)
    options = note_decoding or getattr(vae, "note_decoding", NoteDecoding())
    options = options if isinstance(options, NoteDecoding) else NoteDecoding(**options)

    def new_confidence():
        confidence = {k: np.zeros(reference.velocity.shape, dtype=np.float32)
                      for k in ("activity", "onset", "velocity")}
        # Only the explicitly preserved prompt supplies initial state. Generated
        # target confidences are filled from model output, never the source roll.
        confidence["activity"][:prompt_bins] = reference.velocity[:prompt_bins] > 0
        confidence["onset"][:prompt_bins] = reference.onset[:prompt_bins]
        confidence["velocity"][:prompt_bins] = reference.velocity[:prompt_bins] / 127.
        return confidence

    output_confidence, baseline_confidence = (new_confidence(), new_confidence()) if options.mode == "held" else (None, None)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    # Prior target history is reused as latents, not decoded/re-encoded every window.
    # This avoids accumulating an extra VAE round-trip error at each boundary.
    committed = {}
    latent_squared_sum = latent_cells = 0.
    blocks = []
    for start in range(prompt_bins, length, config.TARGET_BINS):
        item = window_tensors(reference, start, config, mode=mode, events=events)
        item["genre"] = torch.tensor(genre)
        if task == "continue":
            item["pitch"][config.HISTORY_BINS:].zero_()
            item["pitch_present"][config.HISTORY_BINS:].zero_()
            item["roll"][config.HISTORY_BINS:].zero_()
            item["pedal"][config.HISTORY_BINS:].zero_()
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in collate_windows([item]).items()}
        with amp_context(device):
            original_z, weights = encode_windows(vae, batch, config.VAE_BATCH_SIZE)
        source = normalizer.normalize(original_z, weights)
        known = batch["known"]
        for slot in known[0].nonzero().flatten().tolist():
            absolute = start - config.HISTORY_BINS + slot
            if absolute in committed:
                source[0, slot] = committed[absolute]
        with amp_context(device):
            prediction = sample(model, source, known, weights, batch, steps=steps, strength=strength, generator=generator)
        h = config.HISTORY_BINS
        count = min(config.TARGET_BINS, length - start)
        ids = np.arange(start, start + count)
        z = normalizer.denormalize(prediction, weights)[0, h:h + count]
        mask = batch["valid_mask"][0, h:h + count]
        enabled = batch["pedal_enabled"].expand(count)
        with amp_context(device):
            rebuilt = vae.decode(z)
        if not all(bool(torch.isfinite(x).all()) for x in (rebuilt.activity_logits, rebuilt.onset_logits, rebuilt.velocity, rebuilt.pedal)):
            raise FloatingPointError("Non-finite VAE decoding")
        arrays = rebuilt.tensors(mask, enabled, activity_threshold=activity_threshold, onset_threshold=onset_threshold)
        for key, value in arrays.items():
            getattr(output, key)[ids] = value.cpu().numpy()
        if output_confidence is not None:
            for key, value in confidence_arrays(rebuilt).items():
                output_confidence[key][ids] = value
        for offset, absolute in enumerate(ids):
            committed[int(absolute)] = prediction[0, h + offset].clone()
        # Bound cached history to the context actually used by the next window.
        committed = {k: v for k, v in committed.items() if k >= start + count - config.HISTORY_BINS}
        # Reference baseline is independent of the generated history and conditioning.
        with amp_context(device):
            roll, pedal, valid, enabled = inputs(reference, ids, device)
            posterior = vae.encode(roll, pedal, valid)
            r = vae.decode(posterior.mean)
        if baseline_confidence is not None:
            for key, value in confidence_arrays(r).items():
                baseline_confidence[key][ids] = value
        for key, value in r.tensors(valid, enabled, activity_threshold=activity_threshold, onset_threshold=onset_threshold).items():
            getattr(baseline, key)[ids] = value.cpu().numpy()
        target_normalized = normalizer.normalize(posterior.mean[None], posterior.weights[None])
        w = posterior.weights[None].expand_as(target_normalized)
        error_sum = float(((prediction[:, h:h + count].float() - target_normalized).square() * w).sum())
        cells = float(w.sum())
        latent_squared_sum += error_sum; latent_cells += cells
        block = dict(start_bin=start, bins=count, latent_nrmse_vs_source=round(float(np.sqrt(error_sum / cells)), 5),
                     **item["accompaniment_info"])
        # Avoid embedding track lists and lots of diagnostics into the final JSON.
        block["tracks"] = len(block.pop("parts"))
        blocks.append(block)
        if progress:
            print(f"Bins {start}:{start+count}: {steps} passes; source distance={block['latent_nrmse_vs_source']:.3f}; "
                  f"accompaniment={mode}, tracks={block['tracks']}, omitted events={block['dropped_events']}", flush=True)
    if output_confidence is not None:
        for target, confidence in ((output, output_confidence), (baseline, baseline_confidence)):
            apply_note_decoding(target, np.arange(length), confidence, options,
                activity_threshold=activity_threshold, onset_threshold=onset_threshold)
    output.refresh_coarse().validate(); baseline.refresh_coarse().validate()
    ids = np.arange(prompt_bins, length)
    report = diagnostics(reference, baseline, output, ids, latent_squared_sum, latent_cells, task)
    report["diffusion_architecture"] = dict(asdict(model.config),
        tokens_per_bin=int(np.prod(model.config.spatial_grid)),
        tokens_per_window=int(model.config.bins * np.prod(model.config.spatial_grid)))
    report["note_decoding"] = vars(options)
    report["accompaniment"] = dict(mode=mode, max_tracks=config.MAX_ACCOMPANIMENT_TRACKS,
        event_budget=config.MAX_ACCOMPANIMENT_EVENTS, max_tracks_used=max(b["tracks"] for b in blocks),
        omitted_events=sum(b["dropped_events"] for b in blocks))
    # A compact decoder stress check at the measured residual magnitude, ONLY for reconstruction.
    # This is illustrative isotropic jitter, not evidence that actual stage-2 errors are isotropic.
    if task == "reconstruct":
        jittered = deepcopy(reference)
        jitter_confidence = new_confidence() if options.mode == "held" else None
        alpha = report["latent_nrmse_vs_source"]
        jitter_generator = torch.Generator(device=device).manual_seed(seed + 100003)
        for offset in range(0, len(ids), config.VAE_BATCH_SIZE):
            selected = ids[offset:offset + config.VAE_BATCH_SIZE]
            with amp_context(device):
                roll, pedal, valid, enabled = inputs(reference, selected, device)
                p = vae.encode(roll, pedal, valid)
                eps = torch.randn(p.mean.shape, device=device, generator=jitter_generator)
                jitter = p.mean + alpha * normalizer.std.to(device)[None, :, None, None] * eps
                r = vae.decode(jitter.masked_fill(~(p.weights > 0), 0))
            if jitter_confidence is not None:
                for key, value in confidence_arrays(r).items():
                    jitter_confidence[key][selected] = value
            for key, value in r.tensors(valid, enabled, activity_threshold=activity_threshold, onset_threshold=onset_threshold).items():
                getattr(jittered, key)[selected] = value.cpu().numpy()
        if jitter_confidence is not None:
            apply_note_decoding(jittered, np.arange(length), jitter_confidence, options,
                activity_threshold=activity_threshold, onset_threshold=onset_threshold)
        jittered.refresh_coarse().validate()
        report["vae_jitter_at_measured_residual"] = score(reference, jittered, ids)
        report["jitter_caveat"] = "One isotropic draw at measured std-relative RMSE; actual diffusion errors may be structured."
    return output, baseline, reference, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(defaults.SAVE_DIR) / "latest_checkpoint.pt")
    parser.add_argument("--vae-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task", choices=("transfer", "reconstruct", "continue"), default="transfer")
    parser.add_argument("--genre", default="jazz", help="Known training genre, or 'none' for unconditioned")
    parser.add_argument("--prompt-bins", type=int, default=0)
    parser.add_argument("--max-bins", type=int, default=32, help="Generated bins, excluding the preserved prompt")
    parser.add_argument("--passes", type=int, default=48)
    parser.add_argument("--noise-strength", type=float, default=1.)
    parser.add_argument("--accompaniment-mode", choices=("full", "rhythm", "none"), default="rhythm")
    parser.add_argument("--accompaniment-midi", type=Path, help="Optional external conditioning MIDI; not mixed into output")
    parser.add_argument("--reference-time-scale", type=float, default=1., help="Explicit external rhythm/tempo alignment")
    parser.add_argument("--reference-offset-seconds", type=float, default=0.)
    parser.add_argument("--target-channel", type=int)
    parser.add_argument("--target-program", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activity-threshold", type=float, help="Override VAE checkpoint calibration")
    parser.add_argument("--onset-threshold", type=float, help="Override VAE checkpoint calibration")
    parser.add_argument("--note-decoding", choices=("legacy", "held"), help="Override saved MIDI note assembly")
    parser.add_argument("--continuation-threshold", type=float)
    parser.add_argument("--max-gap-ms", type=int)
    parser.add_argument("--raw-weights", action="store_true", help="Use raw rather than EMA stage-2 weights")
    args = parser.parse_args(argv)
    if args.output_dir and args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose an empty/new output directory; existing artifacts are never overwritten")
    device = torch.device(args.device)
    model, vae, tokenizer, normalizer, config, info = load_model(args.checkpoint, device,
        vae_checkpoint=args.vae_checkpoint, use_ema=not args.raw_weights)
    resolve_thresholds(args, vae)
    if args.genre.lower() == "none":
        genre = 0
    elif args.genre.lower() in info["genres"]:
        genre = info["genres"].index(args.genre.lower())
    else:
        parser.error(f"Unknown genre {args.genre!r}; available={info['genres'][1:]}")
    piece = tokenizer.encode(args.input, recover=True, seed=args.seed,
        target_channel=args.target_channel, target_program=args.target_program)
    if not piece.metadata["target_present"]:
        raise ValueError("No source target notes")
    events = reference_events(args.accompaniment_midi, tokenizer.config,
        time_scale=args.reference_time_scale, offset_seconds=args.reference_offset_seconds) if args.accompaniment_midi else None
    output, baseline, reference, report = generate(model, vae, normalizer, piece, config, genre=genre,
        task=args.task, prompt_bins=args.prompt_bins, max_bins=args.max_bins, steps=args.passes,
        strength=args.noise_strength, mode=args.accompaniment_mode, events=events, seed=args.seed,
        activity_threshold=args.activity_threshold, onset_threshold=args.onset_threshold,
        note_decoding=note_decoding_options(args))
    if args.output_dir is None:
        root = Path(defaults.EVALUATION_DIR); root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="single-", dir=root))
    else:
        directory = args.output_dir; directory.mkdir(parents=True, exist_ok=True)
    tokenizer.decode(reference, target_only=True).save(directory / "reference_target.mid")
    tokenizer.decode(baseline, target_only=True).save(directory / "vae_reconstruction.mid")
    tokenizer.decode(output, target_only=True).save(directory / "generated_target.mid")
    tokenizer.decode(output).save(directory / "generated_source_mix.mid")
    report = dict(step=info["step"], vae_step=info["vae_step"], weights=info["weights"], task=args.task,
        genre=args.genre, target_program=piece.metadata["target_program"], is_drum=piece.metadata["target_is_drum"],
        prompt_bins=args.prompt_bins, generated_bins=len(output.velocity) - args.prompt_bins,
        passes=args.passes, noise_strength=args.noise_strength,
        external_accompaniment=bool(args.accompaniment_midi), **report)
    write_report(directory / "report.json", report)
    print(json.dumps(report, allow_nan=False))
    print(f"Saved MIDI and concise report: {directory}")
    return report


if __name__ == "__main__":
    main()
