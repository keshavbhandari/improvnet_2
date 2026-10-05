"""Held-out beginning-of-piece selection and metrics for constrained transfer."""

from collections import Counter
from dataclasses import asdict
import csv
import gc
import json
import math
from pathlib import Path
import random
import statistics
from types import SimpleNamespace

import torch

from improvnet.inference.uniform_diffusion_inference import closed_source
from improvnet.metrics.midi_syntax import MidiSyntaxMetric
from improvnet.metrics.pitch_similarity import PitchSimilarityMetric


def write_json(path, value):
    with open(path, "x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def syntax_dict(result):
    value = asdict(result)
    value["issue_counts"] = dict(Counter(issue.code for issue in result.issues))
    return value


def select_test_examples(data_config, processor, *, count, max_blocks, seed):
    from improvnet.utils.ar_utils import load_split_byte_index
    from improvnet.metrics.genre_classifier.audit import normalize_genre
    if not {"JSONL_FILES", "DATA_SPLIT_SEED", "DATA_SPLIT_RATIOS"} <= data_config.keys():
        raise ValueError("Checkpoint lacks split metadata; cannot claim held-out examples")
    config = SimpleNamespace(**data_config)
    candidates = [(str(path), int(offset)) for path in config.JSONL_FILES
                  for offset in load_split_byte_index(str(path), "test", config=config)]
    random.Random(seed).shuffle(candidates)
    selected, seen, handles, rejected = [], set(), {}, Counter()
    syntax = MidiSyntaxMetric(processor.tokenizer)
    try:
        for path, offset in candidates:
            if path not in handles:
                handles[path] = open(path, "rb")
            handles[path].seek(offset)
            entry = json.loads(handles[path].readline())
            if normalize_genre(entry.get("genre")) != "classical":
                rejected["not_classical"] += 1
                continue
            identity = str(entry.get("midi_filepath") or f"{path}:{offset}")
            if identity in seen:
                rejected["duplicate_performance"] += 1
                continue
            tokens = processor.serialized_tokens_to_tokens(entry.get("tokens", []))
            while tokens and tokens[-1] == processor.tokenizer.pad_tok:
                tokens.pop()
            source_length = len(tokens)
            try:
                tokens = closed_source(tokens, max_blocks*256)
            except ValueError:
                rejected["no_closed_source"] += 1
                continue
            grade = syntax.evaluate_canvas(tokens, allow_incomplete_tail=False)
            if len(tokens) < min(512, max_blocks*256)-3 or not grade.valid:
                rejected["short_or_invalid_source"] += 1
                continue
            seen.add(identity)
            selected.append(dict(example_id=f"example_{len(selected):02d}", sample_index=len(selected),
                                 midi_filepath=identity, jsonl_path=path, byte_offset=offset,
                                 source_genre="classical", source_tokens=source_length, target_start=0,
                                 tokens=tokens, source_syntax=syntax_dict(grade)))
            if len(selected) == count:
                break
    finally:
        for handle in handles.values():
            handle.close()
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} eligible classical test pieces; requested {count}")
    return selected, dict(candidate_test_records=len(candidates), rejected=dict(rejected), seed=seed,
                          eligibility="distinct classical performances in diffusion test split; beginning at token 0; >=min(512, token cap)-3 tokens; source MIDI-valid")


def load_encoder(path, kind, device, tokenizer):
    """Load only trained metric weights; mmap optimizer tensors remain on disk."""
    from improvnet.metrics.genre_classifier.tokenization import vocabulary_digest
    if kind == "genre":
        from improvnet.metrics.genre_classifier.genre_classifier_model import GenreClassifier as Model, GenreClassifierConfig as Config, RotaryEmbedding
        from improvnet.metrics.genre_classifier.scoring import GenreClassifierMetric as Metric
        expected_format = "genre_encoder_v2"
    elif kind == "continuity":
        from improvnet.metrics.continuity.continuity_model import ContinuityEncoder as Model, ContinuityConfig as Config, RotaryEmbedding
        from improvnet.metrics.continuity.scoring import ContinuityMetric as Metric
        expected_format = "continuity_encoder_v1"
    else:
        raise ValueError("Unknown metric encoder")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if state.get("format") != expected_format or state.get("trained_steps", 0) < 1:
        raise ValueError(f"Need trained {kind} checkpoint with format {expected_format}")
    if state.get("vocabulary_digest") != vocabulary_digest(tokenizer):
        raise ValueError("Metric checkpoint vocabulary mismatch")
    if kind == "continuity" and state.get("tokenizer_config") != tokenizer.config:
        raise ValueError("Continuity tokenizer differs from training")
    with torch.device("meta"):
        model = Model(Config(**state["config"]))
    model.load_state_dict(state["model_state_dict"], strict=True, assign=True)
    frequencies = []
    for layer in model.layers:
        freq = RotaryEmbedding(layer.attn.head_dim).inv_freq
        layer.attn.rope.inv_freq = freq
        frequencies.append(freq)
    model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
    for layer, freq in zip(model.layers, frequencies):
        layer.attn.rope.inv_freq = freq.to(device)
    model.requires_grad_(False)
    metadata = dict(path=str(Path(path).resolve()), trained_steps=state["trained_steps"], config=state["config"],
                    vocabulary_digest=state["vocabulary_digest"], dtype=str(next(model.parameters()).dtype))
    if kind == "genre":
        metadata.update(labels=state["labels"], temperature=state.get("temperature", 1.), prior_correction=state.get("prior_correction"))
        metric = Metric(model, tokenizer, state["labels"], temperature=state.get("temperature", 1.),
                        prior_correction=state.get("prior_correction"))
    else:
        metric = Metric(model, tokenizer)
    return metric, metadata


def release_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def statistics_for(values):
    values = [float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Nonfinite score")
    return dict(count=len(values), mean=statistics.fmean(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else None)


def note_evidence(tokens):
    return any(isinstance(t, tuple) and t[0] == "note" for t in tokens)


def musical_diagnostics(tokens, segment_ms):
    """Non-reward diagnostics: syntax/continuity can miss ordering or repetition."""
    segment, last, decreases, onsets, pitches = 0, None, 0, 0, []
    for token in tokens:
        if token == "<T>":
            segment += 1
        elif isinstance(token, tuple) and token[0] == "onset":
            absolute = segment*segment_ms + token[1]
            decreases += int(last is not None and absolute < last)
            last, onsets = absolute, onsets+1
        elif isinstance(token, tuple) and token[0] == "note":
            pitches.append(token[1])
    frequencies = Counter(pitches)
    return dict(onsets=onsets, adjacent_onset_decreases=decreases,
                onset_decrease_rate=decreases/max(onsets-1, 1), unique_pitches=len(frequencies),
                most_common_pitch_fraction=max(frequencies.values(), default=0)/max(len(pitches), 1))


def pitch_score(metric, reference, generated, reference_context=(), generated_context=()):
    try:
        return dict(status="ok", **metric.evaluate_tokens(reference, generated,
                    reference_context=reference_context, generated_context=generated_context).to_dict())
    except ValueError as error:
        if "Reference has no pitched notes" not in str(error):
            raise
        return dict(status="not_applicable", reason="drum_only_reference")


def inspect_syntax(report, processor):
    """Grade actual outputs, including joins; don't inject GT boundary suffixes."""
    syntax = MidiSyntaxMetric(processor.tokenizer)
    examples = {e["example_id"]: e for e in report["examples"]}
    for row in report["results"]:
        generated = row["generated_tokens"]
        source = examples[row["example_id"]]["tokens"]
        row["syntax"] = syntax_dict(syntax.evaluate_canvas(generated, allow_incomplete_tail=False, noise_level=row["initial_noise"]))
        row["frozen_slots_preserved"] = all(a == b for a, b in zip(source, generated)
                                           if not isinstance(a, tuple) or a[0] not in ("note", "onset", "dur")) and len(source) == len(generated)
        row["token_accuracy"] = sum(a == b for a, b in zip(source, generated))/len(source)
        row["pitch_velocity_preserved"] = len(source) == len(generated) and all(
            a == b for a, b in zip(source, generated) if isinstance(a, tuple) and a[0] == "note")
        timing = [(a, b) for a, b in zip(source, generated) if isinstance(a, tuple) and a[0] in ("onset", "dur")]
        row["timing_changed_fraction"] = sum(a != b for a, b in timing)/len(timing) if timing else 0.
        row["musical_diagnostics"] = musical_diagnostics(generated, processor.tokenizer.abs_time_step_ms)
        row["source_musical_diagnostics"] = musical_diagnostics(source, processor.tokenizer.abs_time_step_ms)
        for b in row["blocks"]:
            start, end = b["source_start"], b["source_end"]
            tokens = generated[start:end]
            b["syntax"] = syntax_dict(syntax.evaluate_canvas(tokens, history=generated[:start], noise_level=row["initial_noise"]))
            b["token_accuracy"] = sum(a == c for a, c in zip(source[start:end], tokens))/len(tokens)


@torch.inference_mode()
def score_metrics(report, args, processor, device):
    examples = {e["example_id"]: e for e in report["examples"]}
    pitch = PitchSimilarityMetric()
    genre_only = getattr(args, "metrics", "all") == "genre"
    eligible = [r for r in report["results"] if r["syntax"]["valid"]]
    for row in report["results"]:
        row["metrics"] = {name: dict(status="skipped", reason="invalid_output") for name in ("pitch", "genre", "continuity")}
        for b in row["blocks"]:
            b["metrics"] = {name: dict(status="skipped", reason="invalid_output") for name in ("pitch", "genre", "continuity")}
    for row in ([] if genre_only else eligible):
        source = examples[row["example_id"]]["tokens"]
        generated = row["generated_tokens"]
        row["metrics"]["pitch"] = pitch_score(pitch, source, generated)
        for b in row["blocks"]:
            start, end = b["source_start"], b["source_end"]
            b["metrics"]["pitch"] = pitch_score(pitch, source[start:end], generated[start:end], source[:start], generated[:start])
        print(f"  pitch: {row['example_id']} noise={row['initial_noise']:.0%}", flush=True)
    genre, metadata = load_encoder(args.genre_checkpoint, "genre", device, processor.tokenizer)
    report["metric_checkpoints"] = {"genre": metadata}
    reference_genres = {}
    for row in eligible:
        source = examples[row["example_id"]]["tokens"]
        generated = row["generated_tokens"]
        block_items = [] if genre_only else [
                (b, source[b["source_start"]:b["source_end"]], generated[b["source_start"]:b["source_end"]], (row["example_id"], b["block"])) for b in row["blocks"]]
        for item, reference, tokens, key in [(row, source, generated, (row["example_id"], "whole"))] + block_items:
            if not note_evidence(tokens):
                item["metrics"]["genre"] = dict(status="not_applicable", reason="no_notes")
                continue
            if key not in reference_genres:
                reference_genres[key] = genre.probabilities(reference, window_batch_size=1)
            probabilities = genre.probabilities(tokens, window_batch_size=1)
            baseline = reference_genres[key]["jazz"]
            blues_baseline = reference_genres[key].get("blues")
            item["metrics"]["genre"] = dict(status="ok", probabilities=probabilities,
                target_probability=probabilities.get(row["target_genre"]), jazz_probability=probabilities["jazz"],
                blues_probability=probabilities.get("blues"), source_probability=probabilities.get(row["source_genre"]),
                reference_jazz_probability=baseline, jazz_delta_from_reference=probabilities["jazz"]-baseline,
                reference_blues_probability=blues_baseline,
                blues_delta_from_reference=probabilities.get("blues")-blues_baseline if blues_baseline is not None else None)
    del genre
    release_gpu()
    if genre_only:
        for row in eligible:
            for name in ("pitch", "continuity"):
                row["metrics"][name] = dict(status="not_requested")
        return
    continuity, metadata = load_encoder(args.continuity_checkpoint, "continuity", device, processor.tokenizer)
    report["metric_checkpoints"]["continuity"] = metadata
    baselines = {}
    for row in eligible:
        source = examples[row["example_id"]]["tokens"]
        generated = row["generated_tokens"]
        for b in row["blocks"]:
            start, end = b["source_start"], b["source_end"]
            if not start or not note_evidence(generated[max(0, start-256):start]) or not note_evidence(generated[start:end]):
                b["metrics"]["continuity"] = dict(status="not_applicable", reason="promptless_first_block_or_no_notes")
                continue
            key = (row["example_id"], b["block"])
            if key not in baselines:
                baselines[key] = continuity.score_boundary(source[:start], source[start:end], window_size=256)
            score = continuity.score_boundary(generated[:start], generated[start:end], window_size=256)
            cross = continuity.score_boundary(source[:start], generated[start:end], window_size=256)
            b["metrics"]["continuity"] = dict(status="ok", score=score, reference_score=baselines[key],
                                               delta_from_reference=score-baselines[key], source_history_score=cross)
        boundaries = [b["metrics"]["continuity"] for b in row["blocks"]]
        row["metrics"]["continuity"] = dict(status="ok", **{name: statistics_for(v.get(name) for v in boundaries)["mean"]
                for name in ("score", "reference_score", "delta_from_reference", "source_history_score")},
                boundaries=sum(v["status"] == "ok" for v in boundaries))
        if not row["metrics"]["continuity"]["boundaries"]:
            row["metrics"]["continuity"].update(status="not_applicable", reason="no_musical_boundary")
    del continuity
    release_gpu()


def summarize(results):
    summary = []
    for noise, steps in sorted({(r["initial_noise"], r.get("max_steps", 48)) for r in results}, key=lambda p: (-p[0], p[1])):
        group = [r for r in results if (r["initial_noise"], r.get("max_steps", 48)) == (noise, steps)]
        valid = [r for r in group if r["syntax"]["valid"]]
        blocks = [b for r in group for b in r["blocks"]]
        row = dict(noise=noise, steps=steps, pieces=len(group), valid=sum(r["syntax"]["valid"] for r in group),
                   compiles=sum(r["syntax"]["midi_compiles"] is True for r in group), blocks=len(blocks),
                   frozen_preserved=sum(r["frozen_slots_preserved"] for r in group),
                   syntax_score=statistics_for(r["syntax"]["score"] for r in group),
                   passes_per_block=statistics_for(sum(s["actual_steps"] for s in b["stages"].values()) for b in blocks))
        for name in ("score", "pitch_identity", "pattern_preservation", "structural_preservation"):
            row[f"pitch_{name}"] = statistics_for(r["metrics"]["pitch"].get(name) for r in valid)
        for name in ("jazz_probability", "jazz_delta_from_reference", "reference_jazz_probability", "blues_probability", "target_probability"):
            row[name] = statistics_for(r["metrics"]["genre"].get(name) for r in valid)
        for name in ("score", "reference_score", "delta_from_reference", "source_history_score"):
            row[f"continuity_{name}"] = statistics_for(r["metrics"]["continuity"].get(name) for r in valid)
        for name in ("onset_decrease_rate", "most_common_pitch_fraction"):
            row[name] = statistics_for(r.get("musical_diagnostics", {}).get(name) for r in valid)
        summary.append(row)
    return summary


def compact_report(directory, report):
    """Small settings/baseline/results only: no tokens, traces or model configs."""
    baselines, results = {}, []
    for row in report["results"]:
        genre = row["metrics"]["genre"]
        baselines[row["example_id"]] = dict(jazz=genre.get("reference_jazz_probability"), blues=genre.get("reference_blues_probability"))
        value = dict(example=row["example_id"], noise=row["initial_noise"], steps=row["max_steps"],
                     jazz=genre.get("jazz_probability"), blues=genre.get("blues_probability"),
                     valid_midi=row["syntax"]["valid"], pitch_velocity_preserved=row["pitch_velocity_preserved"],
                     timing_changed_fraction=row["timing_changed_fraction"],
                     mean_passes_per_block=statistics_for(sum(s["actual_steps"] for s in b["stages"].values()) for b in row["blocks"])["mean"],
                     onset_decrease_rate=row["musical_diagnostics"]["onset_decrease_rate"],
                     midi=str(Path(row["output_midi"]).relative_to(directory)) if row["output_midi"] else None)
        if report.get("metrics_requested", "all") == "all":
            value.update(pitch_similarity=row["metrics"]["pitch"].get("score"),
                         pitch_identity=row["metrics"]["pitch"].get("pitch_identity"),
                         patterns=row["metrics"]["pitch"].get("pattern_preservation"),
                         ssm=row["metrics"]["pitch"].get("structural_preservation"),
                         continuity=row["metrics"]["continuity"].get("score"))
        # Six significant digits preserve tiny genre probabilities without long
        # float representations. No percentage conversion inside JSON.
        results.append({k: float(f"{v:.6g}") if isinstance(v, float) else v for k, v in value.items()})
    return dict(input=report["selection"].get("input"), target_genre=report["target_genre"],
                strategy=report["strategy"], checkpoint_step=report["checkpoint"]["step"],
                model=report["checkpoint_path"], genre_model=report["metric_checkpoints"]["genre"]["path"],
                max_blocks=report["max_blocks"], block_size=256, seed=report["seed"],
                steps=report["step_budgets"], noise_levels=report["noise_levels"],
                sampler={k: v for k, v in report["sampler_config"].items() if k != "steps"}, baseline=baselines, results=results)


def write_report(directory, report, *, save_details=False):
    compact = compact_report(directory, report)
    write_json(directory/"scores.json", compact)
    if save_details:
        write_json(directory/"details.json", report)
    def percent(value):
        return "N/A" if value is None else f"{100*value:.6g}%"
    strategy = report["strategy"]
    lines = [f"# {report['target_genre'].capitalize()} transfer: {strategy}", "",
        f"{len(report['examples'])} piece(s); first up to {report['max_blocks']} × 256 source tokens; "
        f"pass budgets {report['step_budgets']} per stage. First block is promptless; later history is generated.", "",
        "Only onset/duration fields change; pitch, velocity, instruments and structure stay fixed." if strategy == "timing" else
        "Timing changes first, then joint pitch/velocity; instruments and structure stay fixed.", ""]
    for key, baseline in compact["baseline"].items():
        lines.append(f"Source {key}: jazz **{percent(baseline['jazz'])}**, blues **{percent(baseline['blues'])}**.")
    lines += ["", "Scores are classifier probabilities for the complete generated excerpt, not perceptual judgments.", "",
              "| Piece | Timing noise | Pass budget | Mean actual passes/block | Jazz | Blues | MIDI valid | Output |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for row in compact["results"]:
        output = f"[MIDI]({row['midi']})" if row["midi"] else "not saved"
        lines.append(f"| {row['example']} | {row['noise']:.0%} | {row['steps']} | {row['mean_passes_per_block']:.1f} | "
                     f"{percent(row['jazz'])} | {percent(row['blues'])} | {row['valid_midi']} | {output} |")
    lines += ["", "Noise applies only to editable slots. Adaptive stopping can use fewer passes than the budget. "
              "The checkpoint has no timestep input; effective editable noise scales the sampler temperature. "
              "Unchanged note tokens do not guarantee unchanged audible ordering: perturbed onsets can reorder notes.", "",
              "scores.json is deliberately compact; probabilities are stored as fractions, not percentages. "
              "No source/generated token arrays or per-pass traces are saved unless --save-details is requested.", "",
              f"Generation {report['generation_elapsed_seconds']:.1f}s; total {report['elapsed_seconds']:.1f}s. "
              f"Peak GPU allocation {report.get('peak_gpu_allocated_gib', 0):.2f} GiB.", ""]
    with open(directory/"report.md", "x") as handle:
        handle.write("\n".join(lines))
    with open(directory/"summary.csv", "x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compact["results"][0]))
        writer.writeheader()
        writer.writerows(compact["results"])
