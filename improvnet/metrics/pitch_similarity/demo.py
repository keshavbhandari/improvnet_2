"""Audible, reproducible comparison of two excerpts, with score breakdowns."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import tempfile

import numpy as np

from .metric import PitchSimilarityConfig, PitchSimilarityMetric
from .midi import clip_segment, modify_segment, pitched_sequence, random_segment_start, read_midi, save_midi


DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "inference/examples/debussy-clair-de-lune_original.mid"
DEFAULT_OUTPUT_ROOT = Path("/e/home/jusers/bhandari1/jupiter/improvnet_2/improvnet/metrics/pitch_similarity/pitch_similarity_demo")


def run_demo(input_path, output_dir, *, mode="modified", generated_path=None, start=None,
             second_start=None, seconds=12., modification="mixed", probability=.15,
             transpose=7, seed=42, representation="all-notes", allow_transposition=False, plot=False):
    output_dir = Path(output_dir).resolve()
    targets = [output_dir / name for name in ("original.mid", "comparison.mid", "scores.json", "matrices.npz", "matrices.svg")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Demo outputs exist; choose a fresh output directory")
    if mode not in ("modified", "two-segments", "compare"):
        raise ValueError("Invalid demo mode")
    rng = random.Random(seed)
    source = read_midi(input_path)
    start = random_segment_start(source, seconds, rng) if start is None else start
    original = clip_segment(source, start, seconds)
    if mode == "modified":
        comparison = modify_segment(original, modification, rng, probability=probability, transpose=transpose)
        second_start = start
    else:
        if mode == "compare" and generated_path is None:
            raise ValueError("--generated is required for compare mode")
        other = read_midi(generated_path) if mode == "compare" else source
        if second_start is None:
            second_start = 0. if mode == "compare" else random_segment_start(other, seconds, rng, exclude=(start, start + seconds))
        comparison = clip_segment(other, second_start, seconds)
    reference, generated = pitched_sequence(original, representation), pitched_sequence(comparison, representation)
    metric = PitchSimilarityMetric(replace(PitchSimilarityConfig(), allow_global_transposition=allow_transposition))
    result = metric.evaluate_pitches(reference, generated)
    if not comparison:
        raise ValueError("Comparison has no sounding notes for MIDI listening")
    cross = metric.cross_similarity(reference, generated, transpose=result.generated_to_reference_transpose)
    reference_ssm, generated_ssm = metric.cross_similarity(reference, reference), metric.cross_similarity(generated, generated)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_midi(original, targets[0])
    save_midi(comparison, targets[1])
    roundtrip = {"reference": pitched_sequence(read_midi(targets[0]), representation) == reference,
                 "generated": pitched_sequence(read_midi(targets[1]), representation) == generated}
    report = {"input": str(Path(input_path).resolve()), "generated_input": str(Path(generated_path).resolve()) if generated_path else None,
              "mode": mode, "modification": modification if mode == "modified" else None,
              "seed": seed, "start_seconds": start, "comparison_start_seconds": second_start,
              "seconds": seconds, "representation": representation, "config": asdict(metric.config),
              "reference_pitches": reference, "generated_pitches": generated, "scores": result.to_dict(),
              "midi_pitch_roundtrip": roundtrip,
              "listening_note": "Repeated same-pitch/channel overlaps may have shortened note-offs in the MIDI listening copy; scoring uses the full event pitches."}
    with targets[2].open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with targets[3].open("xb") as handle:
        np.savez_compressed(handle, cross=cross, reference_ssm=reference_ssm, generated_ssm=generated_ssm,
                            alignment=np.asarray(result.alignment, dtype=np.int64).reshape(-1, 2))
    if plot:
        from .plot import save_matrix_plot
        save_matrix_plot(targets[4], cross, reference_ssm, generated_ssm, result.alignment)
    print(json.dumps({key: value for key, value in result.to_dict().items() if key != "alignment"}, indent=2, allow_nan=False))
    print(f"Listening excerpts: {targets[0]}\n                    {targets[1]}\nFull diagnostics: {targets[2]}")
    if not all(roundtrip.values()):
        print("Warning: MIDI timing quantization/overlap handling changed the extracted pitch order. See midi_pitch_roundtrip in scores.json.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--generated", help="Comparison MIDI for --mode compare")
    parser.add_argument("--mode", choices=("modified", "two-segments", "compare"), default="modified")
    parser.add_argument("--output-dir", help="Fresh directory; default is a unique scratch artifact folder")
    parser.add_argument("--start-seconds", type=float)
    parser.add_argument("--second-start-seconds", type=float)
    parser.add_argument("--seconds", type=float, default=12.)
    parser.add_argument("--modification", choices=("copy", "timing", "ornament", "substitute", "shuffle", "transpose", "mixed", "unrelated"), default="mixed")
    parser.add_argument("--probability", type=float, default=.15)
    parser.add_argument("--transpose", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representation", choices=("all-notes", "skyline"), default="all-notes")
    parser.add_argument("--allow-transposition", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    if output is None:
        DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        output = tempfile.mkdtemp(prefix="demo-", dir=DEFAULT_OUTPUT_ROOT)
    run_demo(args.input, output, mode=args.mode, generated_path=args.generated,
             start=args.start_seconds, second_start=args.second_start_seconds, seconds=args.seconds,
             modification=args.modification, probability=args.probability, transpose=args.transpose, seed=args.seed,
             representation=args.representation, allow_transposition=args.allow_transposition, plot=args.plot)


if __name__ == "__main__":
    main()
