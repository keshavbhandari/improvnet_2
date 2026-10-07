"""Encode one MIDI, save its hierarchical representation, decode and compare."""

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import tempfile

from .hierarchical import (
    FORMAT_VERSION, EncodedPiece, GridConfig, HierarchicalTokenizer,
    normalize_roll_notes, read_performance,
)


def compare_notes(original, reconstructed):
    """Compare pitch/velocity/program and sorted timings without assuming track IDs.

    Left/right-hand tracks on the selected piano channel become one target track.
    Onset order can change only within the quantization tolerance. Timing errors
    are undefined when identity counts differ: zipping a group after a discarded
    attack would incorrectly compare different notes and report huge errors.
    """
    def grouped(notes):
        result = defaultdict(list)
        for n in notes:
            result[n.channel, n.program, n.pitch, n.velocity].append((n.start, n.end))
        return {k: sorted(v) for k, v in result.items()}
    a, b = grouped(original), grouped(reconstructed)
    same = a.keys() == b.keys() and all(len(a[k]) == len(b[k]) for k in a)
    errors = [(abs(x[0]-y[0]), abs(x[1]-y[1])) for k in a for x, y in zip(a[k], b[k])] if same else []
    return dict(notes_before=len(original), notes_after=len(reconstructed), pitch_velocity_instrument_counts_match=same,
                max_onset_error_ms=round(1000 * max((x[0] for x in errors), default=0), 6) if same else None,
                max_offset_error_ms=round(1000 * max((x[1] for x in errors), default=0), 6) if same else None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, help="New/empty directory; default is a unique scratch folder")
    p.add_argument("--coarse-ms", type=int, default=1000)
    p.add_argument("--fine-ms", type=int, default=10)
    p.add_argument("--piano-channel", type=int, help="Zero-based MIDI channel; default is most note-rich piano channel")
    p.add_argument("--target-channel", type=int, help="Override piano priority with any MIDI channel (0–15; drums=9)")
    p.add_argument("--target-program", type=int, help="Optional zero-based GM program on --target-channel")
    p.add_argument("--seed", type=int, help="Reproduce random instrument selection when no piano exists")
    p.add_argument("--piano-programs", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--no-pedal", action="store_true", help="Keep CC64 in the frozen sidecar rather than the piano canvas")
    args = p.parse_args(argv)
    config = GridConfig(coarse_ms=args.coarse_ms, fine_ms=args.fine_ms,
                        piano_programs=tuple(args.piano_programs), include_pedal=not args.no_pedal)
    if args.output_dir is not None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error("Use a new/empty output directory; existing files are never overwritten")
    tokenizer = HierarchicalTokenizer(config)
    piece = tokenizer.encode(args.input, piano_channel=args.piano_channel,
                             target_channel=args.target_channel, target_program=args.target_program, seed=args.seed)
    if args.output_dir is None:
        root = Path("/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/hierarchical_tokenizer")
        root.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="roundtrip-", dir=root))
    else:
        output = args.output_dir
        output.mkdir(parents=True, exist_ok=True)
    piece.save(output / "encoded.npz")
    piece = EncodedPiece.load(output / "encoded.npz")
    reconstructed = tokenizer.decode(piece, strict=True)
    reconstructed.save(output / "reconstructed.mid")
    target_midi = tokenizer.decode(piece, target_only=True, strict=True)
    target_midi.save(output / "target_only.mid")
    if piece.metadata["target_is_piano"]:
        target_midi.save(output / "piano_only.mid")  # Existing piano round-trip workflows.
    source = read_performance(args.input)
    original_notes = source["notes"]
    decoded_notes = read_performance(output / "reconstructed.mid")["notes"]
    target = [n for n in original_notes if n.channel == piece.metadata["target_channel"]
              and n.program == piece.metadata["target_program"]]
    target_orders = {n.order for n in target}
    expected, _ = normalize_roll_notes(target, config)
    for n in original_notes:
        if n.order not in target_orders:
            start = config.frame(n.start)
            end = max(start + 1, config.frame(n.end))
            expected.append(replace(n, start=start * config.fine_ms / 1000,
                                    end=end * config.fine_ms / 1000))
    normalized_comparison = compare_notes(expected, decoded_notes)
    # Only output-tick rounding remains after comparing to the explicitly
    # normalized, quantized reference; overlap shortening can exceed 5 ms.
    tolerance_ms = max(v for _, v in source["tempo_map"]) / reconstructed.ticks_per_beat / 2000 + 1e-5
    matches = (normalized_comparison["pitch_velocity_instrument_counts_match"] and
               normalized_comparison["max_onset_error_ms"] <= tolerance_ms and
               normalized_comparison["max_offset_error_ms"] <= tolerance_ms)
    summary = dict(format=FORMAT_VERSION, input=str(args.input.resolve()),
                   config=dict(coarse_ms=config.coarse_ms, fine_ms=config.fine_ms),
                   target_channel=piece.metadata["target_channel"], target_program=piece.metadata["target_program"],
                   target_kind=piece.metadata["target_kind"], target_selection=piece.metadata["target_selection"],
                   selection_seed=args.seed, pedal_enabled=piece.metadata["pedal_enabled"],
                   source_target_notes=piece.metadata["source_target_notes"], target_notes=piece.metadata["target_notes"],
                   accompaniment_notes=len(piece.events), bins=len(piece.velocity), frames=piece.n_frames,
                   velocity_shape=list(piece.velocity.shape), onset_shape=list(piece.onset.shape),
                   pedal_shape=list(piece.pedal.shape), normalization=piece.metadata["normalization"],
                   skipped_nonpositive_notes=piece.metadata["skipped_nonpositive_notes"],
                   context_short_notes_extended=piece.metadata["context_short_notes_extended"],
                   comparison_to_source=compare_notes(original_notes, decoded_notes),
                   comparison_to_normalized_reference=normalized_comparison,
                   normalized_reference_matches=matches)
    with (output / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")
    if not matches:
        raise RuntimeError("Round trip differs from the documented normalized reference; inspect summary")
    return summary


if __name__ == "__main__":
    main()
