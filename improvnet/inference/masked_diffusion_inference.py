"""Style transfer using timing, pitch/velocity or complete-note mask edits."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from improvnet.model import masked_diffusion_config as defaults
from improvnet.model.masked_diffusion_model import MaskedDiffusionConfig, MaskedDiffusionModel
from improvnet.utils.ar_utils import ProcessData
from improvnet.utils.masked_diffusion import prepare_canvas, denoise
from improvnet.utils.masked_diffusion_data import note_units, split_windows, rhythm_features, is_note, RHYTHM_DIM


def load_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "masked_diffusion_v1":
        raise ValueError("Expected a masked diffusion checkpoint")
    model = MaskedDiffusionModel(MaskedDiffusionConfig(**state["model_config"]))
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to(device).eval(), state


@torch.inference_mode()
def transfer(model, processor, source_tokens, *, genre="jazz", rhythm_tokens=None,
             strategy="timing", percentage=.5, elasticity=0., steps=32,
             temperature=1., top_k=50, seed=42, rhythm_enabled=True):
    units = note_units(processor.normalize_token_sequence(source_tokens))
    if not any(is_note(unit) for unit in units):
        raise ValueError("Source contains no complete notes")
    if genre not in processor.genres:
        raise ValueError(f"Unknown genre {genre!r}")
    if not rhythm_enabled and rhythm_tokens is not None:
        raise ValueError("Cannot combine a rhythm reference with disabled rhythm guidance")
    rhythm = (rhythm_features(rhythm_tokens if rhythm_tokens is not None else source_tokens, processor.tokenizer)
              if rhythm_enabled else torch.zeros(RHYTHM_DIM))
    multi_hot = processor.get_instrument_multihot(source_tokens)
    windows = split_windows(units, model.config.seq_len, elasticity)
    output, reports = [], []
    for index, window in enumerate(windows):
        canvas = prepare_canvas(window, processor, strategy=strategy, percentage=percentage,
                                elasticity=elasticity, seed=seed + index)
        rebuilt, report = denoise(model, canvas, processor, genre=processor.get_genre_id(genre),
            rhythm=rhythm, multi_hot=multi_hot, elasticity=elasticity, steps=steps,
            temperature=temperature, top_k=top_k, seed=seed + index)
        output.extend(rebuilt); reports.append(report)
    # Verify complete grammar after blank removal, before the permissive MIDI decoder.
    note_units(output)
    return output, dict(strategy=strategy, mask_percentage=percentage, elasticity=elasticity,
        genre=genre, rhythm_source=("reference" if rhythm_tokens is not None else "source") if rhythm_enabled else "none",
        rhythm_features=rhythm.tolist(), source_notes=sum(is_note(u) for u in units),
        output_notes=sum(is_note(u) for u in note_units(output)),
        blank_slots=sum(r["blank_slots"] for r in reports),
        added_notes=sum(r["added_notes"] for r in reports),
        removed_notes=sum(r["removed_notes"] for r in reports), windows=reports)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(defaults.SAVE_DIR) / "best_model.pt")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--genre", choices=defaults.GENRES, default="jazz")
    rhythm_options = parser.add_mutually_exclusive_group()
    rhythm_options.add_argument("--rhythm-reference", type=Path, help="Arbitrary donor MIDI; only abstract timing statistics are used")
    rhythm_options.add_argument("--no-rhythm-guidance", action="store_true",
                                help="Disable rhythm guidance using the all-zero condition seen during training dropout")
    parser.add_argument("--strategy", choices=("timing", "pitch-velocity", "notes"), default="timing")
    parser.add_argument("--mask-percentage", type=float, default=.5, help="Fraction in [0,1] of existing notes to edit")
    parser.add_argument("--elasticity", type=float, default=0., help="Extra whole-note slots per source note, in [0,1]")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    model, state = load_model(args.model, args.device)
    processor = ProcessData()
    processor.genres = state["train_config"]["GENRES"]
    source = processor.midi_to_tokens(processor.read_midi(str(args.input)))
    donor = processor.midi_to_tokens(processor.read_midi(str(args.rhythm_reference))) if args.rhythm_reference else None
    tokens, report = transfer(model, processor, source, genre=args.genre, rhythm_tokens=donor,
        strategy=args.strategy, percentage=args.mask_percentage, elasticity=args.elasticity,
        steps=args.steps, temperature=args.temperature, top_k=args.top_k, seed=args.seed,
        rhythm_enabled=not args.no_rhythm_guidance)
    if args.output.resolve() in {args.input.resolve(), args.rhythm_reference.resolve() if args.rhythm_reference else args.input.resolve()}:
        raise ValueError("Output must differ from source and rhythm reference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processor.tokens_to_midi(tokens).save(str(args.output))
    report.update(step=state["step"], model=str(args.model), source=str(args.input),
                  rhythm_reference=str(args.rhythm_reference) if args.rhythm_reference else None,
                  model_config=asdict(model.config))
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}: {report['source_notes']} → {report['output_notes']} notes")


if __name__ == "__main__":
    main()
