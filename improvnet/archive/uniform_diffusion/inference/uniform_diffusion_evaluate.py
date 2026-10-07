"""Evaluate promptless, slot-constrained style transfer over up to eight blocks.

Run with --input for one MIDI, or select classical pieces from the checkpoint's
held-out test split. Save source/output MIDI and compact metric scores to scratch.
Existing checkpoints and outputs are never overwritten; no training is started.
"""

import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch

from improvnet.archive.uniform_diffusion.inference.uniform_diffusion_inference import SamplerConfig, closed_source, generate_batch
from improvnet.archive.uniform_diffusion.inference.uniform_diffusion_evaluation_metrics import (
    inspect_syntax, release_gpu, score_metrics, select_test_examples, syntax_dict, write_json, write_report,
)
from improvnet.metrics.midi_syntax import MidiSyntaxMetric

ARTIFACTS = Path("/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts")


def load_model(path, device, *, headroom_gib=2.):
    """Meta/mmap loading; bounded allocator on the shared debugging GPU."""
    if device.type == "cuda":
        index = device.index or 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        gpu = visible[index] if index < len(visible) and visible[index] else str(index)
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", f"--id={gpu}"],
                                    capture_output=True, text=True, check=True, timeout=10)
            free = float(result.stdout.strip())/1024
        except (OSError, ValueError, subprocess.SubprocessError):
            free = None
        if free is not None and free < headroom_gib+4.7:
            raise RuntimeError("Not enough free shared-GPU memory; inference was not started")
    from improvnet.model.ar_model import RotaryEmbedding
    from improvnet.archive.uniform_diffusion.model.uniform_diffusion_model import UniformDiffusionModel
    from improvnet.archive.uniform_diffusion.model.uniform_diffusion_config import UniformDiffusionConfig
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if state.get("format") != "uniform_diffusion_sft_v1":
        raise ValueError("Expected uniform diffusion SFT checkpoint")
    with torch.device("meta"):
        model = UniformDiffusionModel(UniformDiffusionConfig(**state["model_config"]))
    model.load_state_dict({k.removeprefix("module."): v for k, v in state["model_state_dict"].items()}, strict=True, assign=True)
    frequencies = []
    for layer in model.layers:
        freq = RotaryEmbedding(layer.attn.head_dim).inv_freq
        layer.attn.rope.inv_freq = freq
        frequencies.append(freq)
    keys = ("JSONL_FILES", "MIDI_FILEPATH_EXCLUDE_SUBSTRINGS", "DATA_SPLIT_SEED", "DATA_SPLIT_RATIOS", "PROMPT_MAX")
    metadata = dict(step=state.get("step"), best_val=state.get("best_val"), model_config=state["model_config"],
                    data_config={k: state.get("train_config", {})[k] for k in keys if k in state.get("train_config", {})})
    del state
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        gib = 1024**3
        required = sum(p.numel() for p in model.parameters())*2 + 2*gib
        budget = free - headroom_gib*gib
        if budget < required:
            raise RuntimeError("Not enough memory for weights and activations plus shared-GPU headroom")
        limit = min(budget, required+gib)
        torch.cuda.set_per_process_memory_fraction(limit/total, device.index or 0)
        print(f"Shared GPU: {free/gib:.1f} GiB free; allocator limit {limit/gib:.1f} GiB", flush=True)
    model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
    for layer, freq in zip(model.layers, frequencies):
        layer.attn.rope.inv_freq = freq.to(device)
    model.requires_grad_(False)
    return model, metadata


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, help="Single MIDI instead of the held-out classical cohort")
    p.add_argument("--model", type=Path, default=ARTIFACTS/"uniform_diffusion_v1/best_model.pt")
    p.add_argument("--output-dir", type=Path, help="New/empty output directory; default is unique scratch directory")
    p.add_argument("--genre", choices=("jazz", "blues", "classical"), default="jazz")
    p.add_argument("--strategy", choices=("timing", "timing-pitch"), default="timing-pitch",
                   help="timing freezes pitch/velocity too; timing-pitch also runs note refinement")
    p.add_argument("--source-genre", default="classical", help="Source label for --input only")
    p.add_argument("--test-examples", type=int, default=20)
    p.add_argument("--noise-levels", type=float, nargs="+", default=[1., .8, .6, .4, .2])
    p.add_argument("--steps", type=int, nargs="+", default=[48], help="Pass budgets PER STAGE; e.g. 16 32 48")
    p.add_argument("--max-blocks", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sampler", choices=("flow", "entropy"), default="flow")
    p.add_argument("--genre-checkpoint", type=Path, default=ARTIFACTS/"genre_classifier/encoder_run/best_checkpoint.pt")
    p.add_argument("--continuity-checkpoint", type=Path, default=ARTIFACTS/"continuity/encoder_run/best_checkpoint.pt")
    p.add_argument("--metrics", choices=("all", "genre"), default="all", help="genre scores only jazz/blues plus syntax checks")
    p.add_argument("--save-details", action="store_true", help="Opt in to large raw token/trace dumps; default saves only compact scores")
    p.add_argument("--no-adaptive-stopping", action="store_true")
    p.add_argument("--no-self-conditioning", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--headroom-gib", type=float, default=2.)
    args = p.parse_args(argv)
    if any(not 1 <= steps <= 48 for steps in args.steps) or not 1 <= args.max_blocks <= 8 or not 1 <= args.batch_size <= 4 or args.test_examples < 1:
        p.error("Need 1–48 steps per stage, 1–8 blocks, 1–4 inference rows, and positive examples")
    if len(set(args.noise_levels)) != len(args.noise_levels) or any(not math.isfinite(n) or not 0 <= n <= 1 for n in args.noise_levels):
        p.error("Noise levels must be distinct and in [0,1]")
    if len(set(args.steps)) != len(args.steps):
        p.error("Pass budgets must be distinct")
    if not math.isfinite(args.headroom_gib) or args.headroom_gib < 2:
        p.error("Preserve at least 2 GiB shared-GPU headroom")
    return args


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    paths = [args.model, args.genre_checkpoint] + ([args.continuity_checkpoint] if args.metrics == "all" else [])
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir is not None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Outputs are never overwritten; choose a new/empty directory")
    device = torch.device(args.device)
    model, checkpoint = load_model(args.model, device, headroom_gib=args.headroom_gib)
    # ProcessData's AR config probes CUDA; only import after the memory gate.
    from improvnet.utils.ar_utils import ProcessData
    processor = ProcessData()
    if (model.config.vocab_size != processor.tokenizer.vocab_size or model.config.num_genres != len(processor.genres)
            or model.config.num_instruments != len(processor.INSTRUMENT_CLASSES) or model.config.canvas_size != 256):
        raise ValueError("Checkpoint vocabulary/conditioning/canvas size mismatch")
    if args.input:
        tokens = processor.normalize_token_sequence(processor.midi_to_tokens(processor.read_midi(str(args.input))))
        tokens = closed_source(tokens, args.max_blocks*256)
        grade = MidiSyntaxMetric(processor.tokenizer).evaluate_canvas(tokens, allow_incomplete_tail=False)
        if not grade.valid:
            raise ValueError(f"Source scaffold is invalid: {grade.issue}")
        examples = [dict(example_id="example_00", sample_index=0, midi_filepath=str(args.input.resolve()),
                         source_genre=args.source_genre, tokens=tokens, target_start=0, source_syntax=syntax_dict(grade))]
        selection = dict(input=str(args.input.resolve()), held_out_test=False)
    else:
        examples, selection = select_test_examples(checkpoint["data_config"], processor, count=args.test_examples,
                                                   max_blocks=args.max_blocks, seed=args.seed)
    config = SamplerConfig(steps=args.steps[0], sampler=args.sampler, adaptive_stopping=not args.no_adaptive_stopping,
                           self_conditioning=not args.no_self_conditioning)
    root = ARTIFACTS/"uniform_diffusion_style_transfer"
    if args.output_dir is None:
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="transfer-", dir=root))
    else:
        directory = args.output_dir
        directory.mkdir(parents=True, exist_ok=True)
    report = dict(format="uniform_diffusion_constrained_transfer_v1", checkpoint=checkpoint, checkpoint_path=str(args.model.resolve()),
                  device=str(device), seed=args.seed, target_genre=args.genre, sampler_config=asdict(config), selection=selection,
                  max_blocks=args.max_blocks, noise_levels=args.noise_levels, step_budgets=args.steps,
                  strategy=args.strategy, metrics_requested=args.metrics, examples=examples, results=[])
    if args.save_details:
        write_json(directory/"configuration.json", {k: v for k, v in report.items() if k != "results"})
    for e in examples:
        folder = directory/e["example_id"]
        folder.mkdir()
        processor.tokens_to_midi(e["tokens"]).save(str(folder/"reference.mid"))
    started = time.monotonic()
    print(f"Transfer: {len(examples)} pieces; {args.steps} passes/stage; output={directory}", flush=True)
    for noise in args.noise_levels:
        for steps in args.steps:
            config = SamplerConfig(steps=steps, sampler=args.sampler, adaptive_stopping=not args.no_adaptive_stopping,
                                   self_conditioning=not args.no_self_conditioning)
            for start in range(0, len(examples), args.batch_size):
                batch = examples[start:start+args.batch_size]
                print(f"Noise={noise:.0%}; steps={steps}; pieces={[e['example_id'] for e in batch]}", flush=True)
                report["results"].extend(generate_batch(model, processor, batch, noise=noise, config=config,
                    max_blocks=args.max_blocks, seed=args.seed, device=device, genre=args.genre, strategy=args.strategy))
    report["generation_elapsed_seconds"] = time.monotonic()-started
    if device.type == "cuda":
        report["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated(device)/1024**3
    del model
    release_gpu()
    inspect_syntax(report, processor)
    for row in report["results"]:
        row["output_midi"] = None
        if row["syntax"]["midi_compiles"] is True:
            path = directory/row["example_id"]/f"{args.genre}_t{row['initial_noise']!r}_steps{row['max_steps']}.mid"
            processor.tokens_to_midi(row["generated_tokens"]).save(str(path))
            row["output_midi"] = str(path)
    # Save raw generation first so an interrupted learned-metric stage is recoverable.
    if args.save_details:
        write_json(directory/"generation.json", report)
    print(f"Generation done; scoring {args.metrics} metrics", flush=True)
    score_metrics(report, args, processor, device)
    report["elapsed_seconds"] = time.monotonic()-started
    write_report(directory, report, save_details=args.save_details)
    print(f"Saved report: {directory/'report.md'}", flush=True)
    return report


if __name__ == "__main__":
    main()
