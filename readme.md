# ImprovNet: melody refinement and harmonization

This branch extends the existing autoregressive transformer with three tasks:
melody corruption/refinement, onset/duration reconstruction, and harmonization.
The model generates complete response sequences: instrument, pitch/velocity,
onset, duration and structural markers. A continuous condition in [0, 1] controls
corruption strength. No beat-position or diffusion-timestep input is needed.

## Setup

Use the existing PyTorch environment, and run from this worktree:

```bash
source /e/project1/e-dev-2026d09-047/software/miniforge3/etc/profile.d/conda.sh
conda activate /e/project1/e-dev-2026d09-047/envs/improvnet
cd /e/home/jusers/bhandari1/jupiter/improvnet_melody
python -m pip install -e .
export IMPROVNET_ARTIFACTS=/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts
```

MusicXML preparation uses `music21`; MIDI preparation uses the existing AR
performance tokenizer. Raw data remain under
`/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/jazz_blues_data`.

## Prepare data once

```bash
python -m improvnet.scripts.melody_prepare
```

This writes `prepared/performances.jsonl`, its byte-offset index and a report.
It refuses to overwrite an existing prepared dataset. For an independent smoke
test, use `--max-files 20 --output /tmp/melody_pilot.jsonl`.
Conversion uses four CPU workers by default; `--workers 0` runs serially.

The preparer reads ARIA's copied jazz/blues selection, Weimar, PiJAMA, Doug
McKenzie and the 213 labelled WikiMT Jazz scores. Other lead sheets can be added
through `curated_scores.jsonl` in the data root, with `path`, `genre`, `title`
and optional `monophonic` fields. Unlabelled scores and DTL1000's metadata-only
files are excluded. Lead sheets teach melody refinement, while timing and
harmonization use performed data. File parsing failures are recorded in the report.

True solo sources retain their melody; polyphonic performances are partitioned
with skyline into selected melody notes and the remaining accompaniment. Files
are deduplicated, and matched titles/full-melody fingerprints are grouped before
splitting. Skyline is a heuristic: inspect the extracted melodies before a long
run. Overlap with the original AR pretraining corpus is not yet audited.

## Fine-tune

Settings are in `improvnet/model/melody_config.py`, using the AR architecture and
optimizer conventions. Start with a short pilot and adjust the **per-GPU** batch
size for the available hardware. The default is batch 2, accumulation 8.
The config sets `N_STEPS = 20000` optimizer updates, with 500 warmup updates and
learning-rate decay over the final 5,000 updates. No step argument is required.
Treat 20,000 as an initial ceiling and select checkpoints using validation and
listening results.

```bash
# One GPU
python -m improvnet.model.melody_train

# Multiple GPUs
torchrun --standalone --nproc_per_node=4 -m improvnet.model.melody_train
```

For the first launch, the trainer loads the configured AR checkpoint weights,
appends three task-token rows, and initializes a strength projection. It starts
a fresh optimizer/schedule. Later launches resume this run's checkpoint,
including its optimizer, schedule, data position and RNG state. Resume requires
unchanged batch size, accumulation, world size and checkpointed task/data settings.
The AR checkpoint itself is never overwritten.

The trainer follows the existing AR loop: DDP, AMP, accumulation, gradient
clipping, TensorBoard, validation and atomic resumable checkpoints. Only response
tokens contribute to loss. Validation reports each task separately.
`CHECKPOINT_REQUEST_FILE` retains the existing Slurm pre-timeout checkpoint flow.
Optional command-line overrides are `--data`, `--save-dir`, and `--steps`.
`--steps` only stops a run early; the schedule horizon remains the config's
`N_STEPS`. The commands above use the configured step count directly.

## Transfer a performance MIDI

After fine-tuning, use the new checkpoint, not the original AR checkpoint:

```bash
python -m improvnet.inference.melody_inference \
  --model "$IMPROVNET_ARTIFACTS/melody_refinement_v1/best_model.pt" \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --output "$IMPROVNET_ARTIFACTS/clair_jazz.mid" \
  --genre jazz --melody-strength .2 --rhythm-strength .5 --passes 1
```

The default pipeline extracts a skyline melody, refines it, changes its timing,
and harmonizes it. `--genre blues` selects blues. `--task refine`, `rhythm` or
`harmonize` runs one stage; `--no-harmonize` skips accompaniment generation.
Use `--melody-strength 0` to preserve melody pitches before testing timing and
harmony. Timing responses keep the note count, instrument and pitch/velocity
unless `--free-timing-pitches` is set. `--temperature 0` selects greedy decoding;
`--seed` makes sampling repeatable on the same hardware/software.

The output is accompanied by `.melody.mid`, `.refine.mid`, `.rhythm.mid` files for
applicable stages and a JSON report. Harmonization generates a complete arrangement
and then reinstates the conditioning melody exactly, replacing near-duplicate
melody attacks, including same-pitch accompaniment restrikes during held melody
notes. This preserves anchors, but does not certify that every additional
generated note is musically appropriate. Listen to repeated-note and held-note
passages and compare the stage outputs.

Strength controls input corruption, not a measured percentage of jazz character.
Timing masks cover 0–100%, including exact endpoints. At 100%, timing is inferred
from pitch and context; it is not uniquely recoverable. Lower strengths and fewer
passes are sensible first tests for preserving recognizable tunes.

## Existing AR and evaluation tools

The original AR config/model/trainer/inference and independent syntax, genre,
continuity and pitch metrics remain available. Their existing paths/configs still
need to match their own artifacts. The unused diffusion and archived latent-model
experiments were removed from this branch.

```bash
torchrun --standalone --nproc_per_node=1 -m improvnet.model.ar_train
python -m improvnet.inference.ar_inference --help
python -m improvnet.metrics.pitch_similarity.demo --mode compare \
  --input /path/to/original.mid --generated /path/to/generated.mid --plot
CUDA_VISIBLE_DEVICES='' python -m pytest tests -q
```

Install test dependencies with `python -m pip install -e '.[test]'` if needed.

See the [training design](docs/melody_refinement_plan.md) and
[metric documentation](improvnet/metrics/README.md). Production fine-tuning and
musical-quality evaluation have not been run; small CPU tests exercise the code.
