# ImprovNet

On branch `hierarchical-pianoroll-tokenizer`, the new representation is a
hierarchical instrument roll (piano-first, with other instruments/drums supported;
no overlap lanes), plus aligned
accompaniment events. Same-pitch overlaps/collisions are normalized and reported.
See the concise
[design and round-trip guide](docs/hierarchical_tokenizer.md). Existing training
commands below still use the archived event tokenizer through compatibility imports.

```bash
python -m improvnet.tokenizer.roundtrip \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid
```

Symbolic music generation and genre style transfer with an autoregressive (AR)
model and a new hierarchical instrument-roll pipeline. Independent MIDI syntax,
genre, continuity and pitch-preservation metrics support evaluation. The old generative experiments have been removed in favor of latent flow matching.

## Setup

Run commands from the repository root, with a CUDA-compatible PyTorch environment.
On Jupiter, use the existing environment:

```bash
source /e/project1/e-dev-2026d09-047/software/miniforge3/etc/profile.d/conda.sh
conda activate improvnet
cd /e/home/jusers/bhandari1/jupiter/improvnet_2
pip install -e .
export IMPROVNET_ARTIFACTS=/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts
```

Data paths, batch sizes, run directories and training settings live in
`improvnet/model/ar_config.py`, `local_vae_config.py`, and each
metric's config file. Adjust them before launching; batch sizes are **per GPU**.

## Local VAE for the hierarchical roll

The new branch also has an unconditional local VAE: each 1-second patch contains
all 100 fine frames, compressed to a spatial latent grid. It trains directly from
indexed MIDI, not the old JSONL. Paths/settings are in `local_vae_config.py`.
The 176-channel version trains on clean reconstruction, conservative note-event
corruption and latent jitter. Validation jointly tunes activity/onset thresholds;
the 96-channel baseline's checkpoints and logs are retained for comparison.

```bash
# Index only, no training (the full MIDI index has already been prepared).
python -m improvnet.model.local_vae_train --build-index

# Start/resume on one node with four GPUs.
torchrun --standalone --nproc_per_node=4 -m improvnet.model.local_vae_train

# Compare all VAE runs using one shared log directory.
tensorboard --logdir /e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/vae/tensorboard

# Reconstruct one MIDI; checkpoint defaults to the configured VAE run.
python -m improvnet.inference.local_vae_inference \
  --input improvnet/inference/examples/bach_846.mid

# Held-out reconstruction/latent checks, 100 files and 8 bins per file.
python -m improvnet.inference.local_vae_evaluate --num-files 100 --bins-per-file 8

# Focused CPU/synthetic tests.
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_local_vae*.py'
```

See [tensors, architecture, genre metadata and debug commands](docs/local_vae.md).
ARIA genres are available; stage 2 now uses GigaMIDI's supplied curated-genre CSV
through a separate labelled index. Missing genres do not block unconditional VAE
training. The original VAE index is unchanged. Retraining its encoder requires
a fresh stage-2 run and new latent calibration, even if the latent shape matches.

## Stage-2 latent flow matching

Frozen-VAE continuous flow matching with soft per-second pitch guidance, genre/instrument
conditioning, clean history, and up to three full or rhythm-only accompaniment parts.
Settings are in `latent_flow_config.py`. See the
[training, accompaniment and diagnostic inference guide](docs/latent_flow.md).

```bash
# Read-only forward check; does NOT start training.
python -m improvnet.model.latent_flow_train --check

# Start/resume only when ready.
torchrun --standalone --nproc_per_node=4 -m improvnet.model.latent_flow_train

# Evaluate a saved stage-2 checkpoint; outputs go to a fresh scratch directory.
python -m improvnet.inference.latent_flow_inference \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --genre jazz --accompaniment-mode rhythm --strength 0.8 --passes 48

# Separate the VAE reconstruction floor from stage-2 reconstruction errors.
python -m improvnet.inference.latent_flow_inference \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --task reconstruct --genre classical --strength 0.2 --passes 48
```

## Training

Single-node examples below use four GPUs; use `--nproc_per_node=1` for debugging.
These commands start or resume training, so run them only on available GPUs.

```bash
# Autoregressive model
torchrun --standalone --nproc_per_node=4 -m improvnet.model.ar_train

# Genre classifier and contrastive continuity encoder
torchrun --standalone --nproc_per_node=4 -m improvnet.metrics.genre_classifier.genre_classifier_train
torchrun --standalone --nproc_per_node=4 -m improvnet.metrics.continuity.continuity_train
```

Before the first metric-training run, create its dataset manifest if absent:

```bash
python -m improvnet.metrics.genre_classifier.audit
python -m improvnet.metrics.continuity.audit
```

Audits refuse to overwrite existing manifests. For changed data, pass a fresh
`--output-prefix` and configure the matching `MANIFEST` path.
Training resumes from each configured `latest_checkpoint.pt` where supported.

Two-node Jupiter launchers are in the sibling `run_scripts/improvnet` directory:

```bash
sbatch ../run_scripts/improvnet/ar_multi_node_launch.sh
sbatch ../run_scripts/improvnet/genre_multi_node_launch.sh
sbatch ../run_scripts/improvnet/continuity_multi_node_launch.sh
```

Check launcher resources and paths before submitting. View training metrics with
`tensorboard --logdir "$IMPROVNET_ARTIFACTS"`.

## AR inference

```bash
python -m improvnet.inference.ar_inference \
  --model "$IMPROVNET_ARTIFACTS/ar_context_split_instrument/latest_checkpoint.pt" \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --output "$IMPROVNET_ARTIFACTS/ar_jazz.mid" \
  --genre jazz --prompt_len 256 --max_tokens 2048
```

`max_tokens` includes the prompt. AR inference uses underscore flags; flow matching
evaluators use hyphenated flags. Outputs stop when the model emits `<E>`.

## Metrics

```bash
# Genre probabilities (requires a trained classifier)
python -m improvnet.metrics.genre_classifier.scoring \
  --checkpoint "$IMPROVNET_ARTIFACTS/genre_classifier/encoder_run/best_checkpoint.pt" \
  --input "$IMPROVNET_ARTIFACTS/ar_jazz.mid" --genre jazz --device cuda

# Pitch identity, short motifs and aligned self-similarity; also saves listening MIDIs
python -m improvnet.metrics.pitch_similarity.demo --mode modified --modification mixed --plot

# Compare two explicit MIDI excerpts; supply the canvas, not prompt-plus-canvas
python -m improvnet.metrics.pitch_similarity.demo --mode compare \
  --input /path/to/original_canvas.mid --generated /path/to/generated_canvas.mid --plot
```

MIDI syntax and continuity also have simple Python APIs:

```python
from improvnet.tokenizer.absolute import AbsTokenizer
from improvnet.tokenizer.midi import MidiDict
from improvnet.metrics import MidiSyntaxMetric
from improvnet.metrics.continuity.scoring import ContinuityMetric

tokenizer = AbsTokenizer()
previous = tokenizer.tokenize(MidiDict.from_midi("/path/to/previous.mid"))
current = tokenizer.tokenize(MidiDict.from_midi("/path/to/generated.mid"))
print(MidiSyntaxMetric(tokenizer).evaluate_sequence(current))
metric = ContinuityMetric.from_checkpoint("/path/to/continuity/best_checkpoint.pt", device="cuda")
print(metric.score_boundary(previous, current, window_size=256))
```

Syntax checks intended token order **before** MIDI decoding; checking a saved
MIDI alone cannot detect token errors already discarded by a permissive decoder.
For open canvases, `prefix_valid` permits a correctly ordered unfinished tail;
`valid` still requires actual MIDI compilation. Continuity is learned cosine
compatibility, not a probability. Genre and pitch scores do not guarantee good
music—listen to the saved examples too.

## Tests and details

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_midi_syntax.py
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_local_vae.py
```

More: [local VAE](docs/local_vae.md), [metrics](improvnet/metrics/README.md),
and [latent flow matching](docs/latent_flow.md).
Legacy masked TwoTower/FSQ experiments are under `improvnet/archive`.
