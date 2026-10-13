# ImprovNet

Symbolic music generation and genre style transfer with an autoregressive (AR)
model and a bidirectional masked diffusion model. Independent MIDI
syntax, genre, continuity and pitch-preservation metrics support evaluation.

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
`improvnet/model/ar_config.py`, `masked_diffusion_config.py`, and each
metric's config file. Adjust them before launching; batch sizes are **per GPU**.

## Training

Single-node examples below use four GPUs; use `--nproc_per_node=1` for debugging.
These commands start or resume training, so run them only on available GPUs.

```bash
# Autoregressive model
torchrun --standalone --nproc_per_node=4 -m improvnet.model.ar_train

# Masked diffusion, trained from scratch on the AR tokenizer
torchrun --standalone --nproc_per_node=4 -m improvnet.model.masked_diffusion_train

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
sbatch ../run_scripts/improvnet/masked_multi_node_launch.sh
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

`max_tokens` includes the prompt. AR inference uses underscore flags; diffusion
evaluators use hyphenated flags. Outputs stop when the model emits `<E>`.

## Masked diffusion style transfer

Use a trained masked-diffusion checkpoint with the existing AR tokenizer:

```bash
python -m improvnet.inference.masked_diffusion_inference \
  --model "$IMPROVNET_ARTIFACTS/masked_diffusion_v1/best_model.pt" \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --output "$IMPROVNET_ARTIFACTS/masked_jazz.mid" \
  --genre jazz --strategy timing --mask-percentage .5 --elasticity .2 \
  --rhythm-reference /path/to/jazz_or_blues.mid --steps 32
```

`timing` edits onset/duration; `pitch-velocity` edits the combined pitch/velocity
attribute. `notes` edits complete notes and permits bounded deletion. Fractions
are in [0,1]. Elasticity adds up to that fraction of extra note slots; each slot
becomes a complete note or four blanks that are removed. At elasticity zero,
existing-note count stays fixed. A JSON report accompanies the MIDI.

See [masked diffusion](docs/masked_diffusion.md) for the masking objective,
rhythm statistics, oversampling and training details. No training has been started.

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
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_masked_diffusion.py
```

More: [masked diffusion](docs/masked_diffusion.md) and
[metrics](improvnet/metrics/README.md).
Legacy masked TwoTower/FSQ experiments are under `improvnet/archive`.
