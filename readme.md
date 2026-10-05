# ImprovNet

Symbolic music generation and genre style transfer with an autoregressive (AR)
model and a shared-backbone uniform discrete diffusion model. Independent MIDI
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
`improvnet/model/ar_config.py`, `uniform_diffusion_train_config.py`, and each
metric's config file. Adjust them before launching; batch sizes are **per GPU**.

## Training

Single-node examples below use four GPUs; use `--nproc_per_node=1` for debugging.
These commands start or resume training, so run them only on available GPUs.

```bash
# Autoregressive model
torchrun --standalone --nproc_per_node=4 -m improvnet.model.ar_train

# Uniform diffusion SFT, initialized from the configured AR checkpoint
torchrun --standalone --nproc_per_node=4 -m improvnet.model.uniform_diffusion_train

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
sbatch ../run_scripts/improvnet/uniform_multi_node_launch.sh
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

## Uniform diffusion style transfer

Keep the source event scaffold; refine timing first, then pitch/velocity.
Instruments, structure and metadata stay fixed. The first block is promptless;
later blocks use generated history. Source/output MIDI and all four metrics are
saved in a new scratch directory. No training is started.

```bash
# Twenty held-out classical pieces, conditioned on jazz, up to eight blocks
python -m improvnet.inference.uniform_diffusion_evaluate \
  --test-examples 20 --steps 48 --max-blocks 8 --noise-levels 1 .8 .6 .4 .2

# One piece, with the same complete metric report
python -m improvnet.inference.uniform_diffusion_evaluate \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --genre jazz --steps 48 --max-blocks 8 --noise-levels .8 .4 .2

# Timing-only Clair de lune → blues; pitches/velocities stay fixed
python -m improvnet.inference.uniform_diffusion_evaluate \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --genre blues --strategy timing --metrics genre --steps 16 32 48 \
  --max-blocks 8 --noise-levels 1 .8 .6 .4 .2 --batch-size 1
```

`--steps` is **per stage**, limited to 48 (96 total passes/block); adaptive
stopping can use fewer. Noise applies only to editable slots in each stage.
Default `--sampler flow` retains low-noise source information; `entropy` is an
alternative that repeatedly re-noises unselected editable slots. Timing slots
cannot become notes; note slots jointly sample pitch/velocity, including drums.
The last source block closes at an event boundary before generation. This is
constrained reconstruction, not novel prompt completion or event insertion.
Reports use one compact JSON row per result; large token/trace dumps require
explicit `--save-details`. `--steps` also accepts multiple budgets for a sweep.

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
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_uniform_diffusion*py'
```

More: [diffusion architecture](docs/uniform_diffusion_architecture.md),
[training](docs/uniform_diffusion_training.md),
[evaluation](docs/uniform_diffusion_evaluation.md), and
[metrics](improvnet/metrics/README.md).
Legacy masked TwoTower/FSQ experiments are under `improvnet/archive`.
