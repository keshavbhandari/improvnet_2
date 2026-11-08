# Stage 2: latent flow matching

The current VAE encodes each **1.28-second bin** into a 512-value vector, stored as `[512,1,1]`.
Stage 2 learns a conditional velocity field on normalized posterior means. It
uses the same indexed MIDI data, pitch augmentation, genre and instrument labels,
soft per-bin pitch guidance, accompaniment modes and clean history as before.
The new VAE encoder requires fresh flow weights and newly fitted latent statistics.
Existing one-second flow checkpoints remain usable through their saved settings.

## Objective and sampling

For a clean target latent `x1`, independent Gaussian noise `x0`, and a uniformly
sampled time `t` in [0,1], form:

```
xt = (1-t) * x0 + t * x1
velocity_target = x1 - x0
loss = weighted_mean((predicted_velocity - velocity_target)**2)
```

History is always clean at time 1, excluded from the velocity loss, and clamped
after every sampling update. Padding is excluded from encoding, loss and updates.
Latent mean/std statistics are fitted on training targets only and checkpointed.
The training endpoint diagnostic is `xt + (1-t)*predicted_velocity`; its RMSE is
not a full-sampler reconstruction score. Low-noise metrics use t>=0.75; high-noise
metrics use t<0.25.

Sampling uses the explicit Euler ODE update `x += dt * velocity(x,t)`, with
increasing time toward 1. `--strength 1` starts from pure noise. Lower strength
starts at time `1-strength` from the corresponding source/noise interpolation,
allowing conservative edits. `--passes` controls the number of solver steps.
More passes increase compute and do not guarantee better musical output.

Reference: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
This is independent noise/data coupling on a straight path, without minibatch
optimal-transport matching or reflow/distillation.

## Configuration and training

`improvnet/model/latent_flow_config.py` retains the selected 1024-wide, 24-layer,
16-head model, per-GPU batch 128, 120 target bins (153.6 seconds) and 30 history
bins (38.4 seconds). Bin counts are retained; the complete window is 192 seconds.
The 150 latent tokens per window cross-attend to accompaniment and pitch tokens.
Batch size counts whole windows: the float32 piano rolls alone occupy about
2.34 GiB per batch before conditioning, working copies and caches.

The data loader defaults to four workers and one prefetched batch per worker.
The previous 16 workers with PyTorch's default prefetch of two could queue
32 batches (58.6 GiB of rolls alone), with additional shared/pinned copies and
worker allocations. On the shared debug host this exceeded the account's
143 GiB CPU-memory limit and Linux killed the process without a Python traceback.
Free system RAM and free VRAM do not bypass this account limit. Loader controls
can change on resume without changing batch size, data position or optimizer:

```bash
python -m improvnet.model.latent_flow_train --num-workers 2 --prefetch-factor 1
# Lowest loader memory, if needed:
python -m improvnet.model.latent_flow_train --num-workers 0
```

Resume memory-maps checkpoint tensors and releases the extra checkpoint reference
before loading batches. It still checks both checkpoint destinations' formats.
For multiple GPU ranks, budget CPU loader memory across all ranks on each host.
Latest checkpoints are saved every 100 optimizer steps; validation remains every
1000 steps. Both intervals can change on resume without changing optimization.

The VAE is pinned at `vae/vae176_vector512_128f_corrupt25/best_model.flow.pt`.
This snapshot is intentionally absent until a trained VAE is selected and frozen;
flow training must wait for it. Do not use the actively changing `best_model.pt`.
After VAE validation, create the snapshot once (refuses to overwrite an existing file):

```bash
python - <<'PY'
from pathlib import Path
from shutil import copyfileobj
from improvnet.model.local_vae_config import SAVE_DIR
root = Path(SAVE_DIR)
with (root / "best_model.pt").open("rb") as source, (root / "best_model.flow.pt").open("xb") as target:
    copyfileobj(source, target)
PY
```

Checkpoints/events now go to `artifacts/latent_flow/vae176_vector512_128f_corrupt25`;
inference goes to the matching subfolder under `artifacts/latent_flow_evaluation`.
Old VAE and flow files remain intact. The frozen encoder is SHA-256 checked;
incompatible old flow weights cannot silently resume in the new output directory.

```bash
# Read-only integration check; never starts training.
python -m improvnet.model.latent_flow_train --check
# Start/resume only when ready:
python -m improvnet.model.latent_flow_train
# Distributed workflow:
torchrun --standalone --nproc_per_node=4 -m improvnet.model.latent_flow_train
```

AdamW, warmup/cosine LR scheduling, mixed precision, accumulation, gradient
checkpointing, EMA, validation, atomic best/latest saves, and coordinated
checkpoint requests are retained. Resume restores optimizer/scheduler/scaler,
per-rank RNG and data position. Model/data/VAE identity and world size must match.
`--allow-batch-size-change` permits a documented non-exact continuation with a
new batch size. Flow checkpoints use their own format and cannot load old weights.

## Conditions

- Pitch: a soft 128-value pitch-presence reference per 1.28-second bin, including held
  notes. Whole-reference and note-level dropout are retained; outputs are not
  restricted to supplied pitches.
- Genre: all 28 indexed training genres plus the unconditioned ID. The separate
  curated-GigaMIDI genre index is preserved; no labels are guessed.
- Instrument: target program and drum flag. The tokenizer chooses a piano-family
  target first and otherwise another eligible part.
- Accompaniment: up to three parts. Full-note mode 45%, rhythm-only 45%, absent
  10%. Full mode budgets 1024 compound note events across the window. Rhythm mode
  encodes onset count, active-note count and attack velocity in 8-frame (80 ms)
  groups: 16 groups per bin. Ten-frame groups cannot divide the new 128-frame bin.
  Missing events/tracks are masked. External references must be aligned manually;
  arbitrary donors are a distribution shift, not paired style-transfer supervision.

## Genre-labelled index

The stage-2 curated-label index remains separate from the VAE index. It preserves
file groups and splits. The genre CSV uses `music_styles_curated`; unlabelled
examples use the unconditioned ID. To publish a new index, configure a fresh
output path and run `python -m improvnet.scripts.update_midi_genres`; it refuses
to overwrite an existing index. The default config module is now `latent_flow_config`.

## Inference

The CLI automatically selects CUDA when available, otherwise CPU. It uses EMA
weights, the VAE pinned by the flow checkpoint, and that VAE's calibrated
thresholds and held-note assembly. There are no seed, device, threshold,
note-decoder, raw-weight or VAE-override flags. Normal runs use fresh noise.
The current frozen 512-value VAE stores activity **0.90** and onset **0.985**;
these were calibrated on clean and noisy VAE reconstructions, not on flow output.
The resolved thresholds and device are written into `report.json`.

When the frozen VAE omits note-assembly metadata, flow inference looks up a
validation-calibrated profile in `improvnet/inference/latent_flow_note_decoding.json`,
bound to the exact VAE SHA-256. The current 512-value snapshot uses held-note
decoding with continuation threshold **0.85** and a **20 ms** internal-gap allowance,
selected on 64 validation clips. Onset and activity must agree to start a note;
accepted onsets still retrigger held pitches, and state crosses bin boundaries.
Existing explicit VAE note-assembly settings are preserved. An unknown VAE without
such metadata uses held-note decoding with zero gap bridging until calibrated.
The decoding source is recorded in `report.json`. Neither VAE nor flow checkpoint
is rewritten, so frozen-encoder identity checks remain valid.

Use the bundled Clair de Lune MIDI for a roughly 30-second comparison
(24 bins = 30.72 seconds with the new VAE):

```bash
cd /e/home/jusers/bhandari1/jupiter/improvnet_2
midi="/e/home/jusers/bhandari1/jupiter/improvnet_2/improvnet/inference/examples/debussy-clair-de-lune_original.mid"

# Transfer without accompaniment guidance; keep source pitch guidance and jazz genre.
python -m improvnet.inference.latent_flow_inference \
  --input "$midi" --task transfer --genre jazz \
  --accompaniment-mode none --strength .6 --max-bins 24

# Rhythm reference: use Clair de Lune itself as a runnable example.
# Replace this with an aligned jazz MIDI when you have one.
reference="$midi"
python -m improvnet.inference.latent_flow_inference \
  --input "$midi" --task transfer --genre jazz \
  --accompaniment-midi "$reference" --accompaniment-mode rhythm \
  --strength .6 --max-bins 24

# Full reference: conditioning includes pitches, attacks, durations and velocities.
python -m improvnet.inference.latent_flow_inference \
  --input "$midi" --task transfer --genre jazz \
  --accompaniment-midi "$reference" --accompaniment-mode full \
  --strength .6 --max-bins 24

# Reconstruction diagnostic: conservative source/noise interpolation.
python -m improvnet.inference.latent_flow_inference \
  --input "$midi" --task reconstruct --genre classical \
  --accompaniment-mode none --strength .2 --max-bins 24

# Preserve 6 bins (7.68 seconds); generate 24 new bins (30.72 seconds).
python -m improvnet.inference.latent_flow_inference \
  --input "$midi" --task continue --genre jazz \
  --accompaniment-mode none --prompt-bins 6 --max-bins 24
```

`transfer` and `reconstruct` use the same sampler; `reconstruct` adds decoder
stress diagnostics and interprets distances as reconstruction errors. Genre,
strength and references determine musical behavior. Continuation removes target
pitch guidance after the preserved prompt and requires strength 1 (the default).
Lower transfer strength retains more source latent information; at strength 1
the target starts from pure noise, with the source represented by its conditions.

`none` disables accompaniment, not source pitch or genre conditioning. `rhythm`
supplies onset counts, active-note counts and attack velocities; `full` supplies
pitch-bearing note events too. Without `--accompaniment-midi`, both modes use
the source's other parts, excluding its selected target instrument. This Clair
de Lune file has only one piano channel, so it supplies no other parts: use an
explicit reference to obtain rhythm/full guidance. Using itself as the reference
supplies classical timing/notes, not a jazz performance. A separate jazz donor
must be aligned; the script does not automatically synchronize beats or tempos.
The optional `--reference-time-scale` multiplies reference times, and
`--reference-offset-seconds` shifts them on the source clock. Donor conditioning
is not a guarantee of faithful style transfer.

Sampling defaults to 48 solver steps; `--passes 96` requests more. One bin is
1.28 seconds with the new VAE; older checkpoints retain their saved clock.
`--max-bins` counts generated bins, excluding `--prompt-bins`; the default 32 bins
span at most 40.96 seconds with the new VAE. Reports include the saved clock and
actual prompt/generated durations, excluding padding. The flow checkpoint restores its training window and
architecture; longer clips are processed in consecutive windows, with generated
history retained as latents. Use `--checkpoint` to choose a different flow
checkpoint (the default is `latest_checkpoint.pt`; `best_model.pt` is available).
Reference alignment and instrument-selection options are grouped separately
in `--help`.

Each invocation creates a fresh output directory under
the configured `EVALUATION_DIR`
and prints its path. `--output-dir` can select an existing or new directory.
Rerunning into the same directory overwrites the four generated MIDI files and
`report.json`; unrelated files are retained.
Listen to `generated_target.mid` for the result, and `vae_reconstruction.mid`
for the VAE baseline. `generated_source_mix.mid` adds the source's original
accompaniment; the external donor is conditioning only and is not mixed in.

Compare the VAE-only floor, generated note statistics, release timing and latent
error. Transfer distances include intentional
changes. The optional isotropic jitter comparison is not proof of robustness to
structured flow prediction errors. Listen to results before interpreting a lower
training loss as successful classical-to-jazz conversion.

## Verification

```bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_latent_flow*.py'
```

Tests cover path/velocity identities, an exact oracle ODE, preserved history,
source-independent full-noise initialization, padding, vector-VAE integration,
conditioning, checkpoint identities, and resume behavior. Only temporary synthetic
CPU fixtures are optimized in tests; no production training is launched.

## Decoding and sensitivity diagnostics

`python -m improvnet.inference.latent_flow_diagnostics --output-dir /path/to/new/report`
runs inference only. It compares legacy and held decoding on identical decoder
confidences, a frozen-decoder noise sweep scaled by the flow checkpoint's saved
training latent standard deviations, and paired 48/96/192-step Euler runs at
strengths 0.1/0.3/0.5/0.7. Test clips retain their indexed source genre; unlabelled
clips use unconditioned ID 0. Clair de Lune is evaluated separately with classical
and jazz conditioning. All comparisons omit accompaniment to isolate these effects.

The diagnostic tool's RNG controls are internal to reproducible experiments;
the normal inference CLI still has no seed flag. Cached predictions are checked
against weights, selected windows, seeds and decoding settings. The output includes
JSON metrics, exact test-window manifests, confidence/residual caches and paired
listening MIDIs. Extra-note categories distinguish activity-only attacks,
same-pitch extra strikes and attacks with no same-pitch source activity within
50 ms. The last category is a source mismatch and can include intended jazz edits.
Release MAE is measured on onset-matched notes, so it must be read with recall.
Noise-draw repeats and the small sampling-step subset are exploratory evidence,
not independent musical pieces or a final generalization benchmark.
