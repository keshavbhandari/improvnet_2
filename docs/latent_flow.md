# Stage 2: latent flow matching

The frozen VAE encodes each second into a 512-value vector, stored as `[512,1,1]`.
Stage 2 learns a conditional velocity field on normalized posterior means. It
uses the same indexed MIDI data, pitch augmentation, genre and instrument labels,
soft per-second pitch guidance, accompaniment modes and clean history as before.
No VAE retraining is required. Existing generative-stage weights are incompatible.

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
16-head model, per-GPU batch 128, 30 target seconds and eight history seconds.
The 38 latent tokens per window cross-attend to accompaniment and pitch tokens.
Batch 128 is retained from the user's configuration; its memory/throughput has
not been benchmarked here. Adjust the batch before a new run if necessary.

The VAE is pinned at `vae/vae176_vector512/best_model.flow.pt`. New checkpoints
and events go to `artifacts/latent_flow`; inference goes to
`artifacts/latent_flow_evaluation`. The frozen encoder is SHA-256 checked.

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

- Pitch: a soft 128-value pitch-presence reference per second, including held
  notes. Whole-reference and note-level dropout are retained; outputs are not
  restricted to supplied pitches.
- Genre: all 28 indexed training genres plus the unconditioned ID. The separate
  curated-GigaMIDI genre index is preserved; no labels are guessed.
- Instrument: target program and drum flag. The tokenizer chooses a piano-family
  target first and otherwise another eligible part.
- Accompaniment: up to three parts. Full-note mode 45%, rhythm-only 45%, absent
  10%. Full mode budgets 1024 compound note events across the window. Rhythm mode
  encodes onset count, active-note count and attack velocity at 100ms resolution.
  Missing events/tracks are masked. External references must be aligned manually;
  arbitrary donors are a distribution shift, not paired style-transfer supervision.

## Genre-labelled index

The stage-2 curated-label index remains separate from the VAE index. It preserves
file groups and splits. The genre CSV uses `music_styles_curated`; unlabelled
examples use the unconditioned ID. To publish a new index, configure a fresh
output path and run `python -m improvnet.scripts.update_midi_genres`; it refuses
to overwrite an existing index. The default config module is now `latent_flow_config`.

## Inference

```bash
python -m improvnet.inference.latent_flow_inference \
  --input piece.mid --genre jazz --task transfer \
  --accompaniment-mode rhythm --strength .8 --passes 48 --max-bins 30

python -m improvnet.inference.latent_flow_inference \
  --input piece.mid --task reconstruct --genre classical --strength .2 --passes 48

python -m improvnet.inference.latent_flow_inference \
  --input piece.mid --task continue --prompt-bins 8 --strength 1 --max-bins 60
```

The checkpoint restores window lengths and architecture. EMA is the default;
`--raw-weights` selects the live training weights. Optional `--accompaniment-midi`,
`--reference-offset-seconds` and `--reference-time-scale` supply external
accompaniment. `--help` lists all options. Generated history is retained directly
as latents across windows, and MIDI notes are assembled continuously across bins.
The VAE's saved decoding calibration is used unless explicitly overridden.

Outputs include source, VAE-only reconstruction, generated target, original-source
accompaniment mix, and a JSON report. Compare the VAE-only floor, generated note
statistics, release timing and latent error. Transfer distances include intentional
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
