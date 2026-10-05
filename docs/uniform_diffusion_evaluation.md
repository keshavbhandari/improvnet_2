# Constrained uniform diffusion style transfer

The active evaluator is `improvnet/inference/uniform_diffusion_evaluate.py`.
Sampling is isolated in `uniform_diffusion_inference.py`; held-out selection,
metric loading/scoring and reports are in `uniform_diffusion_evaluation_metrics.py`.
Previous unrestricted continuation/teacher experiments have been retired.

## Run

```bash
python -m improvnet.inference.uniform_diffusion_evaluate \
  --test-examples 20 --genre jazz --steps 48 --max-blocks 8 \
  --noise-levels 1 .8 .6 .4 .2 --batch-size 4

python -m improvnet.inference.uniform_diffusion_evaluate \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --genre blues --strategy timing --metrics genre --steps 16 32 48 \
  --max-blocks 8 --noise-levels 1 .8 .6 .4 .2 --batch-size 1
```

Defaults load the SFT best model and trained genre/continuity encoders under the
scratch `artifacts` directory. Use the corresponding checkpoint flags to change
them. Outputs go to a unique `uniform_diffusion_style_transfer/transfer-*` directory;
`--output-dir` must be new or empty. Checkpoints are read-only; no jobs are submitted.

## Two stages per block

1. With no musical context on the first block, encode only the selected genre, multi-hot source
   instrumentation and zero-elasticity controls. Later blocks use finalized
   generated history in the causal cache, never ground-truth history or future KV.
2. Read the next exact 256 source tokens. Identify fields from token types, not
   position modulo four, so events split across block boundaries remain correct.
3. Replace the requested fraction of **onset/duration** slots with uniformly
   sampled same-type tokens. Freeze every other position. Refine up to 48 passes.
4. Take that timing result, replace the requested fraction of joint **pitch/velocity
   note** slots, and refine up to 48 passes. Freeze the revised timing and all other
   fields. Instruments, metadata, `<S>`, `<T>`, `<D>`, `<E>` never change.
5. Encode only the finalized second-stage result into history. Stop at the source
   `<E>`, source exhaustion, or eight blocks. No generated E or new events can be
   inserted by this strategy. The final SOURCE endpoint is closed before inference,
   allowing a partial final block with an explicit padding mask; outputs aren't repaired.

`--strategy timing` skips stage 4 entirely and commits the timing result. Pitch,
velocity and instruments remain identical to the source at every position.

The current tokenizer uses instrument/note/onset/duration for drums too. A note
token couples pitch and velocity, so those cannot be edited independently here.
Drum pitch changes the percussion voice. Zero duration/velocity candidates are
excluded from editable slots; this doesn't guarantee coherent timing or harmony.

## Noise and sampling

`--steps 48` means **48 per stage**, not 48 total. `--steps 16 32 48` runs three
separate budget comparisons with the same seeded initial perturbations.
Stage self-conditioning resets,
and both sampling and soft self-conditioning are restricted/clamped at every pass.
Padding is excluded from editable counts, entropy, stopping and cache visibility.

The saved model has **no explicit timestep embedding/input**. Define stage-local
noise `t_editable`, decaying linearly from the requested level to zero, and an
approximate whole-canvas level `t_effective = t_editable * editable_count / valid_count`.
The latter scales temperature between the unchanged 0.8 and 0.4 endpoints.
These typed perturbations are **not the full-vocabulary forward distribution used
in SFT**, so matching replacement fraction is not exact distribution matching.

Default flow sampling uses
`p(next) = (t_next / t) * delta(current) + (1 - t_next / t) * p_model(clean)`
on editable slots only. The editable fraction cancels in that ratio. This preserves
partially noisy source information until updates, while allowing any editable field
to change. Every intermediate and final draw is temperature sampled. Argmax is
only a stopping diagnostic. Stable modal predictions plus mean editable entropy
below 0.005 allow independent per-example adaptive stopping.

`--sampler entropy` instead selects low-entropy fields under the Gemma-style
cumulative entropy bound 0.1 and re-noises unselected editable fields each pass.
It is useful for comparison but does not retain the same partial-noise semantics
after its first update. No temperature or stopping sweeps are conducted by default.

## Metrics and report

- MIDI syntax checks the actual joined tokens and compilation, without source
  suffixes. Closed rollouts must compile; open internal event tails can be pending.
- Genre class probabilities are computed on generated canvases, with paired source
  probabilities and jazz uplift. Metadata prefix tokens are excluded by the scorer.
- Pitch identity, short-pattern retention and aligned SSM structure compare the
  generated/source canvases, ignoring rhythm and velocity. Whole-canvas and
  per-block scores are saved; timing-only snapshots remain available for inspection.
  These scores follow serialized note order. Changed onsets can reorder the audible
  melody; onset-order/repetition diagnostics are also saved to expose this limitation.
- Continuity compares preceding up-to-256 generated tokens to the next block.
  Baselines compare source-to-source; source-history-to-generated scores are also
  saved. The first block is explicitly N/A, not zero.

`--metrics genre` scores only jazz/blues probabilities on the whole generated
excerpt, plus syntax and preservation diagnostics, without loading continuity.
Invalid outputs stay in the syntax denominator and cannot silently receive high
other scores. By default, `scores.json` contains small settings, source jazz/blues
scores and one flat row per result. `summary.csv` and `report.md` show the same
compact results alongside MIDI links. Token arrays, detailed class probabilities,
per-block scores and traces are saved only with explicit `--save-details`.

This experiment tests musical editing, **not** whether the model has learned
sequence grammar. Diffusion-test split membership doesn't guarantee held-out
membership for independently trained metric encoders. A single seed and a
source-valid classical cohort are exploratory; inspect and listen before distillation/RL.

## Tests

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover \
  -s tests -p 'test_uniform_diffusion*py'
```

Tests cover immutable slots, typed corruption/sampling/self-conditioning, drums,
effective noise, zero noise, independent stopping, event-spanning cuts, promptless
prefill, generated-only cache appends and closed MIDI output without padding.
