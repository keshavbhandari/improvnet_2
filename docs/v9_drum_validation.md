# V9 drum validation diagnosis — 12 November 2026

TensorBoard source: `/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/melody_refinement_v9/runs/melody_refinement_v9/`.
Snapshot ends at step 19,000; the ongoing V9 allocation was left running.

| Validation CE | Step 10,000 | Step 19,000 | Lowest observed |
|---|---:|---:|---|
| Full stem response | 0.2676 | 0.2826 | 0.2676 at 10,000 |
| Missing drum attributes | 0.4871 | 0.5800 | 0.4738 at 7,000 |
| Kit-only missing drum attributes | 0.7348 | 1.0493 | 0.6925 at 5,500 |
| Full-percussion missing drum attributes | 0.3665 | 0.3515 | 0.3515 at 19,000 |
| Overall validation | 0.6099 | 0.6063 | 0.6062 at 18,500 |

Mean training CE falls from 0.4820 over steps 9,001–10,000 to 0.4367 over
17,001–18,000. Kit CE rises about 43% between 10k and 19k while full percussion
improves. This is consistent with a kit-specific generalization problem, possibly
overfitting; global training CE alone cannot prove kit-specific overfitting.
V9 did not log training kit CE, so V10 adds it.

Reconstructing the deterministic V9 validation sampler over all 728 files gives
200 refinement, 460 harmonize, and 68 stem examples. The stem examples contain
33 drum windows: **12 kit and 21 full**. Other intended stem draws route to a
valid task when the music lacks a usable missing stem. Small fixed kit coverage
limits how confidently this curve represents all jazz drum performances.

Recommended fix: retain the requested clean V10 stem inputs and always-present
safe references, then evaluate a larger **piece-held-out kit cohort**, with
several deterministic windows per eligible recording. Select/checkpoint and
listen using missing-stem kit metrics or generated kit/beat quality, rather than
only total response CE, which includes easy copied notes and can conceal a kit
regression. For V9 the kit curve favors roughly 5.5k–10k rather than 19k, but the
current overall `best_model.pt` was selected at 18.5k. Those earlier weights may
no longer exist; this analysis does not recreate them.

Do not assume that merely increasing kit oversampling or training longer fixes
overfitting. V10 keeps the existing loss, sampling weights and optimizer schedule
for review; it adds train/validation palette comparisons, sample/token counts,
and generated beat metrics. The expanded independent kit cohort and a change to
the checkpoint-selection objective remain recommendations, not silently applied
changes to the held-out split or training objective.
