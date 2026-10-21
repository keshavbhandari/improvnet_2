# ImprovNet: classical-to-jazz/blues style transfer

V5 trains three tasks: monophonic melody refinement, combined rhythm/harmony,
and groove-conditioned stem addition. Stem training adds mild timing/duration
errors and rare accompaniment pitch errors, so the model can repair the
retained mix while adding drums. Skyline melody pitches are protected.
Drum inputs, history, targets and generated hits use only kick, snare/side-stick,
hi-hat, tom and cymbal sounds; auxiliary percussion is excluded.

[Task sequences, prompt history, data and decoding](docs/melody_refinement_plan.md)

## Setup

```bash
source /e/project1/e-dev-2026d09-047/software/miniforge3/etc/profile.d/conda.sh
conda activate improvnet
cd /e/home/jusers/bhandari1/jupiter/improvnet_melody
python -m pip install -e .
```

## Training

V5 initializes from the original AR checkpoint, with a new optimizer/scheduler.
Checkpoints use `artifacts/melody_refinement_v5`; previous runs are preserved.
Task weights are refinement 20%, harmonize 40%, and stem addition 40%.
Refinement first merges all detected ornaments, then independently chooses
a contour, ordinary-merge, pitch or timing edit. Its conditioned strength reflects
the mandatory merge, while the additional edit uses the requested strength.
Additional refinement levels 10%, 20%, …, 100% receive equal exposure; 5% of
requests omit the additional edit. Pitch/timing edits affect that fraction of
remaining notes/attack groups directly. Actual total changes are logged
separately because mandatory merges can increase them. Pitch errors span up to ±7 semitones and additive onset/
duration errors up to ±50 ms, in 10-ms steps; prompt-history edits use the same
bounds. Stem pitch errors still protect the skyline melody.
All tasks show eligible prompt history 75% of the time. Harmonize masks
60–100% of onset/duration groups, with 25% explicit full masking. Stem source
pitch-error rates vary between 0% and 5%, with shifts up to ±7 semitones applied
only to accompaniment; 20% of examples are explicitly pitch-clean.
Batch size stays 24/GPU, accumulation 1. Task sequences and corruption remain
on the fly; the expanded existing corpus is not prepared again.

The launcher is staged at
`/e/home/jusers/bhandari1/jupiter/run_scripts/improvnet/melody_single_node_launch.sh`
with logs in `melody_run_logs_v5`. The pre-kit V5 job `2431554` was stopped for a fresh kit-only run; its
checkpoints and TensorBoard logs are archived under
`artifacts/melody_refinement_v5_pre_kit_2431554`. Fresh kit-only job `2432602`
was submitted from the original AR checkpoint on 2026-10-21.

[Additional performance MIDI and provenance](docs/jazz_performance_v5.md)

## Inference after training

```bash
cd /e/home/jusers/bhandari1/jupiter/improvnet_melody
python -m improvnet.inference.melody_inference \
  --model /e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/melody_refinement_v5/best_model.pt \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --output improvnet/inference/clair.mid \
  --task pipeline --genre jazz --melody-strength .2 --rhythm-strength .9 \
  --instruments "Acoustic Piano" drum
```

The canonical command batches ready tasks on one GPU/model. The old
`parallel_inference` module is a compatibility alias. V5 runs
`refine -> harmonize -> stem`; there is no orchestration task. Requested pitched
instruments condition harmonization directly. `--target-instrument` reserves a
specific part for later addition. Each task can also run on its own.

One rhythm reference and source pulse guide harmonization and stem generation.
`--style-reference` accepts a MIDI; otherwise a compatible training reference
is selected. Seed/top-k stay internal, device selection is automatic, and
progress always appears. Intermediates use short names in `clair/`, including
`refine.mid`, `harmonize.mid`, `stem.mid`, `melody.mid` and `report.json`; the final
file is `clair.mid`. Missing drums are reported without aborting the pipeline.

This branch now supports V5 inference only; use the older branch for older
checkpoints. Reserved tokenizer IDs remain fixed for AR initialization.
Run current checks with `python -m pytest -q` (six feature modules under
`tests/melody`). Other-model tests remain separate.
