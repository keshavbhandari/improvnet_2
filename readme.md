# ImprovNet: refinement, rhythm, harmony and drums

One causal AR transformer learns four genre-conditioned tasks. Each note remains
`instrument, (pitch, velocity), onset, duration`. The tasks support both solo
melodies and polyphonic pitched arrangements. Timing resolution is 10 ms; `<T>`
advances the clock by five seconds.

## Setup

```bash
source /e/project1/e-dev-2026d09-047/software/miniforge3/etc/profile.d/conda.sh
conda activate /e/project1/e-dev-2026d09-047/envs/improvnet
cd /e/home/jusers/bhandari1/jupiter/improvnet_melody
python -m pip install -e .
export IMPROVNET_ARTIFACTS=/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts
```

## Data and controls

The v3 dataset and reference bank are already built in this workspace. For a new
checkout/dataset, prepare them with:

```bash
python -m improvnet.scripts.melody_prepare
python -m improvnet.scripts.melody_style_references
```

The prepared dataset is `prepared/performances_v3.jsonl` under the configured
`DATA_ROOT`. It stores pitched notes, melody indices, and aligned drum notes
separately. The earlier prepared dataset and raw MIDI/MusicXML files are retained.
Preparation refuses to overwrite an existing output. The current v3 dataset has
23,764 performances with a 21,422/1,160/1,182 train/validation/test split. There
are 158 drum performances from Doug McKenzie: 140/9/9 in those splits. These
aligned drum labels are jazz; this collection has no labelled blues drum examples.
Parse exclusions and duplicate counts are recorded in the preparation report.

The style exporter writes [style_references.json](improvnet/inference/style_references.json).
It contains 40 examples from the training split, with source provenance, aggregate
statistics for the pitched mix and a separate set for its extracted melody.
Notated lead sheets are excluded from this performance reference bank. Rebuild
with `--data`, `--output` and `--per-source` to use a different prepared corpus.

The harmony controls are accompaniment attack groups per second and mean
sounding voicing size at accompaniment attacks. Held notes count toward voicing
size. Rhythm controls are onset-interval mean, standard deviation, 25th/75th
percentiles, and lag-one correlation. Performed attacks are grouped within
50 ms of the first attack, preserving retriggers and nonoverlapping fast notes.
These are aggregate controls: pitches and individual reference note times are
not passed to the model. Interval alternation is a rhythmic descriptor, not a
verified beat-based swing ratio.

## Training

Settings are in `improvnet/model/melody_config.py`:

- Refine, rhythm and harmony: at most **512 source and 512 complete response
  tokens**, including music wrappers. Whole notes/onset groups are preserved.
- Drums: at most **512 source tokens**; the complete pitched-plus-drums response
  can use the remaining **8,192-position** context, including all history,
  separators and continuous conditions.
- Inputs choose mono/poly views equally where available. History is present
  50% of the time and capped at 2,048 tokens. Its view follows the clean target:
  refinement/rhythm use the same mono/poly view as the selected input; harmony
  always uses the preceding polyphonic arrangement; drums use the selected
  pitched view plus preceding drums. The view is selected before corruption,
  with no independent history-view draw. History precedes the target and has no
  right side.
- Harmony/rhythm controls are independently configured to be visible on 50% of
  their respective examples. Explicit presence masks distinguish missing values
  from genuine zero-valued statistics.
- Task weights are 30% refinement, 25% rhythm, 30% harmony and 15% drums. Source
  and genre balancing happens within each eligible task. Drum tasks use only
  performances containing aligned drums.

The new four-task/style-conditioning protocol is **version 3**. Three-task
checkpoints cannot resume its optimizer/data state. Their weights can initialize
it; the original three task IDs are retained, a `<DRUMS>` row is appended, and
new style parameters start with a zero output projection.

Your configured batch size (24), accumulation (1) and run directory have been
retained. The old melody checkpoints and TensorBoard logs have been cleared from
`melody_refinement_v1`, so the next run starts this protocol there using the
original AR initialization. Subsequent jobs resume the new melody checkpoint.

```bash
python -m improvnet.model.melody_train

# Four GPUs
torchrun --standalone --nproc_per_node=4 -m improvnet.model.melody_train
```

The cluster launcher is
`/e/home/jusers/bhandari1/jupiter/run_scripts/improvnet/melody_single_node_launch.sh`.
It writes Slurm output/error to the adjacent `melody_run_logs/` directory.
It follows the AR launcher's rendezvous and checkpoint-on-signal structure.
`AUTO_RESUBMIT=1` (default) queues a successor after the time-limit signal requests
a checkpoint and training exits successfully. Set `AUTO_RESUBMIT=0` in the
submission environment to disable continuation. Normal completion and training
failures do not queue another job. No job is submitted by editing the launcher.

The trainer retains DDP, AMP, accumulation, clipping, TensorBoard and resumable
atomic checkpoints. Only response tokens contribute to loss; validation reports
all four tasks. `N_STEPS` sets the schedule horizon. `--steps` can stop a pilot
early without shortening that horizon. Monitor each task and listen to stage
outputs when selecting checkpoints. The original AR weights are preserved.

## Generation

After training a v3 checkpoint, the default pipeline runs refinement, rhythm,
harmony and then drums. A seeded random jazz/blues reference from the JSON bank
supplies the controls unless `--style-reference /path/to/performed_jazz.mid` is
provided. The same reference is used across chunks/stages; mono rhythm uses its
melody statistics, and poly rhythm uses its pitched-mix statistics.

```bash
python -m improvnet.inference.melody_inference \
  --model "$IMPROVNET_ARTIFACTS/melody_refinement_v1/best_model.pt" \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --output "$IMPROVNET_ARTIFACTS/clair_jazz.mid" \
  --genre jazz --melody-strength .2 --rhythm-strength .5 --passes 1
```

Examples of controls:

```bash
# Additional arguments to the command above:
--style-reference /path/to/performed_jazz.mid
--pipeline-order refine,harmonize,rhythm,drums
--input-view poly --history-view poly
--no-drums
--no-style-conditioning
```

`--task refine|rhythm|harmonize|drums` selects one task. Auto input view extracts
a melody for a multi-stage pipeline and retains the pitched arrangement for a
single task; `--input-view mono|poly` overrides it. Harmony can precede rhythm:
polyphonic timing keeps chord attacks together while preserving instrument,
pitch, velocity and note count by default. `--free-timing-pitches` permits free
attribute/count changes. Harmony preserves all supplied pitched notes. Drum
addition preserves the complete supplied pitched arrangement and adds only drum
notes, exported on MIDI channel 10. Drums must be the final selected task.
`--passes` repeats the pitched tasks in their specified order; drums run once.

Each stage begins without history. Later chunks use only that stage's generated
output. The default `--history-view match` follows the target pitched view;
harmony always keeps its full polyphonic history, and drum history retains its
generated drums even with a monophonic pitched view. Explicit `mono`/`poly`
overrides remain available for the other tasks. Corrupted inputs and complete
outputs are measured after serialization, including `<T>` tokens. Growing
corruptions or dense drum targets shrink input spans at complete onset groups.
Refinement treats a substantially premature `<E>` as a completed prefix and
continues the uncovered input; its ending allowance is 5% of the input attack
span, capped at five seconds. It supplies no copied classical fallback.

Progress bars are enabled by default, including checkpoint/device status,
per-stage input coverage and per-chunk prefill/token decoding. `--no-progress`
suppresses them. Token bars show maximum budgets; a valid response can end early.

Outputs include `.melody.mid`, `.input.mid`, the selected stage MIDIs, final
MIDI and a JSON report. The report records checkpoint identity, stage order,
reference statistics, token budgets, actual input consumption and early endings.
`--seed` controls reference selection/corruptions/sampling; `--temperature 0`
selects greedy decoding. Strength is input corruption severity, not a measured
percentage of jazz character. The controls and token limits do not establish
musical quality; compare motif retention, repeated attacks, releases and style.

## Checks and existing tools

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests -q
python -m improvnet.inference.melody_inference --help
```

The original AR tools and independent syntax, genre, continuity and pitch metrics
remain available. See the [design](docs/melody_refinement_plan.md) and
[metric documentation](improvnet/metrics/README.md). No production run of the new
four-task model is claimed. Original-AR pretraining overlap is not yet audited.
