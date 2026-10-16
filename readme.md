# ImprovNet: refinement, rhythm, harmony and targeted stems

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

The active protocol uses `prepared/performances_stems_v4.jsonl` under `DATA_ROOT`.
To prepare a fresh corpus (existing output paths are never overwritten):

```bash
python -m improvnet.scripts.melody_prepare
python -m improvnet.scripts.melody_prepare_stems --workers 24
python -m improvnet.scripts.melody_style_references
```

The corpus retains the 23,764 jazz/blues performances and adds **47,528 labelled
classical performances**, a **2:1** non-jazz to jazz/blues ratio. Preparation
prefers genuine instrument mixtures from the labelled GigaMIDI, ClassicalArchives
and SymphonyNet pools, targeting 60% classical mixtures where available. The
report beside the dataset records the actual fraction and exclusions. A mixture
requires at least two distinct tokenizer instrument classes, counting drums.
Multiple piano tracks alone are one instrument class and cannot supply this task.
All notes, including percussion, remain aligned to the original performance.

The current build contains **17,237 classical mixtures (36.3%)**, up from 2,654
(11.2%) in the previous classical subset. This uses every verified mixture in
the scanned pool; the requested 60% was not attainable from those candidates.
Train/validation/test contain 64,233/3,538/3,521 pieces. Classical drum-bearing
pieces number 9,574; the original 158 jazz drum-bearing pieces remain unchanged.

The original jazz/blues splits are retained. Exact duplicates and full transposed
melody copies of the base corpus are excluded. Added recording groups and full
melody copies stay within one deterministic split. The new index records actual
instrument classes and mixture eligibility. Raw data and older prepared corpora
are retained. Original AR pretraining overlap remains unaudited.

The style exporter writes [style_references.json](improvnet/inference/style_references.json).
It contains 40 examples from the training split, with source provenance, aggregate
statistics for the pitched mix and a separate set for its extracted melody.
References remain jazz/blues; classical additions are training examples.
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
- Missing stem: at most **512 source tokens**; the complete original-mix response
  can use the remaining **8,192-position** context, including all history,
  separators and continuous conditions.
- Refinement/rhythm/harmony choose mono/poly input views equally where available. History is present
  50% of the time and capped at 2,048 tokens. Its view follows the clean target:
  refinement/rhythm use the same mono/poly view as the selected input; harmony
  always uses the preceding polyphonic arrangement; missing-stem examples use
  the preceding complete mixture and retain all remaining instruments in the input.
  The view is selected before corruption,
  with no independent history-view draw. History precedes the target and has no
  right side.
- Harmony/rhythm controls are independently configured to be visible on 50% of
  their respective examples. Explicit presence masks distinguish missing values
  from genuine zero-valued statistics.
- Task weights are 30% refinement, 25% rhythm, 30% harmony and 15% missing-stem
  restoration. In the fourth task, remove **all notes of one instrument class**,
  explicitly name that class in the prompt, and predict the **complete original
  mix**. Only verified mixtures qualify. Drums have selection weight 2; each
  other eligible instrument has weight 1. There are no fabricated missing stems.
- `CLASSICAL_PROB=2/3` samples classical versus jazz/blues combined at 2:1 within
  each eligible task. Jazz/blues share the remaining portion. A task uses only
  its eligible genres. Classical refinement/rhythm/harmony prefer mixture records
  with probability `MULTI_INSTRUMENT_PROB=0.75` where both views are available.
  Stem restoration always draws from mixtures. Sources are balanced within the
  chosen genre and mixture pool.

The targeted-stem protocol is **version 4**. Earlier optimizer/data state cannot
resume it. Weight-only migration preserves all existing token IDs, including the
reserved legacy `<DRUMS>` ID, and appends `<STEM>`. The stem prompt starts with
`<STEM> (instrument, requested_class)` before the optional full-mix history.

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
all four tasks, plus missing-stem-only and per-instrument losses. These extra
metrics exclude copied input notes so copying cannot conceal poor stem prediction. `N_STEPS` sets the schedule horizon. `--steps` can stop a pilot
early without shortening that horizon. Monitor each task and listen to stage
outputs when selecting checkpoints. The original AR weights are preserved.

## Generation

After training a v4 checkpoint, the default pipeline runs refinement, rhythm,
harmony and then the requested stem (drums by default). A seeded random jazz/blues reference from the JSON bank
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
--pipeline-order refine,harmonize,rhythm,stem
--input-view poly --history-view poly
--no-stem
--no-style-conditioning
--task stem --target-instrument "Acoustic Bass" --input-view poly
```

`--task refine|rhythm|harmonize|stem` selects one task. Auto input view extracts
a melody for a multi-stage pipeline and retains the complete supplied mix for a
single task; `--input-view mono|poly` overrides it. Harmony can precede rhythm:
polyphonic timing keeps chord attacks together while preserving instrument,
pitch, velocity and note count by default. `--free-timing-pitches` permits free
attribute/count changes. Harmony preserves supplied pitched notes.

`--target-instrument` selects an existing tokenizer instrument class, default
`drum`. The stem task preserves **every supplied note**, including existing drums,
and adds only notes of the requested instrument. Supply a mix with that instrument
absent; requesting an instrument already present raises an error. Same-class
tracks cannot be targeted separately. Stem addition is last and runs once after
`--passes` repeats the other tasks. `--no-stem` skips it; `--no-drums` remains an
alias. Drums export on MIDI channel 10.

Each stage begins without history, then uses its generated output. Harmony keeps
its preceding polyphonic pitched history. Stem restoration always keeps the full
generated mixture, irrespective of `--history-view`. Refinement/rhythm default to
the target's mono/poly view, with explicit overrides available.
Corrupted inputs and complete
outputs are measured after serialization, including `<T>` tokens. Growing
corruptions or dense missing-stem targets shrink input spans at complete onset groups.
If a clean chord fits but a note-adding corruption makes that indivisible chord
too large, refinement uses attribute masking at the same strength instead.
This fallback applies during training and generation and preserves the target.
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
