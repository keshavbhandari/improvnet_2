# Four-task refinement design, protocol 4

This branch uses one causal autoregressive transformer for refinement, rhythmic
repair, harmonization and targeted missing-stem restoration. The architecture retains the original
20-layer/2,048-width backbone and its genre/instrument conditioning positions.
Musical quality of the new four-task model still requires training and listening.

## Representation and prompt

Each note is four tokens: instrument, pitch/velocity, onset, duration. Timing is
10 ms and `<T>` advances the local clock by 5,000 ms. Held-note durations cross
chunk boundaries without creating new attacks. The existing 2,778 musical IDs
are retained; the three original task IDs and reserved legacy `<DRUMS>` ID
keep their values. Appending `<STEM>` gives 2,783 IDs.
The four genre IDs remain
`classical, jazz, blues, unknown`. Drums use the existing drum instrument token
and the existing 41-instrument condition; MIDI export reserves channel 10.

```text
continuous genre + strength + masked style controls / instrument conditions
<TASK> [requested_instrument_for_STEM] optional_clean_left_history <SEP> corrupted_or_incomplete_source <SEP>
<S> complete_response <E>
```

Music streams have independent local clocks and `<S>/<E>` wrappers. There is no
right context. Teacher forcing predicts every response attribute and structural
marker, with prompt/padding labels ignored. `<D>` is reserved and unused here.
The note grammar permits only complete valid events during inference.

## Lengths and mixed views

Refine/rhythm/harmony cap both source and serialized complete response at 512
tokens. Harmony can use a shorter source to leave room for added voices in its
512-token response. Inference's expansion estimate also accounts for requested
voicing density. Stem restoration caps its source at 512 but permits a longer full
mix response. Every example, including its clean history, wrappers, task,
separators, response and both continuous positions, fits at most 8,192 positions.

Lengths are measured after corruption and serialization. Overflow shrinks the
source/target time span at whole onset groups and reapplies the same corruption
RNG state. Targets and note attributes are never blindly truncated. Added and
removed notes, drum density, and all clock markers count. Shrinking must make
progress; an indivisible group that cannot fit raises an explicit error.
Inference also bounds the final anchor-preserving arrangement/mix, adding only
complete onset groups within the remaining response budget.

For refinement, rhythm and harmony, choose a mono or poly pitched input view with
probability 0.5 where available. Solo sources naturally have identical views. Refinement/rhythm targets
use the selected view. Harmony targets use the full pitched arrangement. A mono
harmony source is melody only, with strength 1; poly harmony sources retain a
random subset of accompaniment according to strength. Stem input retains all
instrument classes except the selected missing class;
target is the full original mixture. This task never applies skyline to its
input or history and requires at least two original instrument classes. Drums
are not transposed. All paired pitched notes/history/targets share augmentation.

History is available with probability 0.5. For harmony, always use the clean
preceding polyphonic pitched performance, irrespective of the input view.
Refinement/rhythm history uses the same mono/poly view as its clean target.
Stem history uses the preceding full original mix, including the selected stem.
Its prompt explicitly names the missing tokenizer instrument class. Drums have
selection weight 2 and each other eligible class has weight 1. There is no independent history
view draw: corruption or masking cannot change the selected clean history view.
Cap it at 2,048 tokens, keeping complete tail notes and rebasing its first attack
to zero. Short pieces may start at a later attack to provide a clean prefix.
No future notes, corrupted history or genre-mismatched history are supplied.

## Corruptions

The [ImprovNet paper](https://arxiv.org/html/2502.04522v1) supplies the starting
corruption families. Refinement composes up to three of: pitch/velocity masking,
onset/duration masking, whole masking, pitch permutation, pitch/velocity
permutation, fragmentation, incorrect transposition, and note modification.
Strength scales the affected subset; zero is identity. Fragmentation removes a
suffix. Note modification merges omissions or splits long notes. Slot masks are
applied after structural edits. Corruption names are never serialized.

Rhythm keeps instrument/pitch/velocity and masks onset/duration together for a
sampled fraction of simultaneous attack groups. Polyphonic decoding samples one
onset per original simultaneous chord and repeats it for its remaining notes.
It reserves space for remaining attack groups inside each five-second bin.
Durations are generated individually. Exact strengths zero/one occur with 5%
probability each; other strengths are uniform. Harmony controls the fraction of
accompaniment removed. Stem restoration always removes all notes for one tokenizer instrument class
from its source; its response contains the complete original mix. Multi-track
recordings of a single instrument class cannot supply missing-stem supervision.

## Aggregate style conditioning

A separate small MLP receives seven transformed values plus seven presence bits.
Its output is added to the same genre vector as the strength embedding; no new
sequence positions are consumed. Its final projection starts at zero so weight
migration initially preserves the previous model's behavior.

| Control | Definition |
| --- | --- |
| Accompaniment attack frequency | Anchored accompaniment attack groups per second |
| Accompaniment voicing density | Mean distinct sounding pitch/instrument pairs at accompaniment group attacks, including held notes |
| IOI mean | Mean pitched attack-group interval, milliseconds |
| IOI standard deviation | Population standard deviation of those intervals |
| IOI quartiles | 25th and 75th percentiles, milliseconds |
| IOI lag-one correlation | Correlation between adjacent intervals; negative values describe alternation |

Performed groups are anchored at the first attack and extend at most 50 ms.
Same-pitch retriggers and nonoverlapping fast notes start new groups. Style
statistics contain no pitches, individual note times, instrument IDs or chord
progressions. IOI measures are not beat-grid features or a guaranteed swing ratio.
Positive controls use `log1p` scaling, with milliseconds first divided by 1,000;
correlation is unscaled. Missing values are zeroed behind explicit presence bits.

Harmony controls are visible on 50% of harmony examples; rhythm controls on 50%
of rhythm examples. They summarize the clean local target, so this is deliberate
conditional supervision. Hidden controls cannot expose their values to the model.
Refinement/stem restoration have no additional numeric style controls. Validation retains
fixed examples and visibility masks; held-out reconstruction with target-derived
controls is distinct from listening to transfer with an external reference.

Inference either samples a same-genre entry from the training-only JSON reference
bank using the seed, or summarizes `--style-reference` MIDI. One reference is
held fixed across chunks/stages. Mono rhythm uses its melody IOIs; poly rhythm
uses pitched-mix IOIs. A mono reference has no harmony controls. The JSON bank
contains aggregate statistics and provenance, not note sequences. It currently
has 40 entries: 32 jazz and 8 blues, from performed sources. `--no-style-conditioning`
tests the learned missing-control mode. All control probabilities are configurable.

## Data and splitting

Raw root is `/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/jazz_blues_data`.
Sources remain the copied ARIA jazz/blues subset, PiJAMA, Doug McKenzie, Weimar
solos, and labelled WikiMT Jazz scores. MusicXML repeats are expanded with
music21; chord symbols are excluded from note events. Other unlabelled scores
and DTL metadata are excluded. Scores teach refinement but not performed timing
or drums. Raw licences/acquisition reports remain with the data; R&B is not
relabeled as blues. Explicit additional sources use `curated_scores.jsonl`.

Polyphonic pitched performances use attack-based 50 ms skyline with original
note identity/velocity/duration retained. Quantized simultaneous melody ties
resolve to the upper note. Skyline is approximate: high accompaniment and inner
melodies remain failure cases. Known solo sources retain their own melody.

`performances_stems_v4.jsonl` retains pitched notes, selected melody indices and
aligned drum notes separately. Its byte-offset index adds the actual canonical
instrument classes and mixture eligibility. The 23,764 original jazz/blues
performances retain their existing 21,422/1,160/1,182 splits. The new preparation
adds exactly 47,528 explicitly labelled classical performances, preferring genuine
mixtures from GigaMIDI, ClassicalArchives and SymphonyNet. It targets 60% classical
mixtures, using the verified attainable count when the pool is smaller; the
adjacent report records the actual count. Raw files and old prepared corpora remain.
The current build uses all 17,237 verified classical mixtures (36.3% of classical
pieces), compared with 2,654 (11.2%) previously. Total split sizes are
64,233/3,538/3,521. Aligned percussion is present in 9,574 classical pieces and
the existing 158 jazz pieces.

Preparation excludes exact/base-melody copies and groups added recordings/full
transposed melody fingerprints before deterministic 90/5/5 splitting. This does
not identify every alternate arrangement. Original AR pretraining overlap remains
unaudited. `CLASSICAL_PROB=2/3` gives eligible classical draws twice the combined
jazz/blues share. Jazz/blues split their remaining portion. Non-stem classical
tasks choose mixture versus solo pools with probability 0.75 where both exist.
Stem tasks always require actual mixtures. Sources balance within each selected
pool. Task weights are 30% refine, 25% rhythm, 30% harmony and 15% missing stem.
Validation fixes tasks/chunks/edits; training resamples deterministically each epoch.

## Operation and verification

`melody_prepare_stems` builds the active corpus and refuses to overwrite outputs.
Protocol 4 checkpoints record the vocabulary, missing-stem semantics, instrument
weighting, mixture and genre probabilities, history policy, numeric controls,
length budgets and data-index hash. Earlier checkpoints cannot resume this
optimizer/data state; weight-only migration can initialize a fresh run while
preserving old IDs. The active fresh run uses the original AR initialization,
with batch 24 and accumulation 1 unchanged at the existing checkpoint path.
The four-GPU launcher retains its AR-style signal/checkpoint/continuation flow.

Validation reports the full response loss for each task, plus `stem_target` and
`stem_target/<instrument>` for attributes of the missing notes only. These separate
metrics prevent copied context from hiding poor missing-stem predictions. Training
still learns the full mix with ordinary response-token cross-entropy.

Default inference order is refine, rhythm, harmonize, stem. `--target-instrument`
defaults to `drum`; `--task stem --input-view poly --target-instrument "Acoustic Bass"`
adds bass to a bass-free mix. The stem decoder restricts additions to that class
and preserves every input anchor exactly, including existing drums. It rejects
requests for an instrument already supplied. Stem addition runs once at the end;
other task order remains configurable. Harmony reinstates supplied pitched anchors.

Each stage starts without history, then uses only its generated output. Stem
history is always the complete mix; harmony uses the full pitched arrangement.
Other tasks retain target-matched mono/poly history. Generated complete outputs,
clock markers, target instrument prefixes and both condition positions count
toward the 8,192 limit. Source windows and whole responses shrink at onset groups
when needed; a missing-stem training window must contain selected-stem attacks
and retained notes. Unusable stem windows resample instead of producing empty targets.
Dense note-adding corruption falls back to attribute masking at the same strength
when an indivisible clean chord fits but the corruption would not.

Refinement continues uncovered source suffixes after premature endings. The
ending allowance is 5% of the input attack span, capped at five seconds. Empty
refinement raises an error. Progress bars and JSON reports retain budget,
consumption, reference, task and requested-instrument information. Musical quality
of this new task still requires fresh training and listening.

Syntax, genre, pitch and continuity metrics support listening; none alone proves
recognizable style transfer. Check repeats, rests, endings, releases, accompaniment
clashes and drum alignment. The first production training/listening evaluation
of the four-task model is still to be run.
