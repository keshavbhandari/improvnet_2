# Four-task refinement design, protocol 3

This branch uses one causal autoregressive transformer for refinement, rhythmic
repair, harmonization and drum addition. The architecture retains the original
20-layer/2,048-width backbone and its genre/instrument conditioning positions.
Musical quality of the new four-task model still requires training and listening.

## Representation and prompt

Each note is four tokens: instrument, pitch/velocity, onset, duration. Timing is
10 ms and `<T>` advances the local clock by 5,000 ms. Held-note durations cross
chunk boundaries without creating new attacks. The existing 2,778 musical IDs
are retained; `<REFINE>`, `<RHYTHM>`, `<HARMONIZE>`, `<DRUMS>` give 2,782 IDs.
The original three task IDs keep their values. The four genre IDs remain
`classical, jazz, blues, unknown`. Drums use the existing drum instrument token
and the existing 41-instrument condition; MIDI export reserves channel 10.

```text
continuous genre + strength + masked style controls / instrument conditions
<TASK> optional_clean_left_history <SEP> corrupted_or_incomplete_source <SEP>
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
voicing density. Drum addition caps its source at 512 but permits a longer full
mix response. Every example, including its clean history, wrappers, task,
separators, response and both continuous positions, fits at most 8,192 positions.

Lengths are measured after corruption and serialization. Overflow shrinks the
source/target time span at whole onset groups and reapplies the same corruption
RNG state. Targets and note attributes are never blindly truncated. Added and
removed notes, drum density, and all clock markers count. Shrinking must make
progress; an indivisible group that cannot fit raises an explicit error.
Inference also bounds the final anchor-preserving arrangement/mix, adding only
complete onset groups within the remaining response budget.

For each task, choose a mono or poly pitched input view with probability 0.5 where
available. Solo sources naturally have identical views. Refinement/rhythm targets
use the selected view. Harmony targets use the full pitched arrangement. A mono
harmony source is melody only, with strength 1; poly harmony sources retain a
random subset of accompaniment according to strength. Drum input is the chosen
pitched view; target is that same view plus its aligned original drums. Drums
are not transposed. All paired pitched notes/history/targets share augmentation.

History is available with probability 0.5. For harmony, always use the clean
preceding polyphonic pitched performance, irrespective of the input view.
Refinement/rhythm history uses the same mono/poly view as its clean target.
Drum history uses that target's pitched view plus its preceding original drums,
including when the pitched view is monophonic. There is no independent history
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
accompaniment removed. Drum addition always removes all original drum notes
from its source; its response contains the complete pitched-plus-drums mix.

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
Refinement/drums have no additional numeric style controls. Validation retains
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

`performances_v3.jsonl` stores pitched notes, selected melody indices and aligned
drum notes separately, with byte-offset indexing and task eligibility. Rebuilt
v3 data has 23,764 performances (21,422/1,160/1,182 splits) and 158 drum-bearing
performances (140/9/9 splits). All current drum labels are jazz from Doug McKenzie;
blues drum transfer has no corresponding labelled supervision here. Old prepared
files and raw files remain intact. Older prepared data discarded drums and must
not silently supply empty drum targets for the fourth task.

Files are hash-deduplicated; title/full transposition-invariant melody fingerprints
group matched tunes before the deterministic 90/5/5 split. This does not guarantee
identifying every alternate arrangement. Original AR pretraining overlap remains
unaudited. Sampling balances eligible task, genre, then source. Defaults are
30% refine, 25% rhythm, 30% harmony, 15% drums. Missing enabled-task supervision
raises an error. Training resamples deterministic task/chunk/edits each epoch;
validation stays fixed, with the same worker/resume guarantees.

## Operation and verification

Protocol 3 checkpoints record vocabulary, seven-feature ordering, all view/style
probabilities, the clean-target history policy, the inclusion of preceding drums,
task weights, budgets and data-index hash. Earlier three-task
checkpoints cannot resume optimizer/data state. `--init-checkpoint` can migrate
their weights into a fresh run, retaining all old IDs and strength parameters,
while initializing the fourth task row and style projection. The original AR
checkpoint can also initialize it. Batch/accumulation/run-directory settings
remain user-controlled. DDP, AMP, atomic saves, per-task validation, RNG/data
position resume and Slurm checkpoint requests retain the existing AR workflow.

The default inference order is refine, rhythm, harmonize, drums. `--pipeline-order`
can move harmony before rhythm; `--input-view poly` retains a complete input
arrangement. Single-task auto input is poly; multi-stage auto input extracts a
melody. Drum addition is last and preserves every pitched input note exactly,
discarding any invented pitched notes in its response. Harmony reinstates all
supplied pitched anchors and removes generated near-duplicates. A held anchor
has priority over same-pitch/instrument accompaniment restrikes. Export releases
an earlier same-pitch strike at its next attack.

Each stage starts without history, then uses its generated output. Harmony always
keeps its full polyphonic history. The default history view for the other tasks
matches their target pitched view; drum history also retains generated drums.
Explicit mono/poly history overrides remain available for those other tasks.
Refinement continues an uncovered source suffix
when the model ends prematurely; the phrase-ending allowance is 5% of the input
attack span, capped at five seconds. This prevents short responses from dropping
large sections and works for short chunks too. No classical-copy fallback is
inserted. Empty refinement raises an error. Reports distinguish attempted spans
from consumed input and include checkpoint/reference identity and all budgets.

Syntax, genre, pitch and continuity metrics support listening; none alone proves
recognizable style transfer. Check repeats, rests, endings, releases, accompaniment
clashes and drum alignment. The first production training/listening evaluation
of the four-task model is still to be run.
