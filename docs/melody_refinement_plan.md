# Melody corruption/refinement: implemented design

The branch uses one causal AR transformer for three tasks. This document describes
the implementation, rather than a second-stage diffusion experiment. The first
full fine-tuning run and listening evaluation remain future work.

## Representation and model

Use the existing AR tokenizer unchanged for musical events: each note is
`instrument, (pitch, velocity), onset_ms, duration_ms`. Timing resolution is 10 ms;
`<T>` advances the clock by 5,000 ms. Notes are assigned to chunks by onset, and
held-note durations can extend through the boundary without creating an attack.

Three task tokens are appended to the original 2,778 IDs, giving 2,781 tokens:
`<REFINE>`, `<RHYTHM>`, `<HARMONIZE>`. No corruption-type tokens are added. A task
token identifies the requested job; it does not identify the sampled corruption.
Old token IDs, genre IDs and transformer weights are retained. The four genre
conditions remain `classical, jazz, blues, unknown`.

A small MLP embeds a scalar strength in [0,1] and adds it to the existing genre
conditioning vector. Its final projection starts at zero, initially preserving
the old model's conditioning behavior. The backbone remains the AR transformer;
no separate encoder or diffusion model is introduced. Existing AR calls retain
their original default behavior.

## Decoder-only seq2seq

```text
continuous genre/instrument/strength conditions
<TASK> optional_clean_left_history <SEP> corrupted_chunk <SEP>
<S> complete_target_response <E>
```

Each nonempty history/source stream has its own `<S>`/`<E>` wrapper and local
millisecond clock. There is no right context. Chunks use token budgets rather
than fixed durations: `CHUNK_TOKENS=3000` caps the corrupted source and the
serialized clean melody/rhythm response. `HISTORY_PROB=0.5` provides a clean
left prefix on half the training examples, capped at `HISTORY_TOKENS=2048`.
For a short piece fitting one chunk, history examples start at a later attack
when possible. A one-attack piece cannot supply a nonempty earlier history.

Full-length melody/rhythm examples with history use about 8,053 positions.
`SEQ_LEN=8192` includes history, source, response, wrappers, separators, task
token and both continuous conditioning positions. Harmonization fits the actual
melody-plus-retained-harmony source and full arrangement response jointly;
its denser response may exceed 3,000 tokens, so its source may be shorter.
No example exceeds the full 8K budget.

Budget checks happen **after corruption and serialization**, so note additions,
removals and all `<T>` tokens count. If the source or combined stream is too long,
shorten the clean chunk by whole onset groups and reapply corruption with the
same RNG state. Never truncate a target, split note attributes, or split a
simultaneous chord. Shrinking must make progress; an individual onset group that
cannot fit raises an explicit error rather than looping. History keeps complete
notes from the tail and resets its clock at the first retained attack.

Teacher forcing predicts every response attribute, including copied attributes,
note count and required structural tokens. Prompt and padding labels are ignored.
The initial response `<S>` is supplied like a decoder start symbol. `<T>` is
needed in streams spanning multiple 5-second segments. `<D>` remains reserved
in the base vocabulary and is not introduced into these task examples. Only valid musical tokens can be
sampled by inference's note grammar.

## Corruptions and strength

The [ImprovNet paper](https://arxiv.org/html/2502.04522v1) supplies the starting
corruption families: pitch/velocity masking, timing masking, whole masking,
pitch permutation, pitch/velocity permutation, fragmentation, incorrect
transposition, note modification, and skyline for harmonization. Our version
scales the affected subset by strength and composes up to three edits. Fragmentation
drops a suffix; note modification merges omissions or splits longer notes.
Slot masks are applied after note edits so they remain attached correctly.

For melody refinement, sample 1–3 distinct corruptions from the eight non-skyline
families. The model sees their combined result and the scalar strength, never
the corruption names. Zero strength is identity. Strong fragmentation or masking
can remove all note evidence in the center, leaving context to guide recovery.

For timing refinement, retain pitches/velocities/instruments and randomly choose
a fraction of note slots from 0–100%, masking both onset and duration for each
selected note. The fraction is the strength condition. Sample exact zero and
one with 5% probability each; otherwise draw a continuous uniform strength.
Each training example/epoch resamples deterministically from its data seed so
resume and worker scheduling do not change the experiment.

No beat-position, beat-grid, inferred-tempo feature or diffusion timestep enters
the model. With 100% timing masking, the target is a plausible learned rhythmic
interpretation, not uniquely determined timing. The surrounding context helps;
examples at piece boundaries have less evidence. Jazz labels include straight
and swung music, so this system does not promise a universal swing ratio.

## Dataset preparation and skyline

Raw root:
`/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/jazz_blues_data`.

| Source | Role |
| --- | --- |
| ARIA: 18,839 jazz + 1,248 blues copied MIDIs | Melody/timing repair and aligned harmonization |
| PiJAMA: 2,736 MIDIs; Doug McKenzie: 307 MIDIs | Piano melody/timing repair and aligned harmonization |
| Weimar: 456 unquantized solos | Clean solo melody/timing repair |
| WikiMT: 213 Jazz-labelled MusicXML lead sheets | Notated melody repair; excluded from performed timing and harmonization tasks |
| Other 3,859-sheet Wikifonia mirror | Requires genre curation; excluded by default |
| DTL1000 metadata | Excluded until note transcriptions are acquired |

MusicXML is read with music21; written repeats are expanded and chord symbols
are excluded from note events. Missing score tempo uses music21's notated-score
default; this does not turn lead sheets into expressive performances. Source
licences and acquisition reports remain with the raw data. R&B is not relabelled
as blues. Additional curated scores can be listed in `curated_scores.jsonl`.

Known solo/lead-sheet sources retain their melody directly. Simultaneous attacks
in melody candidates, including ties introduced by 10 ms rounding, are resolved
to the upper note when constructing a task. This applies to training and inference;
the stored performance remains unchanged. Polyphonic performance
files use attack-based skyline: group overlapping, distinct-pitch attacks within
50 ms of the first attack, keep the highest, and put the other original notes
in accompaniment. Same-pitch repeated attacks and nonoverlapping fast notes start
new groups. Preserve original velocities/durations rather than fixing dynamics.

Skyline partitions original notes without duplication; it is not a reliable
separator for every arrangement. High accompaniment notes or inner-voice tunes
remain extraction failure cases. Inspect `.melody.mid` examples before a long
run. No hand-verified melody quality filter is claimed.

The preparer writes one record per performance plus a byte-offset index. It
hash-deduplicates files and groups title/full-melody-fingerprint matches before
a deterministic 90/5/5 split. Full melody fingerprints are transposition invariant;
they do not guarantee detection of every arrangement or alternate performance.
Full-performance prepared records from the earlier fixed-window implementation
are reusable; old window lists are ignored. New preparation stores full notes
without excluding dense five-second spans.
Original AR pretraining overlap is explicitly not audited yet, so held-out
fine-tuning results must not be described as proof of unseen-composition ability.

Sampling balances task, genre and source rather than letting raw ARIA file count
dominate. Defaults are 35% melody refinement, 30% timing and 35% harmonization.
Each example gets a new chunk, strength and edits each epoch. Validation examples
remain fixed. Paired pitch augmentation is applied to context, source and target.

## Harmonization supervision

For each polyphonic performance, let `P` be all notes, `M` the skyline selection,
and `H` the remaining original notes. The source is `M` plus a randomly retained
fraction of `H`; strength controls the fraction of harmony removed. The response
is the complete original `P`. At strength one, source is melody only, matching
harmonization inference. Optional clean left history is the preceding arrangement;
melody/rhythm tasks use the preceding clean melody. Neither task sees future context.

Melody and harmony must come from the same aligned performance. Unrelated
monophonic and polyphonic datasets cannot be paired as supervised harmonization
examples. Lead-sheet chord symbols are not treated as recorded piano voicings.

Inference generates a complete polyphonic response, replaces generated melody
attacks within 50 ms of each anchor or during its held span, and reinstates the conditioning melody
exactly. Additional generated notes can still be inappropriate; this safeguard
preserves anchors rather than proving a correct arrangement or eliminating every
possible extra-note error. Melody has priority over accompaniment restrikes of
the same pitch/instrument while held. MIDI export ends a previous same-pitch
strike at its next attack to prevent a late note-off from cutting off a retrigger.
Evaluate harmony density, clashes and releases.

## Training and inference operation

`melody_train.py` follows the existing AR trainer's DDP/AMP/accumulation,
AdamW, scheduler, gradient clipping, TensorBoard and resumable atomic checkpoint
structure. The first launch migrates AR weights only, appends task rows and starts
a new optimizer/schedule. Subsequent launches restore the fine-tuning run state.
Checkpoint metadata records exact vocabulary, model settings, genre ordering,
chunk/history budgets and data-index hash. The left-only protocol has metadata
version 2; old short-window fine-tuning checkpoints cannot resume into it. Resume
rejects incompatible settings. The configured batch size, accumulation and run
directory remain user-controlled.

Current settings: batch 48 per GPU, accumulation 1, learning rate 2e-5, warmup 500,
and 20,000 optimizer updates, with learning-rate decay over the final 5,000
updates. The step count is set by `N_STEPS` in `melody_config.py`; the standard
training command needs no step argument. Treat this as an initial fine-tuning
ceiling, not a benchmarked optimum, and select checkpoints using per-task
validation curves and listening results.

`melody_inference.py` extracts a classical melody, optionally refines it, restores
new timing and harmonizes. It processes a full piece chunk by chunk and can run
multiple melody/timing passes. Each pass starts with **no history**, then uses
only that pass's generated output as left history for subsequent chunks (up to
2,048 tokens). Original classical left history and right context are never
provided. Input ownership advances through all source notes even if corruption
removes notes or generation changes their count. Timing decoding preserves note count and
pitch/velocity/instrument by default while generating onsets and durations;
free attribute generation is also available. Use melody strength zero for the
first timing-plus-harmony experiment, then gradually permit melodic changes.

Inference uses checkpointed budgets. Melody/rhythm sources are capped at 3,000
tokens and reserve a complete response of up to 2,999 new tokens (the initial
`<S>` is already in the prompt). Harmony reserves more output space using
`HARMONY_EXPANSION=3.0`, an adjustable estimate of arrangement tokens per source
melody token; the hard bound is still the actual remaining 8K context. At every
note start the grammar reserves four attributes plus `<E>`; a `<T>` also reserves
room for a subsequent complete note. This permits added notes without cutting
off a response mid-note. Fixed-pitch timing additionally checks room for every
anchored note and required clock marker. Per-chunk budgets/history lengths are
written to the inference JSON report.

Saved stage MIDIs and JSON counts allow comparison of input melody, refinement,
timing and final arrangement. Check recognizability, motif/contour retention,
repeated attacks, rests, releases and musical style by listening. Syntax/genre/
pitch/continuity metrics are useful supporting evidence; reconstruction loss or
a genre score alone cannot establish successful style transfer.

The original AR tools and metrics are retained. Unused diffusion and archived
latent-model scripts are removed only from this branch. Raw data, original AR
weights, and other worktrees are not deleted.
