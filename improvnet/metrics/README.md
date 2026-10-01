# Generation metrics

Independent metrics for sampler evaluation and future distillation/RL. MIDI
syntax and a [genre-classifier package](genre_classifier/README.md) are available.
Importing the base metrics package does not load a model or start training.
Reference similarity and continuity remain future, separate metrics.

## MIDI syntax

`MidiSyntaxMetric` now returns **two distinct signals**:

- `result.score`: a graded score in `[0, 1]`, for ranking negative samples.
- `result.valid` / `result.validity_score`: the strict boolean / `0|1` flag,
  requiring correct grammar and a successful MIDI round-trip.

The graded score is `max(0, 1 - error_count / scored_tokens)`. One faulty actual
token counts once, even if multiple rules flag it; each missing required field
or wrapper counts once. `result.issues` contains all diagnostics, and
`result.issue` remains the first diagnostic for older callers. This is a
recovering grammar heuristic, not an exact minimum edit distance or probability.
More faults at the same length generally score lower; different error patterns
can require different numbers of missing-field/orphan-token diagnoses.

For a canvas, `scored_tokens` counts **only canvas positions**, not clean history
or a boundary suffix. All faults in the joined excerpt still count: bad history
is not ignored, and good history cannot inflate a damaged canvas's score. For
whole sequences the denominator is the entire non-padding sequence length.

The existing decoder skips some malformed tokens and incomplete events. This
metric validates before decoding, and checks that the decoder/serialization
does not discard note events. The scanner resynchronizes after faults only to
count later errors; it never repairs predictions, removes noise tokens, or
compiles a repaired sequence. MIDI conversion uses
the project's existing overlap resolution and message ordering behavior.

By default strict validity requires at least one positive-velocity,
positive-duration note. Empty/all-`<T>` and nonsounding outputs score zero.
Any **two consecutive `<T>` tokens involving the generated output** also force
score zero and `valid=False`, even if one musical event exists elsewhere.
The `consecutive_time_tokens` diagnostic is a generation-quality/anti-cheating
rule, not a tokenizer grammar error: long rests can still have
`syntax_valid=True` and `midi_compiles=True`. For canvas scoring, pairs entirely
inside clean history are ignored; pairs crossing into the canvas are checked.
Malformed but recognizably musical events can receive partial credit even if
none are complete; otherwise a missing field would collapse back to a binary
reward. Clean history alone cannot satisfy the canvas's music-evidence guard.
A sole terminal `<E>` after musical history is allowed. Actual MIDI compilation
failures always score zero; malformed grammar skips compilation
(`midi_compiles=None`). `min_sounding_notes=0` explicitly disables the nonempty
guard for pure grammar tests, and should not be used for RL rewards.

### Grammar

- Complete pieces: optional metadata prefixes, `<S>`, events/markers, final `<E>`.
- Melodic **and drum** notes: `instrument → note(pitch, velocity) → onset → dur`.
  Legacy combined note/drum tokens are not part of the current model vocabulary.
- Pedals are not supported; use the current no-pedal tokenizer configuration.
- `<T>`, at most one `<D>`, and the optional `<X>` delimiter occur only at event
  boundaries. Repeated `<T>` is allowed for long rests. `<D>` is optional and is
  not required to be exactly 100 tokens before `<E>`.
- No valid tokens after `<E>`, duplicate `<S>`, misplaced prefixes, bare fields,
  out-of-vocabulary values, or unfinished events.
- `<P>` (the tokenizer's padding token), `<MASK>`, `<SEP>`, `<BLANK>` and `<U>`
  cannot occur in the scored sequence. Padding is excluded **only** by an
  explicit right-padding validity mask. Elastic blank removal, if desired,
  belongs to the sampler's defined slot-removal policy, not this validator.

Values must belong to the supplied tokenizer's exact vocabulary. Pass the
tokenizer used by the model if its configuration differs from the default.
`check_time_order=True` additionally rejects backwards event onsets within the
`<T>` timeline; this is optional because MIDI serialization can reorder events.
Serialization also catches real MIDI limitations, such as too many melodic
instrument channels, that a token grammar alone would miss.

### Whole-piece API

```python
from improvnet.metrics import MidiSyntaxMetric, summarize_syntax

metric = MidiSyntaxMetric(tokenizer)
result = metric.evaluate_sequence(generated_ids)
reward = result.score
if not result.valid:
    print(result.issue)

# Open excerpts need complete events but need not contain piece wrappers.
result = metric.evaluate_sequence(excerpt_ids, require_bos=False, require_eos=False)

results = metric.evaluate_batch(batch_ids, valid_masks=batch_valid_masks,
                                noise_levels=batch_noise_levels)
summary = summarize_syntax(results)
```

Inputs may be integer IDs (including 1-D Torch/NumPy arrays), token tuples, or
JSON-style token lists. `summary["valid_rate"]` is the strict pass rate;
`summary["mean_score"]` and `summary["mean_errors"]` track partial improvement.
`failures` counts first failure codes; `issue_counts` counts all diagnostics.
Empty groups return `None`, not a misleading zero. Malformed API arguments
raise `ValueError`.

### Canvas boundaries

A 256-token window can begin/end in the middle of an event. Evaluate it with
its clean history and, when necessary, **only** the clean fields completing
that final event, not an arbitrary future passage:

```python
result = metric.evaluate_canvas(
    generated_canvas_ids,
    history=clean_history_ids,
    boundary_suffix=clean_fields_finishing_the_cut_event,
    valid_mask=canvas_valid_mask,
    noise_level=corruption_probability,
)
```

The joined excerpt must have complete events. An absent required boundary
suffix fails with `incomplete_event`; this describes an unresolved boundary,
not necessarily a model error. For unconstrained generation there is no known
suffix: evaluate complete committed events or provide generated boundary
completion before scoring. Never use ground-truth suffixes for tokens that
were actually generated by the model.

`allow_cropped_history=True` permits dropping an initial partial event **from
clean history only**, e.g. when the 4096-token history cap begins on a duration
field. No canvas token is dropped. Include enough history to identify the event
containing the canvas start. Invalid history still invalidates the joined
excerpt. Diagnostic indices and event counts refer to that joined excerpt
(after any explicitly allowed clean-history cropping), not canvas-local offsets.

### Noise and timestep evaluation

Score **predicted clean outputs**, not noisy input canvases, at each selected
diffusion timestep/noise level. The grammar and scoring criterion do not change
with noise. `noise_level` is optional metadata in `[0, 1]`; `summarize_syntax`
reports rates in configurable noise bins alongside total rates and failure
counts. For discrete timestep curves, group the returned flags by the caller's
timestep index and summarize each group separately. The metric does not infer
a timestep from mask fraction or change the sampler. For near-miss negatives,
filter `not result.valid`, then rank by descending `result.score` (or ascending
`error_count` within same-length candidates). Retain lower-scoring negatives too.
Never treat partial credit as an assertion that the MIDI compiles, or use syntax
score alone as a measure of music quality.

## Future scores

- Genre: implemented separately under `genre_classifier/`. Architecture,
  training and calibration tools are ready; no production classifier is trained
  yet. Do not use jazz reward for unconditional prompt continuation.
- Reference similarity: start with aligned pitch/note overlap and onset/duration
  tolerances for continuation; use timing-invariant pitch/contour measures for
  style transfer so rhythmic changes are not automatically punished.
- Continuity: use adjacent same-piece positives and different-piece negatives
  matched by genre/instrumentation. Similarity measures compatibility, not a
  guarantee of musical transitions. Comparing jazz transfer against classical
  history can conflict with genre reward; make its use task-dependent.

Keep validity separate from these musical scores. A syntax pass does not mean
the music is good; a binary failure alone also does not provide dense guidance
about which malformed field should be corrected during training. Contrastive
cosine similarity can be negative, but different pieces are not guaranteed to
receive negative scores. For a future gated RL reward, normalize musical scores
to nonnegative ranges or assign invalid sequences an explicit lower penalty:
invalid reward zero must not outperform a valid sequence with negative reward.
