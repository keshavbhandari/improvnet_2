# Pitch retention against a reference canvas

An untrained, reference-conditioned diagnostic—not a genre score or a calibrated
probability of human recognition. Compare the generated **canvas** against the
corresponding original excerpt, without adding the clean prompt to either side.
No existing model, training script, shared preprocessing or checkpoint is changed.

## Components and safeguards

1. **Pitch identity:** construct a cross-similarity matrix and run global,
   monotone, one-to-one affine-gap alignment. The pitch kernel is the actual
   cosine of three normalized feature blocks: exact MIDI pitch, pitch-class
   one-hot, and circularly smoothed pitch-class vectors. It is NOT cosine of
   raw positive MIDI numbers. With the default weights, an exact match is 1,
   an octave equivalent is 0.85, and chromatic neighbors receive limited credit.
   Credible alignment pairs contribute their similarity to a soft reference
   recall and generated precision. F-beta (beta=2) combines them: reference
   coverage matters most, but arbitrarily much added output is penalized.
2. **Short-pattern preservation:** average retention of ALL reference windows
   of 3, 4 and 5 notes, not just the best motif. A window must retain all its
   reference positions along the same global alignment, with at most four
   inserted notes. Score its pitch matches and the pitch-class intervals
   **between matched backbone notes**, ignoring inserted ornaments' intervals.
   Missing notes/windows score zero. Note windows are not time windows.
3. **Aligned structure:** select corresponding reference/generated note
   positions from the retained alignment and construct their SSMs. Compare
   weighted upper-triangle entries by cosine, EXCLUDING the diagonal. Multiply
   this cosine by confidence-weighted reference pair coverage; matching one
   tiny subsection cannot receive full structural credit. If both off-diagonal
   vectors contain no evidence, abstain rather than award 1 to two identity
   matrices. Raw `ssm_cosine` and `structural_pair_coverage` remain inspectable.

The total is 0.70 identity + 0.20 patterns + 0.10 structure by default. When a
component is unavailable (`None`, e.g. a singleton or no structural evidence),
renormalize the available weights. Empty generated pitch sequences score 0;
an empty/drum-only reference raises an explicit “not applicable” error.
Every score is bounded in [0,1]. Weights, penalties and thresholds are heuristic
and should be validated by listening before use as RL rewards. Alignment chooses
the highest affine-gap objective, not an exhaustive optimum of the final composite.

Global transposition search is optional and OFF by default. If enabled, apply
one global shift to the generated sequence; never transpose individual notes
freely or clip virtual shifted pitches to MIDI range. Stable ties prefer shift 0.
Quadratic alignment is capped at `MAX_NOTES=2048` per side; no silent truncation.

## API

```python
from improvnet.metrics.pitch_similarity import PitchSimilarityMetric

metric = PitchSimilarityMetric()
result = metric.evaluate_pitches([60, 62, 64, 67], [60, 61, 62, 63, 64, 66, 67])
print(result.to_dict())

# Flat tokenizer tokens or IDs, including partial canvas boundaries:
result = metric.evaluate_tokens(
    reference_canvas, generated_canvas,
    reference_context=original_history,
    generated_context=clean_history,
    generated_mask=valid_canvas_mask,
)
```

Context is used ONLY to resolve active instruments at a mid-event boundary;
context pitches never enter the metric. Provide it for slices beginning inside
drum events; otherwise instrument-less notes are treated as pitched notes.
Metadata, velocity, durations, onsets, `<T>`, `<D>` and `<E>` do not enter the
pitch score. Drum/percussive token instruments are excluded. Explicit masks
exclude padding; unresolved noise tokens raise errors. This is not a substitute
for `MidiSyntaxMetric`: missing/reordered event fields must be checked separately.

## Audible demo you can run

From the repository root in the `improvnet` environment:

```bash
# Random excerpt of the bundled Clair de lune, then rule-based changes:
python -m improvnet.metrics.pitch_similarity.demo --modification mixed --seed 42 --plot

# Your input and a fixed starting point; no files are overwritten:
python -m improvnet.metrics.pitch_similarity.demo --input /path/to/piece.mid \
  --start-seconds 30 --seconds 12 --modification ornament --probability 0.3 --plot

# Two different excerpts from one piece (not expected to score 1):
python -m improvnet.metrics.pitch_similarity.demo --input /path/to/piece.mid \
  --mode two-segments --start-seconds 10 --second-start-seconds 40 --seconds 12

# A real generated canvas against its original counterpart:
python -m improvnet.metrics.pitch_similarity.demo --input /path/to/original.mid \
  --generated /path/to/generated.mid --mode compare \
  --start-seconds 30 --second-start-seconds 0 --seconds 12
```

Outputs go to a unique scratch folder unless `--output-dir` specifies a fresh
directory: `original.mid`, `comparison.mid`, `scores.json` (pitches, alignment,
configuration and component breakdown), `matrices.npz`, and optional
`matrices.svg` (no matplotlib required). Large SVG previews use block means;
**scoring and saved arrays never resize SSMs**. Listening copies may shorten
overlapping repeated same-pitch/channel notes to avoid MIDI note-off ambiguity.
Input files and metric pitch sequences are never mutated.
`midi_pitch_roundtrip` verifies the listening copy's extracted pitch sequence
against the scored sequence; any discrepancy from MIDI tick quantization or
skyline/overlap handling is reported, not hidden. Excerpts containing multiple
programs on the same channel are rejected by the simple listening serializer,
rather than silently replaying them with the wrong instrument.

Modifications: `copy`, `timing`, `ornament`, `substitute`, `shuffle`, `transpose`,
`mixed`, `unrelated`. These are controlled tests, not claims of genuine jazz
style transfer. `timing` applies a monotone onset warp, duration and velocity
changes without changing attack order. `--allow-transposition` enables metric
key tolerance; `--transpose` changes the demo's pitches.

`--representation skyline` optionally extracts the highest currently sounding
non-drum voice before comparison. It can miss inner-voice/bass melodies and is
timing-dependent; it is not strict rhythm invariance. `all-notes` is the default
and can be distracted by reharmonized accompaniment. For token inputs, the
metric uses their existing order; for MIDI inputs it orders attacks by onset,
breaking chord ties by descending pitch. Rhythm changes that reorder attacks
can change the score, even though numerical timings are not scored. A trustworthy
melody/voice selection is still needed for strongly polyphonic style transfer.

## Tests

```bash
python -m unittest discover -s tests -p 'test_pitch_similarity*.py' -v
```

Tests cover explicit cosine feature equivalence/PSD, exhaustive tiny affine-gap
paths, deterministic monotone alignment, randomized bounds, edits and ornament
retention, reordered identical histograms, identity-SSM ambiguity, diagonal
exclusion, partial coverage, repeated-note reuse, excess-output attacks, short/
empty inputs, global transposition, drums and boundary context, padding and
invalid inputs. Demo tests check deterministic edits, skyline endpoints,
score breakdowns, saved matrix dimensions, MIDI pitch round trips and no overwrite.
These establish implementation invariants—not perfect perceptual recognition.
