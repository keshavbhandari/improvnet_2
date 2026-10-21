# V5: melody refinement, rhythm/harmony and stem addition

V5 has three active tasks and uses protocol 9. It starts from the original
`ar_context_split_instrument/best_model.pt` weights with a fresh optimizer and
scheduler. V4 checkpoints and data indexes are preserved; a V4 training state
cannot resume as V5. Tokenizer IDs remain fixed for AR initialization. This branch now runs V5
inference only; use the older branch for older checkpoints.

## Task sequences

| Task | Source sequence | Clean response | Requested share | Left history when available |
| --- | --- | --- | ---: | --- |
| `refine` | Corrupted monophonic skyline melody | Original monophonic melody | 20% | Monophonic target history, 75% chance |
| `harmonize` | Monophonic melody with 60–100% of onset/duration groups masked | Original pitched mix, including harmony and performed rhythm | 40% | Pitched polyphonic target history, 75% chance |
| `stem` | Original mix minus one instrument class, with mild timing/duration and accompaniment-pitch errors | Original complete mix, including the missing stem | 40% | Full polyphonic target history, 75% chance |

The serialized example is a task token, optional stem-class tokens, optional
left history, `<SEP>`, the source event sequence, `<SEP>`, `<S>`, and the clean
response ending in `<E>`. Genre and the target instrument mix are continuous
conditions; strength and visible rhythm statistics enter the condition encoder.
Only clean response tokens carry next-token prediction loss. No right-hand
melody/history prompt is supplied. Harmony density/attack-rate conditioning and
the orchestration task remain absent from V5.

Whole-example training transposition still shifts source and target together;
it is separate from source-only pitch corruption.

Skyline uses anchored groups of attacks within 50 ms to account for performed
chords. Chunking itself preserves complete exact-onset groups. The model still
uses the shared instrument/note/onset/duration representation at 10-ms precision.

## Refinement corruptions in detail

The clean skyline melody is copied before any source-only corruption. First,
all detected ornament spans are merged into one held note per span, regardless
of the requested strength. This pass is mandatory even when requested strength
is zero. If there are no candidates, it leaves the source alone.

Next, one additional edit is sampled independently of ornament presence. Its
default weights are below. An edit can do nothing when its selected span is
ineligible or no note passes its probability gate. A zero-strength example with
no ornaments remains clean. The clean target always retains its original notes.

| Additional edit | Weight | What happens to the source |
| --- | ---: | --- |
| Contour inversion | 40% | Reflect a short pitch contour around its first note |
| Ordinary merge | 10% | Merge one eligible 2–4-note nonornament run |
| Pitch errors | 25% | Shift a small subset of melody notes by a nonzero amount up to ±7 semitones |
| Timing errors | 25% | Nudge a small subset of melody onsets and durations by up to ±50 ms |

For each chunk, conditioned strength is the maximum of requested strength and
the fraction of original note attacks removed by mandatory ornament merging.
For example, merging four notes into one removes three of four attacks and
sets a minimum strength of 0.75. The additional edit uses the requested strength, independently of this floor.
Training and inference calculate the same value, recalculating it after any
chunk shrink. Inference reports both requested and effective strength. Thus an
ornament-heavy source is never labelled as uncorrupted simply because requested
strength was low. This is a severity floor, not a complete error-distance metric.

**Ornaments.** `ornament_spans` scans a single voice for runs of up to 12 notes,
using a fast-onset threshold of `1.25 * median IOI`, clamped to 90–220 ms.
Successive notes must keep the same instrument, move by at most two semitones,
and start no later than 60 ms after the preceding release. Candidates satisfy
one of these rules: a returning neighbour figure with at least three notes,
within four semitones, and a direction change; a chromatic run of at least three
notes with one-semitone steps and net movement at most seven semitones; or a
short grace-like lead-in whose final note lasts at least twice the first note.
These are conservative MIDI heuristics, not theoretical ornament annotations.
Every detected candidate is merged. Its first note's pitch/onset are kept and its duration extends to the latest release in the span.
For example, fast `C–D–C–D` becomes one held `C`; the target retains the figure.

**Contour inversion.** Attempts cover 3–10 consecutive notes, with roughly
`max(1, round(strength * note_count / 8))` attempts. Pitch is reflected as
`new_pitch = 2 * first_pitch - original_pitch`: `C4–D4–E4` becomes
`C4–Bb3–Ab3`. Onsets, durations and velocities stay intact. Overlapping attempts,
instrument changes, IOIs over one second, release gaps over 150 ms and
out-of-range reflected pitches cause an attempt to be skipped.

**Ordinary merging.** With a strength-dependent gate, choose one 2–4-note run
outside detected ornament spans. Notes must keep the same instrument, have
positive IOIs no greater than 800 ms, have no release gap greater than 150 ms,
and avoid leaps over an octave. Merge into the first pitch held through the
latest release. For example, a connected `C–E–G` can become a held `C` even when
it is not classified as an ornament. The clean target teaches restoration.

**Pitch errors.** When selected, sample `round(requested_strength * eligible_notes)`
notes without replacement, with at least one note for a positive request. Each
selected note moves by a nonzero integer offset from −7 to +7 semitones. Only valid MIDI pitches are sampled; there is no boundary clipping.
These intentional *refinement* errors are separate from the stem task's
accompaniment-only ±7-semitone policy.

**Timing errors.** Sample the requested fraction of exact attack groups without
replacement, rounded to a whole group, with at least one for a positive request. Selected onsets and note durations receive independent,
absolute offsets from −50 to +50 ms in 10-ms steps. This replaces proportional
duration scaling, which could exceed 50 ms on held notes. Simultaneous attacks
move together; distinct attack order, nonnegative onsets and positive durations
are preserved. Boundary constraints can reduce an onset offset to zero; selected notes still
receive a duration change when needed to avoid a complete no-op.

The weights describe edit choices, not percentages of notes or guaranteed
changed examples. The default is one extra edit; if increased, extra edit types
are sampled without replacement, so local pitch/timing errors cannot compound
through repeated selection. Current V5 does not split notes or mask the whole
melody in refinement. The source may lose attacks while the full clean target
remains, subject to the source/response token budgets. The retired probabilistic ornament selector and V2–V4 inference paths have been
removed from this branch.

## Balanced corruption levels and measurement

Refinement samples no additional edit in 5% of requests. The other 95% choose
10%, 20%, …, 100% with equal probability, so each nonzero level has 9.5% expected
exposure. Pitch and timing edits now act directly on that fraction of the
remaining notes/attack groups; they no longer multiply strength by 0.15.
Contour edits and ordinary merging retain their musical span constraints.

These are balanced **additional edit requests**, not a guarantee of uniformly
changed total sequences: compulsory ornament merging can raise the total, small
chunks require rounding, and structural edits may be ineligible. Mandatory
ornament merging still applies to the 5% without additional edits. Example
metadata includes `requested_strength`, effective `strength` and
`refinement_changed_fraction`. The latter counts clean notes whose serialized
instrument, pitch/velocity, onset or duration no longer match the source,
including removed notes; it is a note-level edit metric rather than a musical
distance score.

Training logs and TensorBoard report requested/effective means, actual changed
note fraction, and proportions of low-change (≤20%) and high-change (≥80%)
refinement examples. Harmonize intentionally stays at 60–100% timing masking;
stem pitch errors intentionally stay at 0–5%. These task-specific limits remain
as configured rather than receiving refinement's 10–100% rule.

## Stem denoising and melody protection

Drums are restricted to a drum-set subset of the
[General MIDI drum mapping](https://midi.org/midi-ci-profile-for-default-drum-note-map).
The policy is defined once in `melody_stems.py` and includes 21 sound IDs:

| Family | MIDI note IDs |
| --- | --- |
| Kick | 35, 36 |
| Snare variants and side-stick | 37, 38, 40 |
| Toms | 41, 43, 45, 47, 48, 50 |
| Closed, pedal and open hi-hat | 42, 44, 46 |
| Crash, ride, Chinese/splash cymbals and ride bell | 49, 51, 52, 53, 55, 57, 59 |

Handclaps, tambourines, cowbells, congas, bongos, shakers, whistles and other
auxiliary percussion are excluded. No auxiliary hit is remapped into a kick or snare. Extended electronic hi-hat
edge articulations are normalized first: 22→42 (closed) and 26→46 (open),
following the documented [Groove MIDI mapping](https://magenta.withgoogle.com/datasets/groove#drum-mapping).
This is an explicit ingestion policy for those two note IDs, not a claim that
JazzSAMBA documentation identifies the recording hardware. Coincident normalized
hi-hat/bow attacks collapse to one hit using the stronger velocity; onset timing
is unchanged. Other drum variants already inside the kit remain intact. Filtering happens before stem selection, local rhythm statistics,
chunking and prompt-history construction. Raw MIDI and parsing caches retain
the original data; task examples are filtered on the fly without an offline
corpus rebuild. Auxiliary-only drum tracks do not count as an eligible missing
drum stem: piano-only results route to refinement/harmonization without a parse
retry, while remaining multi-instrument mixtures can target another instrument.

Inference filters input and reference auxiliary percussion and masks unsupported
drum pitches before sampling; masking applies only to the `drum` instrument,
so pitched instrument ranges remain intact. Retained drum anchors use the same
policy. MIDI export continues to select program 0 on zero-based channel 9
(Standard Kit on channel 10). The model still predicts when and how strongly
to strike each allowed sound; drum rests remain legal. The policy and allowed
IDs are recorded in checkpoint metadata (data-policy revision 10).

The missing class is chosen from the clean mixture. When drums are available,
they are selected with 85% probability; other choices prefer supported jazz
instruments. Eligible windows contain both the target part and retained pitched
music. Piano-plus-drums is valid supervision; a piano-only source is useful for
refinement/harmonization and is routed there instead of causing a parse retry.

Retained source notes are corrupted before tokenization:

- Exact-onset chord groups receive a shared offset of at most ±50 ms. Distinct
  groups retain their order and remain inside the owned window.
- Each retained duration receives an independent change of at most ±50 ms,
  with a positive duration. Held notes can continue across a window boundary.
- Each example samples an accompaniment pitch-error probability between 0%
  and 5%. Twenty percent of examples explicitly use 0%; the other 80% sample
  uniformly in that interval. Selected notes shift by a nonzero amount of up
  to ±7 semitones, subject to chord-order and collision guards. Eligibility is
  computed with 50-ms skyline before timing corruption.
  Skyline melody pitches and percussion pitch IDs are protected. Changes avoid
  duplicate attacks and preserve retained pitch ordering within a chord.
- Instrument labels, velocity and retained note count stay unchanged.

The response is the original full mix, with clean source timings/pitches and the
missing part. The model therefore learns to correct the retained accompaniment
while adding drums, rather than learning only to copy its input.

Inference permits the same bounded timing/duration repairs. Each retained note
and instrument must survive; exact chord groups stay together, and owned chunk
boundaries remain intact. Melody pitches use the global input skyline and are
protected across chunk boundaries. Accompaniment pitch changes are limited to
±7 semitones, favor keeping the existing pitch, and have a conservative cap of
`floor(0.05 * eligible accompaniment notes)` per chunk.

Drum rests remain legal. If no requested stem is produced anywhere, the retained
mix is still saved and the pipeline completes; the console and `report.json`
show `missing_instruments` and `stem_summary`. Genuine decoding failures such as
losing a retained note still fail. Track `stem_target/drum` validation loss in
addition to full-response loss, which includes many retained notes.

## Prompt history

History always comes from the clean target domain before the current window:
monophonic for refinement, pitched polyphonic for harmonization, and the complete
mix for stem addition. A stem example can therefore include preceding drums in
history while its current source lacks drums. This models continuation of an
already generated part.

All three tasks show history with 75% probability when preceding music exists.
For 30% of shown training mix histories, about 10% of notes receive mild pitch,
addition, deletion or timing edits. Additions have twice deletion's weight.
History pitch edits and added pitched notes use valid nonzero offsets up to
±7 semitones. History onset/duration edits use absolute offsets up to ±50 ms
in 10-ms steps. Repeated edits are bounded against the original note rather
than accumulating larger errors. Percussion pitch IDs stay unchanged and onsets
stay before the current window. This history augmentation is separate from the
accompaniment-only stem source corruption. Refinement histories and all validation histories remain
clean. The history token cap is 2,048, retaining complete attack groups.

At generation, every task starts without history. Later chunks use only that
task's previous generated output: a refined melody for `refine`, the generated
pitched mix for `harmonize`, and the generated complete mix for `stem`. Classical
input is never inserted as an initial clean history prompt.

## Harmonize timing masks

The second task is named `harmonize` in data, logs, checkpoint metadata, CLI,
prompts and output filenames (`harmonize.mid`). It uses the already-reserved
`<HARMONIZE>` token, preserving every vocabulary ID. Unused historical tokens stay reserved so initialization never renumbers the
vocabulary; obsolete task paths are removed.

Training explicitly uses 100% timing masking in 25% of examples; otherwise it
samples uniformly between 60% and 100%. Both onset and duration attributes are
masked together for selected attack groups. Rounding is upward, so even tiny
chunks mask strictly more than half of their timings. Small chunks can become
fully masked more often due to this rounding. Pitch, velocity, instrument and
coarse five-second time-bin structure remain available; targets stay clean.

Inference enforces the same 60% minimum. `--rhythm-strength 1` masks all note
onsets/durations; the default is 0.8, and smaller values are clamped to the
checkpoint's minimum. Actual mask coverage is recorded as
`timing_mask_fraction` in example diagnostics and inference chunk reports.

## Shared rhythm guidance

Both harmonization and stem examples expose the same 20-feature rhythm schema in
75% of training examples. Features come from the clean local melody and an
inferred pulse: five IOI summaries, beat duration, bar-phase sine/cosine, swing
subdivision, expressive offset/spread, pulse confidence and eight beat-phase
histogram bins. They contain no pitches or exact note sequence.

During inference, one selected jazz reference and the source pulse/origin guide
both stages. Local windows follow the same clock and scale timing summaries to
the source pulse. Automatic references use training entries only, including the
existing PiJAMA bank. `--style-reference` supplies a MIDI instead;
`--no-rhythm-conditioning` disables both numeric controls and the soft onset
phase bias. Beat inference can have half/double-tempo ambiguity and assumes
four beats per bar; low-confidence phase bias is disabled.

## Data and on-the-fly preparation

`melody_manifest_v5.json` extends the existing composition-group index without
reprocessing the expanded original corpus. The original V4 manifest remains
unchanged. Existing MIDI parsing caches are reused. Skyline selection, task
choice, windowing, corruption, history selection and token budgeting happen in
the DataLoader. Only newly acquired performances are parsed once into additional
base records, not expanded into fixed task examples.

Every main traversal keeps all admitted jazz/blues files: four jazz/blues passes
plus eight rotating non-jazz slots per original jazz/blues record. With the
three-task weights, main refinement/harmonization slots account for 60% of requests
and separate eligible stem draws account for 40%. Thus the effective main-stream
exposure is 8:4 non-jazz:jazz/blues, while the non-jazz candidate multiplier stays
8. Within non-jazz stem draws, 80% prefer the original AR mixture pool. Batch
size stays 24/GPU, accumulation 1, with four workers/GPU and prefetch 1.

V5 adds 658 unique performance records (627 training, 15 validation, 16 test),
including 80 additional recorded piano/drum pairs in the training stem pool.

New data acquisition is recorded in `raw/acquisition_manifest.json`, including
source URLs, raw file inventories, licenses and author split information.
[New performance MIDI details](jazz_performance_v5.md) records which files are
admitted after overlap checks. Recorded JazzSAMBA piano/drum MIDI is used as
clean paired supervision; its derived bass/wind estimates are kept separate.
Jazz Trio Database MIDI is automatically transcribed performed piano, not
recorded keyboard MIDI and not a complete piano/bass/drum note mix.

Official JazzSAMBA song splits are locked, including alternate takes. Additional
copies of existing prepared performances are excluded by hashes/full-melody
fingerprints. Recording metadata is also checked for JTD overlap because
independent transcriptions need not have identical note timings. No author
validation/test song is promoted into training.

## Budgets and inference modules

Sources and refinement responses are capped at 512 tokens. Mix responses and
history are capped at 2,048 tokens. Separators and the two continuous condition
positions fit inside 8,192 positions. Oversized windows shrink by complete onset
groups. Addition/removal corruptions are checked against the resulting source
and clean target budgets rather than assumed to preserve length.

The canonical command is now `improvnet.inference.melody_inference`, with ready
tasks batched automatically on one GPU/model. V5 runs
`refine -> harmonize -> stem`; a missing-stem stage is omitted when not requested.
The old `parallel_inference` module is a thin compatibility alias. The main
class handles chunk/history preparation; focused helpers own the CLI, grammar,
batch scheduler, bounded stem repairs and output paths. Retired register reconstruction and instrument relabelling are removed. Active
stem-selection helpers live in `melody_stems.py`; missing-stem supervision is
preserved.

Batching preserves each stage's ordering and generated history. Variable-length
caches have independent positions and attention masks. The scheduler starts a
downstream task once its source chunk and safe lookahead are available. Up to
three V5 stages can normally overlap; startup, drain and short pieces offer
less overlap. One persistent inference autocast context reuses weight casts.
Per-task fixed random streams make scheduling independent of random draws;
batched floating-point kernels can still differ slightly by batch shape.

Progress always appears. GPU is used when available, otherwise CPU; seed 42 and
top-k 50 stay internal. With `--output clair.mid`, intermediate files are
`clair/melody.mid`, `clair/refine.mid`, `clair/harmonize.mid`, `clair/stem.mid` and
`clair/report.json`. The final MIDI remains `clair.mid`. The report includes
budgets, rhythm controls, repair counts, missing stems and observed batch sizes.

## Run paths and review status

- Checkpoints: `/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/melody_refinement_v5`.
- Data manifest: `/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/jazz_blues_data/prepared/melody_manifest_v5.json`.
- Logs: `/e/home/jusers/bhandari1/jupiter/run_scripts/improvnet/melody_run_logs_v5`.
- Launcher: `/e/home/jusers/bhandari1/jupiter/run_scripts/improvnet/melody_single_node_launch.sh`.

Pre-kit job `2431554` was cancelled and its run directory archived as
`melody_refinement_v5_pre_kit_2431554`. Fresh kit-only job `2432602` was submitted
on 2026-10-21 from the original AR checkpoint, with automatic continuation enabled. Prior runs/checkpoints are retained. V5
musical quality still needs evaluation after training; tests validate the data,
masking, budgets, decoding constraints and resume/initialization behavior.

After training:

```bash
cd /e/home/jusers/bhandari1/jupiter/improvnet_melody
python -m improvnet.inference.melody_inference \
  --model /e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/melody_refinement_v5/best_model.pt \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --output improvnet/inference/clair.mid \
  --task pipeline --genre jazz --melody-strength .2 --rhythm-strength .9 \
  --instruments "Acoustic Piano" drum
```

Current V5 tests: `python -m pytest -q` (defaults to `tests/melody`). Six
feature-based modules cover training/resume, data/splits, corruption, MIDI
decoding, groove/memory bounds and batched inference. Shared fixtures live in
`melody_test_helpers.py`; obsolete orchestration and old-protocol tests are
removed. Tests for other models remain outside the V5 default suite.
