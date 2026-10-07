# Hierarchical instrument roll + accompaniment events

Branch: `hierarchical-pianoroll-tokenizer`. This is a structured encoding, not a
learned VAE compression stage. No model architectures or checkpoints are changed.
Legacy event tokenizers remain in `improvnet/archive/tokenizer_legacy`, with the
existing public compatibility imports used by trained AR/diffusion/metric models.

## Time hierarchy and channels

Inspired by the three information channels in [Huang et al., §5](https://arxiv.org/html/2402.14285v4):
velocity, onset and sustain pedal. We borrow the representation idea, not their
learned VAE or diffusion implementation. Our pedal array does **not** duplicate
the controller across every pitch row.

Default `coarse_ms=1000`, `fine_ms=10`: each coarse bin contains 100 fine frames.
Both values are configurable positive integers; `coarse_ms` must be divisible by
`fine_ms`. Absolute time is `(bin * frames_per_bin + offset) * fine_ms` in ms.
This is an absolute-time grid, not beat quantization, preserving swing/rubato to
the chosen fine resolution. Quantization uses nearest-frame, half-up rounding.

| Array | Shape | Meaning |
|---|---|---|
| `velocity` | `[bins, 128, frames_per_bin]` | uint8 0–127; attack velocity throughout key-down duration; zero inactive |
| `onset` | same | uint8 0/1; explicit attacks, including identical-velocity reattacks |
| `pedal` | `[bins, frames_per_bin]` | uint8 0–127; full CC64 controller state |
| `valid_mask` | `[bins, frames_per_bin]` | bool; real silence is valid, trailing padding is not |
| `coarse_velocity` | `[bins, 128]` | maximum fine velocity per pitch/bin |
| `coarse_onsets` | `[bins, 128]` | sum of fine onsets per pitch/bin |
| `coarse_pedal` | `[bins]` | maximum fine pedal value per bin |

There is no lane dimension, HOLD/OFF vocabulary, or generated event-token grammar
in the piano representation. Pedal sustains sound, **not** key-down duration: it
never extends note releases in `velocity`. Pedal state carries across bins;
decoding closes any still-held controller at the end. `--no-pedal` instead keeps
selected-channel CC64 messages in the frozen sidecar and leaves `pedal` zero.

Coarse summaries are derived from the fine arrays and are not invertible. Saving,
loading and decoding verify exact consistency. After editing fine arrays, call
`piece.refresh_coarse()` before validation/save/decode. The optional `image()`
display view is `[bins, 3, 128, frames_per_bin]`; only this view broadcasts pedal.
Modifying that view does not change the stored arrays.

## Target selection and same-pitch normalization

The canvas is a **single instrument part**, not necessarily piano:

- If piano notes exist, prioritize the non-drum channel with the most notes using
  GM piano-family programs 0–7 (ties: lowest channel). Within that channel choose
  the most note-rich piano program (ties: lowest program).
- Otherwise randomly choose one populated `(channel, program)` part, uniformly
  over parts rather than notes. Drums are eligible, including drum-only files.
  `--seed` makes the fallback reproducible; without it, selection is random.
- `--target-channel 0..15` explicitly overrides piano priority. If that channel
  has several programs, select its most note-rich program or specify
  `--target-program 0..127`. Channels/programs are zero-based; drums use channel 9.
- Legacy `--piano-channel` still explicitly selects piano only and cannot be
  combined with the new target options. `--piano-programs` controls piano-family
  detection. A target override must have valid notes; empty MIDI still produces
  an empty canvas with `target_present=False`.

Tracks belonging to the selected channel/program combine into one roll track.
All other notes, including other programs on that same channel, remain context.
`target_channel`, `target_program`, `target_kind` (piano/pitched/drums),
`target_is_drum`, `target_is_piano`, `target_selection` and `selection_seed` record
the choice. This identity must condition a future multi-instrument generator;
the tokenizer stores it but does not change model architectures.

Velocity/onset work unchanged for strings, winds, etc. Non-piano targets have
`pedal_enabled=False` and a zero pedal array, including coarse pedal summaries.
Original non-piano CC64 stays in the frozen sidecar, not in the generated canvas.
Other expressive controllers likewise remain sidecar information.

For General MIDI drums, rows are **drum note IDs**, not melodic pitches; the whole
kit is the selected part. Onsets retrigger hits; velocity preserves hit strength.
Key-down durations are stored without pretending to describe acoustic decay.
Decoding preserves channel 9 (displayed as channel 10 in MIDI applications).
The same overlap/collision policy applies, not a special lossless representation
of overlapping drum voices. See the [General MIDI percussion map](https://www.cs.cmu.edu/~music/cmp/archives/cmsip/readings/GMSpecs_PercMap.htm).
Detection follows the file's channel/program, not its filename or enclosing
dataset directory. Some GigaMIDI `drums-only` files actually use other channels
and programs; their intended percussion identity cannot safely be inferred from
the name alone. Audit/remap such non-standard files before building a corpus.

One active note per pitch is represented. The deterministic policy is:

1. Order attacks by original onset time, then stable source MIDI event order.
2. A new attack ends the previous active note of that pitch. Its old tail never
   resumes. The new attack keeps its own velocity and release.
3. If several attacks quantize to one pitch/frame, retain the latest original
   attack; exact-time ties retain the greatest source event order. Do not choose
   maximum velocity or extend to the displaced note's release.
4. Quantize releases and give each surviving note at least one frame, never past
   the next surviving attack of that pitch.

`metadata.normalization` reports `overlap_normalizations` (shortened original
releases), `same_frame_collisions` (discarded attacks), `collision_frames` (number
of affected pitch/frame cells), and `short_notes_extended` (surviving roll notes
extended to one frame). Overlap and collision counts can refer to the same source
attacks; they are not mutually exclusive. `source_target_notes` and `target_notes`
report pre/post normalization counts. Legacy `source_piano_notes`/`piano_notes`
metadata is retained, with zero counts for a non-piano target. Accompaniment short-note extensions have a
separate `context_short_notes_extended` count.

Thus overlapping MIDI notes and quantized attack collisions are explicitly lossy,
not exactly preserved. No originals are hidden in metadata for target playback.
Raw overlapping MIDI note-offs are paired FIFO on channel/pitch before applying
this policy; nested unison MIDI notes do not identify which attack an off belongs
to. Normalization applies to the durations established by that deterministic
reader, not to an inferred physical fingering.

## Decoding and accompaniment

Decoding scans each pitch's fine frames continuously across all bins. A zero
velocity releases it; an onset retriggers it even when the velocity is unchanged.
Coarse-bin boundaries do **not** split held notes. Strict mode checks that each
positive run begins with an onset, each onset has positive velocity, and velocity
remains constant until another onset or release.

For future generated arrays, non-strict decoding deterministically infers a
missing onset on an inactive-to-positive transition, ignores zero-velocity
attacks, and ignores mid-note velocity drift until an explicit onset or release.
Use strict mode for clean-data round-trip tests; this fallback is not a training
representation or a guarantee of musical quality.

Each non-target note remains a structured event row:

```text
track, channel, GM_program, is_drum, pitch, velocity,
onset_bin, onset_offset, end_bin, end_offset, stable_source_order
```

All parts share the same clock, with half-open intervals `[start, end)`. Quantized
start/end positions can cross any number of bins. Channel 9 drums retain their
pitches, velocities and durations. Events are onset-sorted;
`event_ptr[b:b+2]` gives the onset-indexed slice. `context_for_bin(b)` also includes
notes carried from earlier bins, excludes those ending exactly at the bin start,
and includes those starting exactly at its start. Same-pitch accompaniment
overlaps remain separate event rows; there are no piano lanes for accompaniment.

Tempo, signatures, track names, programs, bends and other controllers stay in a
frozen sidecar. Channel messages are quantized to the same fine clock; tempo/meta
event times are retained. No source target note messages are stored/replayed there.
Accompaniment track/channel/program identity is retained. Exotic bank/port or
mid-channel program-changing datasets still need
auditing before using their sidecars as future model inputs.

## Format, limits and compatibility

NPZ format **`hierarchical_piano_events_v2`** stores the nine arrays in the table
and accompaniment description (`velocity`, `onset`, `pedal`, `valid_mask`,
`coarse_velocity`, `coarse_onsets`, `coarse_pedal`, `events`, `event_ptr`), plus a
scalar JSON `metadata` header containing format/config/source metadata. No pickle
or lane fields. Previous lane-based v1 NPZ files are rejected with a re-encode
message: regenerate them from the original MIDI. Existing artifacts are not
overwritten. This format change does not alter legacy trained-model tokenizers.
The instrument extension adds JSON selection metadata only: shapes and v2 array
semantics are unchanged, and existing piano v2 files still load/decode.

Only trailing incomplete frames are padded, all with zero velocity/onset/pedal.
Output length comes from `valid_mask`, not an end token. A future model must
predict/request length separately. Four minutes still has 24,000 fine frames and
240 coarse bins: grouping is **not** compression or a global-attention shortcut.

Default timing error is ±5 ms plus small output-tick rounding, except explicit
overlap shortening and short-note extension. Output PPQ is an integer multiple
of source PPQ, raised to at least 9600. Controllers changing in the same fine
frame collapse to their final state; short pedal pulses may disappear. This is
performance reconstruction, not byte-identical MIDI restoration. All 128 pitches
and nonzero attack velocities 1–127 are supported.

Velocity-zero note_on is note_off. Nonpositive-duration note pairs are skipped
and reported as `skipped_nonpositive_notes` in metadata and the round-trip summary.
Zero-time on/off pairs can occur in otherwise readable MIDI but have no key-down
span for this grid. Negative delta times and dangling note-ons remain errors:
silently changing the event timeline or inventing durations would be unsafe.
Training explicitly opts into `recover=True`, which also ignores malformed
SMPTE-offset metadata, pitch-bend messages and channel-volume CC7 values (along
with the existing key-signature, bank-select and SysEx recoveries). Valid messages
are preserved, event timing is retained, and source files are never rewritten.
These fields are not represented in the relative PPQ note/pedal canvas; invalid
note pitches, attack velocities, sustain values and truncated events remain errors.
Unmatched note-offs are counted/ignored. Only synchronous PPQ
MIDI types 0/1 are supported. Configured duration/memory limits prevent giant
allocations. Old genre/continuity models still require their event representation
or an adapter, not this canvas directly.

## Try a round trip

From the repository root with the `improvnet` environment:

```bash
python -m improvnet.tokenizer.roundtrip \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid

python -m improvnet.tokenizer.roundtrip \
  --input improvnet/inference/examples/pop909_001.mid --piano-channel 2 \
  --coarse-ms 1000 --fine-ms 10

# Non-piano fallback, reproducible choice; no piano means another real part.
python -m improvnet.tokenizer.roundtrip --input piece.mid --seed 42

# Explicit target override, including drums; channel/program are zero-based.
python -m improvnet.tokenizer.roundtrip --input piece.mid --target-channel 9
python -m improvnet.tokenizer.roundtrip --input piece.mid \
  --target-channel 2 --target-program 40
```

Default output: a unique scratch `artifacts/hierarchical_tokenizer/roundtrip-*`
directory with `encoded.npz`, `reconstructed.mid`, `target_only.mid`, and a small
`summary.json`. `--output-dir` requires a new/empty directory. The script prints
normalization counts, comparison to the raw source, and comparison to the
documented normalized/quantized reference. It checks the latter's note identities
and timing within output-tick tolerance, instead of treating deliberate discarded
attacks or shortened overlaps as a codec bug.
Piano targets also retain the legacy `piano_only.mid` output. Summary fields name
the selected instrument, its source/normalized target counts, and pedal status.
Raw-source timing errors are `null` if note counts differ, since notes cannot be
paired one-to-one after discarded attacks; the normalized comparison remains
fully checked.

```python
from improvnet.tokenizer import GridConfig, HierarchicalTokenizer, EncodedPiece

codec = HierarchicalTokenizer(GridConfig(coarse_ms=1000, fine_ms=10))
piece = codec.encode("piece.mid", seed=42)  # Piano first; random fallback.
# Explicit override: codec.encode("piece.mid", target_channel=2, target_program=40)
piece.save("piece.npz")
piece = EncodedPiece.load("piece.npz")
codec.decode(piece, strict=True).save("roundtrip.mid")
conditioning_notes = piece.context_for_bin(0)

# After changing velocity/onset/pedal:
piece.refresh_coarse().validate()
```

Run the tokenizer tests (CPU only):

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests -p test_hierarchical_tokenizer.py -v
```

## Unpack data (unchanged)

```bash
python -m improvnet.scripts.extract_midi_archives \
  --input-dir /e/scratch/e-dev-2026d09-047/bhandari1/improvnet/data \
  --output-dir /e/scratch/e-dev-2026d09-047/bhandari1/improvnet/data/extracted \
  --workers 2 --nested-workers 2
```

Original archives are retained. Nested archives expand beside their file into
`*.unpacked`; completed collections have an `.improvnet-extraction.json` marker.
Reruns skip completed collections and verify incomplete members before resuming.
Symlinks/special files are skipped; unsafe paths/conflicts rejected. Default
safety limits: depth 8, five million files, 512 GiB. Extraction retains upstream
splits and does not tokenize, deduplicate or build model datasets.
