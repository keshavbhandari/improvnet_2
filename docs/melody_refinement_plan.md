# V15 melody refinement, harmonization and stem addition

V15 initializes from the original pretrained AR model. The two contrastive
encoders stay frozen. There are no beat-marker tokens and no cross-attention
modules in the decoder. Previous finetuning checkpoints are incompatible;
checkpoint protocol 19 rejects them explicitly.

## Model and context

The input layout is:

```
[Genre] [32 tokens: window 1] [32 tokens: window 2] [Instrument mix] [Task + history + input] -> Response
```

A small learned resampler combines the frozen style/beat features, relative-time
coordinates and that window's projected drum profile into 32 vectors of width 256.
The same resampler processes each window independently, producing 64 ordered
memory tokens in total. Features and statistics cannot mix between the two
32-token banks before the decoder. A learned bias-free projection maps them to decoder width 2048. The original AR decoder
blocks consume the entire sequence through causal self-attention; their weights
and structure are reused unchanged. Reference tokens appear only during prefill
and remain in the ordinary self-attention K/V cache thereafter. There are no
separate reference K/V caches, per-layer gates or decoder adapters.

All tasks reserve 66 conditioning positions: genre, 64 memory tokens and mix.
All three tasks now populate the reference prefix during training. Refinement
uses exact output-instrument references, normally solo piano. Music logits
exclude the entire prefix and align with the normal shifted labels. Loss applies
only to response tokens. Both frozen encoders remain immutable.

The combined context, including these 66 positions, is at most 8192 tokens.
Source chunks contain at most 512 tokens and history at most 2048. Complete target
responses are retained. History is shortened first; if source plus the complete
response alone cannot fit, the whole time interval is shortened at an onset-group
boundary. Inference also reclaims old history by rebuilding the cache when a
response needs more space. No partial chord or target tail is silently trimmed.

## Training sequences

1. **Refine:** a monophonic skyline melody is corrupted and reconstructed.
   Detected ornaments are merged first, then one additional contour, ordinary
   merge, pitch or timing edit is sampled. Additional severity is balanced across
   10–100%, with 5% clean additional-edit cases. The conditioning strength also
   accounts for the mandatory ornament merges. Monophonic history is shown 75%
   of the time. After corruption, every current-source pitch/velocity token keeps
   its pitch but is assigned velocity **90**, including any added or merged notes.
   Targets and history retain their expressive velocities. After those edits,
   onset and duration are masked together for **90–100% of the corrupted melody's
   attack groups**; 25% of training draws mask every group. The source keeps its
   pitches, instruments and velocity-90 note tokens. Every `<T>` one-second
   time-advance token is removed from this current source; targets, history and
   reference windows retain them.
   Both frozen encoders supply a safe consecutive reference pair from the same
   output instrument(s), with exact matching even for non-piano instruments.
   The response reconstructs the clean melody, including rhythm and expression.
   Masking and velocity normalization are separate from melodic edit severity.
2. **Harmonize:** a monophonic melody has onset and duration attributes masked
   together for **90–100% of attack groups**. A quarter of training draws mask
   all groups; other draws sample uniformly from 90–100%, rounded upward to whole
   groups. The response is the clean pitched mixture. Polyphonic history is
   shown 75% of the time. Both frozen reference encoders condition every eligible
   example.
3. **Stem:** one instrument is removed from a genuine multi-instrument mixture.
   The response is the complete clean mix, including the missing instrument.
   Drums have priority; pitched stems remain eligible. The current source is
   uncorrupted (`STEM_SOURCE_CORRUPTION_PROB=0`). Polyphonic full-mix history is
   shown 75% of the time, and both reference encoders condition every eligible
   example. Full percussion training targets are preserved. Naturally kit-only
   drum windows are preferentially sampled without stripping other windows.

For harmonize/stem, supplied history receives robustness edits in 30% of cases,
covering about 10% of its notes. Additions have twice the deletion weight; pitch
errors are bounded by +/-7 semitones and onset/duration errors by 50 ms on the
10 ms tokenizer grid. Source/history edits never alter response labels.

## Consecutive reference pair and key augmentation

Training chooses **two consecutive** instrument-matched windows of at most 512
tokens each, within 30 seconds of the target. Both come from the same side of
the target: before or after. The pair need not touch the target. Every onset
group is complete and no group is skipped between the two windows. A note held
across their shared boundary retains its duration; neither reference may contain
a target-overlapping note. A 250 ms guard separates the pair from target attacks
and held-note releases. All tasks require a safe pair at `REFERENCE_PROB=1`.
If a mixture task is ineligible, another eligible task on the piece can be tried;
if refinement also has no safe pair, the sampler tries another piece. It never
uses the target as a reference or silently trains unconditioned refinement.

Refinement always uses the exact target instrument set. For harmonize/stem,
reference modes remain 50% exact target instruments and 50% their
piano/drums/acoustic-bass subset. Both encoders receive the **same pair of time
intervals, instruments and note timings**. Training samples one pitch shift per
encoder, independently between encoders but **shared across its two windows**.
The shift is within +/-7 semitones and respects the combined pitch range of both
windows. Nonzero shifts are preferred when legal. Drum pitches remain unchanged.
Validation uses fixed, untransposed reference views.

Each encoder returns its global vector and eight relative-time-bin vectors for
each window. Each window has its own 20-vector feature bank: two global summaries
plus two sets of nine features. Empty bins are masked. The two banks are encoded
as separate attention batch rows, then concatenated in chronological order:
**first 32 slots = window 1; next 32 slots = window 2**. Both are projected from
256 to 2048 dimensions and inserted as the causal decoder prefix.

For stem examples, each drum profile describes its own **reference window**,
rather than the target or the other reference window. It carries coarse sound
presence/usage, attack rate, simultaneous density, coverage and inter-onset
statistics. Harmonization/refinement have zero drum profiles. Statistics remain
soft conditions.

## Genre sampling curriculum

The existing manifest and held-out splits are reused. The non-jazz/blues candidate
pool stays at 2x. No new corpus preprocessing is required.

- Through optimizer step 10,000: jazz/blues weight 1, without oversampling.
- Steps 10,000–15,000: increase that weight linearly from 1 to 4.
- From step 15,000 onward: retain weight 4, matching the previous final balance.

These are relative file-draw weights; with non-jazz weight 2, the requested
jazz/blues proportions are approximately 1/3 initially and 2/3 finally. Eligibility
routing can change the per-task exposure. Every file stays eligible. Fixed-length
sampling epochs count candidate draws rather than promise literal single file
traversals during the changing curriculum. Validation remains fixed.

Planned optimizer-step numbers are attached to sampler indices before worker
prefetch. Resume restores the same shuffled indices and curriculum phase, with
accumulation accounted for. Existing retry/exposure text logs remain available;
no new dashboard cards are added.

## Inference

```
python -m improvnet.inference.parallel_inference \
  --model /e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/melody_refinement_v15/best_model.pt \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --output improvnet/inference/claire_v15.mid \
  --task pipeline --genre jazz \
  --melody-strength .4 --rhythm-strength .95 \
  --instruments "Acoustic Piano" drum \
  --reference-description slow
```

`--reference-aggregation window` is the default: choose a reproducibly random
eligible **consecutive pair**, shared by both encoders. Each of its two windows
must contain the requested reference instruments. A reference too short to
supply a complete pair is rejected with a clear error.
`--reference-aggregation average` averages features at corresponding positions
across all eligible consecutive pairs: first windows form bank 1, second windows
form bank 2. Profiles are averaged separately in the same way; the two banks are
never collapsed into one. Pitch augmentation is not applied during inference.
Stage-specific instrument views are cached: harmonization uses the requested
pitched instruments, drum addition the matching mixture, and later bass addition
its matching mixture.

Refinement automatically normalizes the current corrupted input velocities to
90. `--melody-strength` controls the melodic edits; the existing `--rhythm-strength`
now controls timing masking for both refinement and harmonization, with a 90%
floor (default 95%; 1 masks all onset/duration groups). The same selected reference
piece provides the exact instrument view for refinement and the matching views
for subsequent stages. Piano-only refinement and piano-only harmonization reuse
the cached piano reference pair. Standalone `--task refine` also prepares reference
conditioning, using either descriptor selection or a supplied MIDI.

Generated expressive velocities and predicted timings are retained when output
becomes history or input to later stages. Both the standalone and batched paths
use the same corruption/masking logic. Refinement respects its chunk's onset
window when decoding. An explicit `--melody-strength 0` continues to bypass the
refinement model, timing masking and refinement reference preparation, preserving
the original skyline. Other requested stages still use their references.

Each soft drum profile follows its own selected reference window. The hard inference
palette remains the standard-kit sounds present in the supplied reference piece,
with aliases 22/26 normalized to closed/open hi-hat. No new palette argument is
required. Retained source notes are preserved by stem decoding.

Both inference entrypoints share the same continuously batched pipeline. CUDA
inference keeps a persistent mixed-precision context and selects Flash/efficient
attention. The finetuning job also avoids cuDNN attention and enables expandable
allocator segments. The existing batch size remains 24 per GPU, accumulation 1.

## Reference selection

There are two mutually exclusive ways to select a reference piece:

```
--reference-description slow
--reference-description "slow balanced"
--style-reference /path/to/jazz_trio.mid
```

The first two lines illustrate the same descriptor option. Use either a
description or a MIDI filepath in a command. If neither is given, description
`any` samples a compatible trio. The former reference-ID and embedding-bucket
selection options are removed. Both inference entrypoints share this interface.

Supported descriptors:

| Dimension | Descriptors | Meaning |
| --- | --- | --- |
| Tempo | `slow`, `moderate`, `fast` | Estimated pulse below 90, 90 to below 140, or at least 140 BPM |
| Piano activity | `sparse`, `balanced`, `dense` | Below 2, 2 to below 4, or at least 4 piano attack groups per second |
| Unrestricted | `any` | Any compatible verified trio |

One tempo and one activity descriptor can be combined in either order. `medium`
is accepted as `moderate`, and `busy` as `dense`. Invalid/conflicting descriptions
fail before checkpoint loading. If a valid combination has no compatible piece,
the error lists available combinations. Selection never relaxes a requested
category silently.

All descriptors describe the **whole source piece**, containing exactly one
piano instrument, drums and actual acoustic/double bass. Piano activity counts attack
groups rather than chord voices. Estimated pulse can have half/double-tempo
ambiguity. Descriptors choose the reference; they do not impose an exact tempo
on generation. The actual conditioning still uses a random consecutive pair,
or corresponding-position averages when averaging is requested.

The active file is `improvnet/inference/reference_catalog.json` (format v3).
It contains **129 exact-trio pieces: 129 jazz and 0 blues**, expanded from the 18
previously eligible entries by adding 111 training-fold jazz recordings. Sources
are Doug McKenzie (123) and GigaMIDI (6). Piano types are acoustic (121) and
electric (8); every piece also has drums and actual acoustic/double bass.

The complete instrument set must be **exactly three types**: one supported piano,
drums and acoustic/double bass. Extra instruments, including saxophones, brass,
guitars and additional piano types, make a piece ineligible. This requirement is
checked by the catalogue builder, loader and actual-note selection guard. Custom
`--style-reference` MIDI files follow the same rule. Source MIDI files are
preserved. Training examples and model checkpoints are unchanged.

The training-pool search covered 1,289 blues-labelled records. None had exactly
the trio in its indexed instrument set. The 15 blues candidates containing all
trio roles were inspected and each also had additional instruments, so none was
added. The original AR mixture index had no additional jazz/blues-labelled rows.
Unknown-genre pieces were not relabelled as blues.

Current tempo coverage: **33 slow, 53 moderate, 43 fast**. Piano activity coverage:
**84 balanced and 45 dense; no sparse entries**. Use, for example,
`--reference-description slow` or `--reference-description "slow balanced"`.
`--genre blues` and the `sparse` descriptor report no available matches with this
catalogue. Matching never relaxes those requests silently.

Trio mode allows a piano-family substitution; exact mode requires the specified
piano name. The same stage-specific instrument matching applies to descriptor
selection, with fallback across compatible sources if a piece has no usable local
pair. An explicit reference file gets a clear error if it contains extra
instruments or cannot provide the requested views.

The inference report records the requested description, matched labels, estimated
tempo/activity and selected piece path. Internal IDs remain for tracing reports;
they are not a selection option. Embedding-bucket construction and selection are
removed from the catalogue maintenance code.

To extend the catalogue with more verified training-fold pieces:

```
python -m improvnet.scripts.melody_style_references --add-jazz-trios 32
python -m improvnet.scripts.melody_style_references --add-blues-trios 32
```

Each command adds up to the requested number from the available genre-specific
training pool. Existing pieces are preserved; new pieces must pass actual trio,
standard-kit and reference-pair checks before whole-piece descriptors are added.
The default rebuild also admits only exact three-instrument trios and allows a
genre to be absent when no eligible pieces exist.

Validation on 18 November 2026: 305 CPU melody tests passed (one CUDA-only test
skipped). Every retained piece passed frozen-encoder pair checks for all three
stage views. The full source MIDI files were also audited for exact trio
instrumentation. Real descriptor-selection checks verified exact source
instrumentation and explicit failure for unavailable genre/activity requests.

## Training and cleanup

The sole finetuning config is `improvnet/model/melody_config.py`; the canonical
one-node/four-GPU launcher is `run_scripts/melody_launch.sh`, with synchronized
checkpointing and automatic resubmission near its 12-hour limit. V15 outputs go
to `melody_refinement_v15`, initialized from `ar_context_split_instrument`.

V15 uses a fresh run directory and retains V14. Older melody-finetuning
checkpoint directories are removed; user-generated MIDI outputs are preserved. Pretrained AR and reference-encoder checkpoints are
reused. Batch size remains 24 per GPU with accumulation 1. The canonical launcher starts V15 from the original AR weights.


## V15 change and validation

V15 removes every `<T>` one-second time-advance token from the **current
refinement input only**, after applying the existing V14 melody corruptions,
velocity-90 normalization and 90–100% onset/duration masking. Targets, prompt
history, reference windows, harmonization and stem inputs keep their normal
time tokens. The model must generate the time-advance tokens in its response.
Zero-strength refinement remains an unchanged pass-through.

Architecture, frozen encoders, double reference windows, batch size and all other
V14 training settings are unchanged. Protocol 19 records this new input contract
and prevents a V14 checkpoint from silently using V15 inference preparation.

A 36-example real-data audit passed across train/validation/test and all tasks.
It verified that refinement sources contain no `<T>`, both safe instrument-matched
reference windows remain present, and all contexts fit the 8192-token limit.
The maximum observed response was 1990 tokens and context was 4610 tokens.

On 19 November 2026, the final suite passed: 311 tests, one CUDA-only skip.
V15 was submitted as Slurm job **3015487**, one node/four GPUs, using
`run_scripts/melody_launch.sh` with automatic checkpoint/resubmission enabled.
Removed only the old V12/V13 melody-finetuning run directories; V14 best/latest
and all other project checkpoint directories remain intact.
