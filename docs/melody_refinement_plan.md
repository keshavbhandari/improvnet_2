# V12 melody refinement, harmonization and stem addition

V12 initializes from the original pretrained AR model. The two contrastive
encoders stay frozen. There are no beat-marker tokens and no cross-attention
modules in the decoder. Previous finetuning checkpoints are incompatible;
checkpoint protocol 16 rejects them explicitly.

## Model and context

The input layout is:

```
[Genre] [32 reference tokens] [Instrument mix] [Task + history + input] -> Response
```

A small learned resampler combines the frozen style/beat features, relative-time
coordinates and a projected drum profile into 32 vectors of width 256. A learned
bias-free projection maps them to decoder width 2048. The original AR decoder
blocks consume the entire sequence through causal self-attention; their weights
and structure are reused unchanged. Reference tokens appear only during prefill
and remain in the ordinary self-attention K/V cache thereafter. There are no
separate reference K/V caches, per-layer gates or decoder adapters.

All tasks reserve 34 conditioning positions: genre, 32 memory tokens and mix.
Refinement has no reference and uses zero-valued memory positions; retaining
these slots keeps the batch and cache layout uniform. Music logits exclude the
entire prefix and align with the normal shifted labels. Loss applies only to
response tokens.

The combined context, including these 34 positions, is at most 8192 tokens.
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
   of the time. Reference encoders are not used for this task.
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

## Shared reference window and key augmentation

Training chooses **one** nearby, instrument-matched window of at most 512 tokens,
within 30 seconds of the target. It cannot overlap target attacks or held-note
releases, and a 250 ms guard separates it from the target. If no safe window
exists, the same piece can route to refinement; the target is never substituted
as its own reference.

Reference modes remain 50% exact target instruments and 50% their
piano/drums/acoustic-bass subset. Both encoders receive the **same time interval,
instruments and note timings**. During training, pitched notes are independently
transposed by up to seven semitones for each encoder, choosing nonzero shifts
when MIDI bounds permit. Drum pitches are unchanged. This discourages key-based
shortcuts; it does not mathematically remove every pitch cue. Validation uses
fixed untransposed reference views.

Each encoder returns its global vector and eight relative-time-bin vectors.
The bank has 20 vectors: two view summaries plus two sets of nine features.
Masks handle empty bins. The resampler always outputs 32 memory vectors.

For stem examples, the drum profile describes this **reference window**, rather
than the clean target. This aligns its training and inference source. It carries
coarse sound presence/usage, attack rate, simultaneous density, coverage and
inter-onset statistics, not target note timings. Harmonization/refinement have
zero drum profiles. Statistics remain soft conditions.

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
  --model /e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/melody_refinement_v12/best_model.pt \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --output improvnet/inference/claire_v12.mid \
  --task pipeline --genre jazz \
  --melody-strength .4 --rhythm-strength .95 \
  --instruments "Acoustic Piano" drum \
  --style-reference /path/to/jazz_piano_and_drums.mid
```

`--reference-aggregation window` is the default: choose one random eligible
excerpt, shared by both encoders. The fixed internal seed makes this reproducible.
`--reference-aggregation average` averages global and relative-bin features over
all eligible excerpts before constructing the same 32-token prefix. No pitch
augmentation is applied during inference. Stage-specific instrument views are
cached: harmonization uses the requested pitched instruments, drum addition the
matching mixture, and later bass addition its matching mixture.

The soft drum profile follows the selected reference view. The hard inference
palette remains the standard-kit sounds present in the supplied reference piece,
with aliases 22/26 normalized to closed/open hi-hat. No new palette argument is
required. Retained source notes are preserved by stem decoding.

Both inference entrypoints share the same continuously batched pipeline. CUDA
inference keeps a persistent mixed-precision context and selects Flash/efficient
attention. The finetuning job also avoids cuDNN attention and enables expandable
allocator segments. The existing batch size remains 24 per GPU, accumulation 1.

## Training and cleanup

The sole finetuning config is `improvnet/model/melody_config.py`; the canonical
one-node/four-GPU launcher is `run_scripts/melody_launch.sh`, with synchronized
checkpointing and automatic resubmission near its 12-hour limit. V12 outputs go
to `melody_refinement_v12`, initialized from `ar_context_split_instrument`.

Unused decoder cross-attention code and its old tests are removed. Current tests
cover prefix order/gradients, original AR block migration, response budgets,
cache batching, frozen weights, reference safety, pitch augmentation, curriculum
resume and the inference CLI. Cleanup of old numbered melody-finetuning artifact
directories is scoped to this project; pretrained AR, reference encoders, VAE,
diffusion and dataset artifacts are retained.


## Verification and submitted run

On 14 November 2026, 261 CPU melody tests passed. The CUDA cached-decoding
numerical regression also passed separately. Tests include two-rank training/
resume, worker prefetch with deterministic curriculum indices, actual initial/
final genre proportions, prefix ordering and gradients, and all task/CLI handoffs.
A 36-example production-data audit passed across train/validation/test; the
largest sampled complete response contained 2177 tokens, with its full context
below 8192. Reference windows respected their bounds and instrument sets.

A real-checkpoint smoke loaded the original pretrained AR and both trained
frozen encoders. The 899,206,144-parameter V12 model completed two real-data GPU
AdamW updates with finite losses and nonzero resampler/projection gradients;
encoder gradients remained absent. A separate full-size update with 24 examples
and 8192 positions completed in about 6.7 seconds, peaking at 70.28 GiB on GH200.
This was a single-GPU capacity check; distributed behavior was checked separately
with the two-rank test.

Audited obsolete `melody_refinement_v9`, `melody_refinement_v10` and
`melody_refinement_v11` artifact directories were removed, including their old
checkpoints and TensorBoard events. The canceled V11 job was inactive before
cleanup. Original AR, style, beat, VAE, diffusion and dataset artifacts were
retained. Existing user-generated MIDI outputs were retained.

V12 was submitted as Slurm job **2909695**, one node/four GPUs, with checkpointing
and automatic continuation enabled. At submission it was pending queue priority.
Logs: `run_scripts/melody_logs/slurm-2909695.out` and `.err`.
