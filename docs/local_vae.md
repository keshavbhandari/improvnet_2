# Local instrument-roll VAE

The first learned compression stage for the hierarchical tokenizer. It is
**unconditional**: no genre, instrument embedding, clean history or accompaniment
is supplied to the network. Hierarchical diffusion and conditioning are later
stages; this does not change any AR or discrete-diffusion checkpoint.

## What gets encoded

One coarse bin contains **all** its fine frames, not only its maximum-velocity
and onset-count summaries. Defaults are 1 second / 10 ms = 100 frames. Each bin
gets one learned **256-value vector**. The internal convolutional features retain
pitch/time order before a learned projection combines the complete second.

| Batched field | Default shape | Meaning |
| --- | --- | --- |
| `roll` | `[B, 2, 128, 100]` | Velocity / 127; binary onsets |
| `corrupted_roll` | `[B, 2, 128, 100]` | Optional perturbed encoder input; `roll` remains the clean target |
| `pedal` | `[B, 1, 100]` | CC64 / 127; not duplicated across pitches |
| `valid_mask` | `[B, 100]` | True for actual fine frames, false for trailing padding |
| `latent` | `[B, 256, 1, 1]` | One vector per second in `vae176_vector256`; singleton axes preserve the diffusion API; sampled for training, mean for validation |
| `events` | `[B, Emax, 11]` | Typed accompaniment rows, padded separately |
| `event_positions` | `[B, Emax, 2]` | Start/end relative to this bin in fine frames |
| `event_mask` | `[B, Emax]` | True for actual accompaniment rows |

Accompaniment rows keep track/channel/program/drum flag/pitch/velocity,
onset-bin/offset, end-bin/offset and source order. Notes carried in from previous
bins are included: their relative start is negative, and their end may extend
beyond this bin. These are **typed integer features, not flat vocabulary IDs**;
an event vocabulary/conditioner is intentionally deferred until its architecture
is designed. The trainer leaves them on CPU rather than feeding them to the VAE.

`bin_tensors` also supplies derived coarse summaries, previous-bin final velocity,
target channel/program/drum flag and a pedal-enabled flag. None is an additional
conditioning input. Model inputs are just roll, pedal and validity.

The existing tokenizer chooses a piano-family part first; if none exists, it
chooses another part reproducibly, including drums. Only piano-family targets
have a sustain channel. Existing same-pitch overlap normalization applies.

## Model and objective

- A convolutional 2-D residual encoder downsamples pitch and time by 4.
- The backbone starts at **176 channels**, about **62.22M parameters**
  (retained baseline: 96 channels, 18.55M parameters).
  The 1-second / 10-ms clock is unchanged. Vector256 has 64,632,440 parameters
  including its learned global compression/expansion projections.
- A separate 1-D encoder handles pedal; hidden pedal features fuse into the
  posterior. Stored pedal remains one value per frame.
- The diagonal Gaussian posterior has learned mean/log-variance. Its decoder
  receives **only the latent**: no encoder-to-decoder shortcut can bypass it.
- With `LATENT_LAYOUT="vector"`, fused encoder features are projected to four
  internal feature planes and flattened in pitch/time order. A linear layer
  predicts the vector's Gaussian mean/log-variance. The decoder learns a linear
  expansion back to those internal planes before convolutional reconstruction.
  These planes are reconstructed from the vector, never passed around it.
  No `tanh` bounds the latent; stage 2 learns per-coordinate mean/std normalization.
  `LATENT_CHANNELS=256` specifies vector width here; spatial mode still interprets
  it as the number of pitch/time planes. Legacy checkpoint configs default to
  spatial mode and retain their existing state keys.
- Separate heads reconstruct activity, onsets, nonzero velocity and pedal.
  The activity head gates velocity; the onset head distinguishes repeated attacks
  with unchanged velocity. No attacks are fabricated at coarse-bin boundaries.
- Configurable frame counts are internally padded to the downsampling factor,
  then cropped. Reconstruction and KL exclude padding; partly valid latent columns
  have fractional KL weight. Entirely padded examples are rejected.

Each reconstruction loss is activity BCE + onset BCE + active-cell velocity L1
+ enabled-pedal L1 + a small onset/activity consistency penalty. Weights are
**activity 1, onset 2, velocity 5, pedal 1, consistency 0.1**. Previously onset
was 1 and velocity 10. BCE averages each present class separately, then assigns
40% of the activity loss / 20% of the onset loss to positives; the rest penalizes
false positives. Silent patches still receive the entire negative-class loss.
These weighted sigmoid outputs are **not calibrated probabilities**; evaluate
thresholds on validation rather than treating 0.99 as a universal requirement.

Training combines three branches in one DDP-compatible forward:

1. Clean input -> clean reconstruction, using posterior samples.
2. Semantically perturbed input -> the **same clean target**, with unchanged pedal.
3. Clean posterior sample + latent jitter -> the **same clean target**.

The total is `clean + ramp * (0.5 * denoising + 0.5 * jitter) + beta * KL`.
`ramp` grows to 1 over 5,000 steps; it also grows the maximum training jitter.
KL uses the mean of clean/corrupted posterior KLs when both are enabled; its
weight remains `1e-4`, warmed up over 5,000 steps. There are no genre or instrument
embeddings, encoder shortcuts, or extra learned noise-scale parameters.
The configured step budget is a starting point, not a convergence guarantee.
Increasing `N_STEPS` (or `--steps`) resumes normally without
`--allow-resume-changes`. Weights, optimizer, RNG, data position and best-score
tracking are retained. The remaining cosine LR decay extends from the saved
learning rate to `MIN_LR` at the new endpoint, without restarting warmup or
jumping back to a higher LR. This extended curve is saved in future checkpoints;
legacy checkpoints are supported. Extending during warmup preserves its original
length; extending after LR has reached its floor keeps it there. Reducing a saved
run's step budget is rejected.

This follows the idea of local piano-roll compression in
[Huang et al.](https://arxiv.org/html/2402.14285v4#S5), adapted to 100-frame bins
and separately stored pedal. Its denoising branch is inspired by the paper's
recipe, with gentler corruption and an additional latent-jitter branch, not an
exact reproduction of its training hyperparameters.

### Conservative corruption and latent jitter

Each nearby note has a 10% probability of **one proposal**, with operation shares
20% pitch shift, 30% adjacent-note addition, 40% timing shift, 10% deletion.
Thus proposal rates are approximately 2%, 3%, 4%, 1% respectively; actual edits
are lower because unsafe proposals are skipped. This is an expected rate, not
a guaranteed or hard-capped fraction in every bin. Sparse/silent bins are never
forced to contain a corruption.

- Pitch shifts: +/-1 semitone, only fully contained notes with at least two local
  attacks; isolated notes and carries are not pitch-shifted.
- Additions: a complete copy at +/-1 semitone, retaining the original note.
- Timing: +/-10, 20 or 30 ms; shift onset and release together, keeping velocity
  and duration. Notes may cross bin boundaries on the shared clock.
- Deletion: only fully contained notes in bins with at least four attacks;
  limited to at most `max(1, attacks // 10)` deletions. Never delete a carry.
- Reject out-of-range pitches/times and same-pitch overlaps, without wrapping,
  clipping or retrying. Reject timing shifts that remove all local evidence of
  an originally visible note. Drums omit pitch shifts/additions, preserving IDs.

Corruption uses complete normalized events extracted once per cached piece,
including neighbouring attacks and carries, **before** cropping the one-second
input. Attacks before the bin stay carries; only real attacks inside the bin
set onset bits. Padding, clean targets, pedal and accompaniment remain unchanged.
Ordinary pitch augmentation then transposes clean and corrupted rolls together.
Neighbouring events are used to construct consistent corruption; the VAE still
sees only its one-second input, not additional context bins.

Training jitter amplitude is uniform per example from 0 to a warmed-up maximum
of 10% of the detached, valid-weighted clean posterior-mean channel standard
deviation in the **current minibatch**. It is additional to posterior sampling,
not a diffusion timestep and not a per-example latent normalization. Entirely
padded columns receive no noise. This online scale follows the changing encoder;
the separate post-training robustness sweep still calibrates on training data.

Set `SEMANTIC_CORRUPTION=False` or `LATENT_JITTER_WEIGHT=0` to ablate a branch.
Corruption utilities live in `improvnet/utils/local_vae_corruption.py`.

## Raw MIDI index and genres

`local_vae_train --build-index` builds a SQLite path/metadata index, **not token
JSONL**. It does not parse the whole collection, copy MIDI, or start training.
The current index contains **2,572,644 candidate MIDI paths**. Symlinks, hidden
folders, `__MACOSX` and AppleDouble `._*` files are excluded; GigaMIDI alone has
2,125,595 `.mid`-named macOS metadata sidecars excluded from that count.

| Collection | Indexed paths | Genre handling |
| --- | ---: | --- |
| GigaMIDI | 2,136,218 | Unknown until the separate genre CSV is supplied |
| ARIA | 371,053 | Its supplied `metadata.json` tags; missing tags stay unknown |
| Pijama | 2,736 | Jazz |
| Doug McKenzie | 307 | Jazz |
| Classical Archives | 14,694 | Classical |
| MAESTRO | 1,276 | Classical |
| SymphonyNet | 46,360 | `classical` folder only; contemporary stays unknown |

All supplied genre tags are retained. Multi-tag entries have `genre="multi"`,
not an arbitrarily chosen primary tag. Unknown genres **do not exclude** music
from this unconditional training stage.

GigaMIDI's genres are in the separately distributed
[`Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv`](https://huggingface.co/datasets/Metacreation/GigaMIDI/tree/main).
That CSV has now been supplied and moved into the GigaMIDI scratch directory.
Stage 2 uses it through a **separate genre-labelled index**; the unconditional
VAE's original index/configuration remains unchanged for checkpoint compatibility.
See [stage-2 genre labels](latent_diffusion.md#genre-labelled-index).

Splits are new deterministic 97% / 2% / 1% hash-based grouped splits, **not** the
published MAESTRO splits or legacy AR train/test split. ARIA segments sharing a
file ID, and GigaMIDI variants sharing a filename hash, stay together. Other
collections group by relative path. Cross-collection duplicates or different
performances of one composition are not deduplicated by this index.

## Sampling and augmentation

Training samples collections uniformly, then files within each collection;
`COLLECTION_WEIGHTS` can change this. Thus GigaMIDI does not overwhelm smaller
collections just because it has more files. The virtual epoch has 50k sampled
bins; it is **not one exhaustive pass through every file**.

`GIGAMIDI_DRUM_ONLY_FRACTION=.05` retains a fixed, randomly selected 5% of files
under GigaMIDI's `drums-only` directories, separately in each split (rounded down).
Selection is seeded by `DATA_SPLIT_SEED` and unchanged across epochs/workers.
All other GigaMIDI files, including mixed-instrument pieces with drums, remain
eligible. This is 5% of the drum-only **file pool**, not a 5% per-batch quota.
The SQLite index and source files are untouched; no index rebuild is necessary.
Changing this fraction changes the data pool. For an existing checkpoint,
explicitly use `--allow-resume-changes` for non-exact continuation, or start a
fresh training run.

90% of training bins come from active target bins; the other 10% are uniform
over the piece, including genuine silence. Validation uses fixed active bins
with no pitch augmentation. Validation corruption is fixed by item seed, and
latent stress noise is fixed at 10% with a dedicated RNG stream that does not
consume training randomness. Pitched targets and pitched accompaniment are coherently
transposed by a bounded random -5 to +6 semitones, without wraparound. Drum targets
are not transposed, and drum accompaniment pitches stay unchanged.

Raw MIDI is read and encoded lazily in workers. An in-memory per-worker byte-bounded
cache avoids some repeated parsing; there is no large encoded-dataset cache yet.
Training uses `encode(..., recover=True)` without rewriting any source MIDI.
Paired notes are retained; unresolved pitched attacks end at their next same-pitch
attack if present, otherwise they are dropped. Unreleased channel-9 drum hits use
one fine frame, capped at the next attack. Invalid SysEx is removed while retaining
event timing; omitted channel status after SysEx is made explicit when the prior
channel status is known. RIFF/RMID containers are unwrapped. Invalid key-signature
and SMPTE-offset metadata, pitch-bend messages, and bank-select/volume CC0/CC32/CC7
values are ignored, not guessed. Valid messages are preserved; these recoveries
discard only information unused by the relative PPQ key-down canvas.
Out-of-range note_off release velocity becomes zero without changing pitch or
release time. Pitch, attack velocity and other controller values are never clipped.
Strict round-trip
encoding is unchanged. No per-file recovery logs or new repair counts are recorded.

Invalid/empty/over-limit files still trigger bounded retries. `LOG_DATA_ERRORS=False`
disables per-file error files by default, including for unrecoverable sources.
Set it true only for debugging; existing logs are not deleted automatically.
Candidate index paths are not proof that every file is parseable. TensorBoard
logs retries so unexpected data problems remain visible without large file logs.

## Scripts and commands

The three principal scripts match the AR layout:

- `improvnet/model/local_vae_model.py`: model, model config class and loss helpers.
- `improvnet/model/local_vae_train.py`: index CLI, training, validation and resume.
- `improvnet/model/local_vae_config.py`: paths and editable values only.

Tensor conversion, lazy MIDI loading and indexing are in
`improvnet/utils/local_vae_data.py`. Commands run from the repository root with
the `improvnet` environment activated:

```bash
# Index only; an existing matching index is reused without overwriting it.
python -m improvnet.model.local_vae_train --build-index

# One node, four GPUs. BATCH_SIZE is per GPU; accumulation defaults to 1.
torchrun --standalone --nproc_per_node=4 -m improvnet.model.local_vae_train

# Short one-GPU debug run, separate from the production SAVE_DIR.
# Reuse precisely the same settings to resume; choose a fresh directory to restart.
torchrun --standalone --nproc_per_node=1 -m improvnet.model.local_vae_train \
  --steps 10 --batch-size 4 --num-workers 2 \
  --save-dir /e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/local_vae_debug

# CPU/synthetic checks; no source datasets or production checkpoints needed.
python -m unittest discover -s tests -p test_local_vae.py
python -m unittest discover -s tests -p test_local_vae_denoising.py
```

`--config` accepts an importable config module. Other debug overrides are
`--index`, `--samples-per-epoch`, `--accum-steps`, `--num-workers`, and `--cpu`.
`--build-index --max-index-files N --index /new/debug.sqlite` makes an explicitly
marked partial index. Never use that as the full-corpus index inadvertently.

The trainer supports DDP, mixed precision, gradient clipping, cosine LR, KL
warmup and deterministic validation. Latest/best checkpoints are written
atomically; best is selected by a fixed-weight mean of clean, denoising and
10%-jitter validation reconstruction, not the changing warmup/KL-weighted total.
Clean, denoising and jitter activity/onset precision, recall and F1 are separately
logged to TensorBoard; terminal output includes clean and jitter onset scores.
Resume restores optimizer/scheduler/scaler, per-rank RNG and data position.
Architecture, world size, data index and optimization settings must match for an
exact resume. Logging cadence, worker count and cache budget may change. An
unrelated checkpoint in the run directory is refused, not overwritten.

The default save directory is `artifacts/vae/vae176_vector256`. The
176-channel model must start fresh, not resume a 96-channel checkpoint. Existing
checkpoint architecture metadata still lets inference load older models;
model state keys and default inference behavior have not changed. Once the new
encoder is trained, recompute stage-2 latent statistics and **retrain stage 2**;
the old diffusion model is not compatible merely because the latent shape matches.
The denoising branches increase training work/memory; the wider model's batch
capacity has not yet been benchmarked at full GPU occupancy.

### Separate experiments and TensorBoard comparison

Set `RUN_NAME` in the config or pass `--run-name` for a separate experiment.
All named runs share one TensorBoard root but keep independent checkpoints:

```text
artifacts/vae/
  baseline96/       best_model.pt, latest_checkpoint.pt, data_errors/
  vae176/          best_model.pt, latest_checkpoint.pt (created when training starts)
  tensorboard/
    baseline96/    preserved events from the previous run
    vae176/        events for the new run
```

```bash
# Fresh architecture, independent weights/logs.
# Adjust per-GPU batch size if the larger model does not fit.
torchrun --standalone --nproc_per_node=4 -m improvnet.model.local_vae_train \
  --run-name vae176 --base-channels 176

# Compare the existing baseline with the new run in the same TensorBoard.
tensorboard --logdir /e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/vae/tensorboard
```

Named runs use `RUNS_DIR/<name>/{latest_checkpoint.pt,best_model.pt}` and
`TENSORBOARD_ROOT/<name>/` for logs. The existing run was moved intact to
`vae/baseline96`; `local_vae_v1` is a compatibility link, so existing stage-2
checkpoint paths still work. The baseline's old `tensorboard/` path also links
to the shared log root. No checkpoint/event contents were changed or deleted.
Identical metric tags remain comparable; the Text tab records each run's settings.
Reuse the same command/name to resume that experiment. Existing-checkpoint guards
still prevent replacing a different architecture, even with `--allow-resume-changes`.
`--save-dir` is an alternative explicit path, mutually exclusive with `--run-name`.
Reconstruction tools accept `--checkpoint` pointing to either run's `best_model.pt`;
their default follows `SAVE_DIR`, currently `vae176_vector256`. Pass
`--checkpoint /.../artifacts/vae/baseline96/best_model.pt` to evaluate the baseline.

Width 176 yields **62,221,828 parameters** versus 18,548,468 at width 96. This
matches the approximate parameter budget of Huang et al.'s released VAE, not its
architecture/compression ratio. `vae176` uses `[8,32,25]`; the current
retired `vae176_latent4` run used `[4,32,25]` (3,200 values per second).
`vae176_vector256` uses `[256,1,1]` (256 values per second, 12.5x fewer).
The 176-wide convolutional backbone is retained; this experiment reduces the
information bottleneck, not the backbone width. Smaller vectors do not guarantee
better diffusion or 99% reconstruction: evaluate clean/robust note scores and
recalibrate held-note decoding on validation after training. The old VAE and
diffusion checkpoint folders were removed at the user's request; historical
evaluation reports and shared VAE TensorBoard logs remain. For a
capacity-only comparison keep the data, loss settings and effective batch fixed
where practical. The new defaults also change onset positive-class share to 20%,
so this default run compares the combined capacity/loss update, not capacity alone.

### Joint threshold calibration

Every validation pass searches configurable activity/onset threshold pairs using
streaming score histograms; it does not keep rolls/logits in memory or perform
extra model forwards. Onsets are gated by activity, exactly as in decoding.
Counts are summed across GPUs, including uneven/empty validation shards.

Selection maximizes the worst branch's mean activity/onset F1, subject to at least
99% activity **and** onset recall on both clean and latent-jitter branches. If no
pair meets the floor, the best unconstrained pair is returned with an explicit
`recall_floor_met=False`; silent-only validation cannot certify success. Semantic
denoising is still scored but excluded from threshold selection because restoring
deleted/shifted notes can be intrinsically ambiguous in a one-second window.

Fixed-threshold `validation/onset_*` and `activity_*` tags remain unchanged for
fair comparison. Selected thresholds and branch precision/recall/F1 appear under
`validation/calibration/`. Their scores are tuning-set measurements, not an
unbiased estimate of test performance or a guarantee of 99% reconstruction.

Checkpoints save the selected pair and its validation step under
`decoder_calibration`, without new model parameters. Reconstruction, robustness
and stage-2 inference CLIs use this pair automatically. Explicit
`--activity-threshold`/`--onset-threshold` override it; older checkpoints retain
the previous `.5/.99` fallback. These are score cutoffs, not probability
calibration. Calibration never changes training targets, loss, or posterior RNG.

The completed 100k-step `vae176_latent4` run's best checkpoint is from step 92k.
A first post-training validation sequence search selected `.95/.995` activity/onset
cutoffs for clean and 10%-jitter decoded-note F1, while preserving at least 99%
recall, using legacy activity-to-note assembly. Its learned weights are unchanged,
and `best_model.before_threshold_calibration_20261009.pt` retains the original
checkpoint. The held-note calibration below supersedes these inference settings. The completed run's original
fixed-threshold TensorBoard tags still reflect `.5/.9`; its training-time
calibration optimized frame scores. Sequence calibration reports explicitly show
whether 99% decoded-note **precision** is also met; the recall flag alone does
not certify it.

With that legacy pair, 200 test files / 1,600 bins give activity F1 99.89%,
onset F1 99.81% and decoded-note F1 99.15%, with note precision 98.37%
(141 extra / 5 missed notes out of 8,505). The Pijama jazz subset's note F1 is
97.80%. A separate test-window noise sweep averages 99.72% note F1 at 10%
channel-std perturbation and 96.87% at 20%, over three noise draws. These meet
the aggregate 99% F1 goal at low noise, but not a 99% note-precision requirement
or a per-genre guarantee.

The subsequent held-note calibration uses the same trained weights and selects
start activity `.95`, continuation activity `.70`, onset `.985`, and a **10 ms**
internal gap allowance, on 200 validation windows only. All one-second bins are
assembled continuously after inference batching. An activity-only return cannot
start a note except an initial-window carry. New accepted onsets always retrigger;
short gaps ending in a new onset are preserved. Unbridged/trailing releases stay
at the first inactive frame rather than being delayed by the lookahead allowance.

On the original 200 test windows / 8,505 reference notes, held-note precision is
**99.62%**, recall **99.95%**, F1 **99.79%**, and F1 with onset **and offset**
matching **99.42%** (previously 98.23%). Extras fall from 141 to 32, misses from
5 to 4, and release MAE from 5.66 to 2.41 ms. Rapid-repeat attack recall remains
100%; its offset-aware recall is 99.76% (two release mismatches). Pijama note F1
rises from 97.80% to 99.77%. This is not a per-collection precision guarantee:
Classical Archives remains at 98.63% note precision. The remaining 32 extras all
have explicit onset predictions; they are no longer inferred activity restarts.

The checkpoint saves these inference rules in `decoder_calibration.note_decoding`;
VAE and stage-2 inference both use them automatically. A checkpoint backup retains
the exact previous metadata/weights. The file digest changes, so start the planned
fresh stage-2 run with the calibrated checkpoint; existing checkpoints pinned to
another digest are deliberately rejected.

To continue the same run after changing batch size, accumulation, the GigaMIDI
drum-only fraction, validation sizes, loss weights, KL warmup, gradient clipping
or sampling/augmentation settings, explicitly pass:

```bash
torchrun --standalone --nproc_per_node=1 -m improvnet.model.local_vae_train \
  --allow-resume-changes
```

This retains model/optimizer/scheduler/scaler state and the training step, but
restarts data at a virtual epoch boundary and resets validation best-score
tracking. It is not an exact replay. Architecture, GPU world size, index identity,
LR/schedule and other training settings must still match. Without this flag,
the exact-resume safeguards remain in effect. Newly saved checkpoints record
the current settings, so subsequent unchanged resumes do not need the flag.
`ACTIVITY_THRESHOLD` and `ONSET_THRESHOLD` affect metrics only: changing them
requires no opt-in and preserves the data position and validation best score.
Changing a threshold does not change what the model learns.

`SIGUSR1`, `SIGTERM`, or a `CHECKPOINT_REQUEST_FILE` request saves at the next
optimizer boundary. There is no new SLURM launcher in this stage.

TensorBoard and console metrics include reconstruction/component losses, KL,
activity/onset precision-recall-F1, velocity/pedal MAE in MIDI units, onset/activity
inconsistency and retries. These are reconstruction metrics, not musical quality
scores. A short default-model GH200 smoke test at batch 32 used about 4.1 GiB
of allocated GPU memory; this is not a maximum-batch-size benchmark.

## Reconstruction and diffusion-readiness checks

```bash
# Reconstruct a full piece, using the checkpoint's tokenizer/architecture.
python -m improvnet.inference.local_vae_inference \
  --input improvnet/inference/examples/bach_846.mid

# Collection-balanced, unique held-out piece groups; no pitch augmentation.
python -m improvnet.inference.local_vae_evaluate \
  --num-files 100 --bins-per-file 8 --save-examples 3
```

Both default to `SAVE_DIR/best_model.pt`; override with `--checkpoint /path/model.pt`.
Use `--device cpu` if needed, `--batch-size` to control memory, and `--output-dir`
for a **new/empty** destination. Otherwise each run creates a unique directory in
scratch alongside the configured run directories. No checkpoints or input MIDIs are changed.

Single-file inference saves `reconstructed.mid` (original accompaniment kept),
`reference_target.mid`, `reconstructed_target.mid`, and one small `report.json`.
Optional `--target-channel`/`--target-program` selects the instrument. The default
uses the posterior mean; `--sample` tests a seeded posterior sample instead.

Batch evaluation scores one random contiguous window per file (up to 8 bins),
including real silence. It uses only the checkpoint index's **test split**, keeps
the training drum-only subset policy, and avoids repeated group IDs. Collections
are round-robin balanced, not weighted by corpus size. Invalid/empty/over-limit
files are skipped with bounded attempts and compact error-category counts.
The index does not deduplicate performances across different collections.
The first three examples are reconstructed in full for listening; only their
selected windows contribute to scores. Set `--save-examples 0` for no MIDI output.

| Report field | Plain-language meaning |
| --- | --- |
| `activity_f1`, `onset_f1` | How well active cells / exact attack frames survive; higher is better. |
| `notes_f1`, missed/extra notes | Same-pitch, one-to-one attacks matched within 50 ms. |
| `onset_mae_ms`, `offset_mae_ms` | Average attack/release timing error for matched notes. |
| `notes_with_offsets_f1` | Attack matching plus release tolerance max(50 ms, 20% of reference duration). |
| `offset_p95_ms`, `releases_within_20ms` | Tail release error and fraction of matched releases within 20 ms. |
| `rapid_note_recall`, `rapid_note_with_offset_recall` | Repeated same-pitch starts <=200 ms apart, with/without the release constraint. |
| `velocity_mae`, `pedal_mae` | Average error in 0–127 MIDI units; lower is better. Missing notes count as zero velocity. |
| `boundary_f1` | Activity F1 in the first 50 ms of each bin; checks carry-over trouble. |
| latent `mean`, `std` | Posterior-mean spread per latent channel over valid pitch/time cells. |
| `posterior_std`, `kl_per_element` | Posterior uncertainty and average distance from the unit Gaussian prior. |
| `active_channel_fraction` | Fraction of channels with mean-latent variance above 0.01; a simple collapse warning. |
| sampled / zero-latent F1 | Does sampling preserve reconstruction, and does removing latent information hurt? |

All errors compare the **normalized tokenizer canvas**, not raw MIDI, and exclude
padding and accompaniment. Velocity error includes every reference-active cell;
false-positive cells are captured by F1. Disabled pedal and undefined metrics
are `null`, not falsely perfect zeros. Entire pieces are decoded across bins;
window note scoring clips both reference and prediction at the same boundaries.
The lossless tokenizer retains its documented non-strict activity-to-note rules.
For calibrated model predictions, confidence-aware held-note assembly first
produces a consistent velocity/onset canvas. Legacy checkpoints remain unchanged.
Use `--note-decoding legacy` to opt out, or `--note-decoding held`,
`--continuation-threshold`, and `--max-gap-ms` to override saved rules. To reproduce
the old baseline, also set `--activity-threshold .95 --onset-threshold .995`.
A playable MIDI alone is therefore **not** evidence of faithful reconstruction.

Calibrate note assembly before evaluating the test split:

```bash
python -m improvnet.inference.local_vae_note_calibration \
  --num-files 200 --bins-per-file 8 --batch-size 8 \
  --cache-confidence --output-dir /new/validation-directory
```

This command uses validation only, keeps the clean reference as the scoring target,
and caches **model** probabilities for complete contiguous windows. Candidate
selection requires >=99% note recall and frame F1, no lower offset-aware F1 than
the legacy baseline, <=2 ms mean release-error increase, <=0.1 percentage-point
rapid-repeat recall loss, and <=0.5 percentage-point loss in releases within 20 ms.
It selects highest note F1 among passing settings, then offset F1/release error.
If none passes, no calibration is selected. `--save-calibration` explicitly saves
selected metadata atomically, verifies unchanged learned weights/training step,
and preserves an original checkpoint backup. No VAE retraining is performed.
`--cached-validation /previous/validation-directory` reuses a cache from the exact
same checkpoint and rechecks validation membership; use `--device cpu` to avoid
GPU work while trying a different candidate grid.

`readiness` is a deliberately conservative **heuristic**, not proof. Fewer than
20 files / 128 bins gives `insufficient_evidence`. Otherwise it flags activity
F1 < 0.90, onset/note F1 < 0.85, velocity MAE > 10, release MAE > 100 ms, boundary
F1 < 0.85, fewer than half the channels varying, sampling losing > 0.10 F1, or
zeroing latents losing < 0.10 activity F1. These are editable starting thresholds,
not established musical-quality cutoffs. `promising_not_proven` means try a small
diffusion pilot and listen. Channel spread includes spatial variation and does
not prove content diversity; low KL can indicate collapse and high KL need not
be bad. Diffusion latents need not already be Gaussian—estimate their scaling
before diffusion training. Check collection-level scores, not just the aggregate.

```bash
python -m unittest discover -s tests -p test_local_vae_inference.py
```

## Frozen latent-noise experiment

```bash
python -m improvnet.inference.local_vae_robustness \
  --num-files 200 --calibration-files 64 --bins-per-file 8 \
  --draws 3 --split test --batch-size 8
```

This **does not train or change the VAE**. It estimates per-channel latent spread
from unaugmented training windows, then evaluates collection-balanced held-out
windows at fixed thresholds. The default split is validation; use `--split test`
after choosing thresholds on validation. Noise is `mean + alpha * training_channel_std * N(0,1)`.
Defaults test alpha 0, 0.005, 0.01, 0.02, 0.05, 0.10 and 0.20, plus ordinary
posterior sampling as a control. These are noise **standard deviations**, not
percentages of replaced notes or diffusion timesteps. Zero alpha is clean mean
decoding; no extra noise is added to a sampled posterior. Padding is excluded.
All strengths share the same random directions per window/draw for fair comparison.

Outputs are a compact `report.json`, two PNG plots, and `selection.csv` recording
index row IDs and bin windows (no large per-file metric logs). Plot shading shows
variation between noise draws, **not** a dataset confidence interval. Window note
scores clip both sides at the same endpoints and retain continuity inside windows.
Held-out splits are unchanged; cross-collection duplicates are not deduplicated.
Training scale is pooled over valid spatial cells; it is not a per-coordinate
whitening transform. A decoder tolerating isotropic jitter may still fail on
structured diffusion errors. Inspect piano/drum/other results separately.

`--alphas 0 0.01 0.05 0.1` customizes the sweep. Outputs go to a new unique
scratch `artifacts/local_vae_evaluation/latent-robustness-*` directory by default.
Plotting requires `matplotlib` (`python -m pip install matplotlib` if missing).

## Using the tensors

```python
from improvnet.tokenizer.hierarchical import HierarchicalTokenizer
from improvnet.utils.local_vae_data import bin_tensors, collate_bins
from improvnet.model.local_vae_model import LocalVAE

piece = HierarchicalTokenizer().encode("/path/to/piece.mid", seed=42)
batch = collate_bins([bin_tensors(piece, 0)])
model = LocalVAE()  # Untrained here; load a trained checkpoint for reconstruction.
output = model(batch["roll"], batch["pedal"], batch["valid_mask"], sample=False)
arrays = output.reconstruction.tensors(batch["valid_mask"], batch["pedal_enabled"])
```

To decode a reconstructed piece, assemble velocity/onset/pedal arrays for **all
bins** into the original hierarchical piece, recompute its coarse summaries with
`refresh_coarse()`, and invoke the existing MIDI decoder once. Decoding each bin
as a separate MIDI would lose note continuity. `previous_velocity` is available
for future boundary-aware modeling, but is deliberately not supplied to this
unconditional local VAE.
