# Known-genre classifier

Independent bidirectional encoder for finalized musical excerpts. It receives
no genre/instrument conditioning and strips all metadata prefixes to prevent
label leakage. It does not import/change AR or diffusion training/checkpoints.

## One config, one model, one trainer

- `genre_classifier_config.py`: uppercase values and paths only, covering
  architecture, data, augmentation, sampling and training. `MIN_LENGTH=128`
  and `MAX_LENGTH=4096` bound training crops; `EVAL_LENGTH=256` controls
  held-out crops. `MAX_LENGTH` also limits encoder/scoring input length.
- `genre_classifier_model.py`: `GenreClassifierConfig`, bidirectional encoder,
  pooling and classification head. The trainer builds model settings from the
  constants plus tokenizer vocabulary/padding and audited genre count.
- `genre_classifier_train.py`: training, distributed evaluation, resume/calibration.

There are no compatibility `config.py`, `model.py`, or `train.py` files and no
second training-config module. Supporting audit/data/tokenization/evaluation/
scoring modules contain distinct logic, not duplicate trainers or configs.

Backbone settings match the AR model: 2048 dimensions, 20 layers, 16 query
heads / 4 KV heads, 128-dimensional heads, SwiGLU 8/3 (5504 hidden units),
RMSNorm and Q/K normalization, RoPE base 500000, dropout 0.1 and gradient
checkpointing. Classification uses masked mean pooling and a **12-class head**.
Padding is masked at every layer and excluded from pooling.

## Dataset and balancing

The complete metadata audit found 614,671 known-labeled records among
2,917,032 total records. **Unknown, null and `other` annotations are excluded
from every fold**, never turned into negative genre labels. The 12 genres are:
ambient, atonal, blues, classical, folk, game, jazz, metal, pop, ragtime, rock,
and soundtrack. Counts/source signatures are in `dataset_audit.json`.

The current known-only report/SQLite manifest are in:

```text
/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/genre_classifier/dataset_manifest.json
/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/genre_classifier/dataset_manifest.sqlite
```

Splits are deterministic 90%/5%/5% piece-group hashes, using explicit piece
IDs/hashes when available, otherwise casefolded MIDI filename stems across
directories/sources. Differently named copies/arrangements cannot be detected
from headers. Old external manifests are preserved, but manifests/checkpoints
with an `other` category are rejected rather than silently relabeled.

Training caps the classical candidate pool at 30,000; all jazz/blues candidates
are retained. Sampling with replacement assigns jazz and blues 4× the mass of
each remaining genre: **4/18 = 22.22% each**, and **1/18 = 5.56% each** for
the remaining ten genres. `TRAIN_CLASS_CAPS`, `FOCUS_GENRES`,
`FOCUS_SAMPLING_WEIGHT` and `SAMPLES_PER_EPOCH` control this policy.
Known-label validation/test are never capped or rebalanced. Very rare genres
have limited held-out evidence (atonal: 15 validation records; blues: 135).

Before the first training run, build the configured manifest with
`python -m improvnet.metrics.genre_classifier.audit`. Training does not build
it automatically, including when launched with `torchrun`.

To audit changed inputs, choose a fresh prefix (existing artifacts are not
overwritten):

```bash
python -m improvnet.metrics.genre_classifier.audit --output-prefix /path/to/new_manifest
```

The audit reads metadata headers using trusted existing byte indexes, not the
large token arrays. It is not exhaustive validation of every token payload.
Source/index mutations reject auditing/loading; malformed labeled music fails
with its source and byte offset during data loading.

## Pitch augmentation and excerpts

`PITCH_AUGMENTATION=True`, `MAX_SEMITONE_SHIFT=7`: training chooses one uniform
random integer semitone shift in [-7, 7], restricted to keep every melodic
pitch in MIDI range 0..127. Zero is allowed. **No pitch clipping**, so melodic
intervals/chords remain unchanged. Drum/percussion notes, velocity, timing,
instrument fields and structural tokens are unchanged. Instruments are resolved
over the full normalized sequence before cropping, including drum events that
cross crop boundaries. This is classifier-only code; shared AR helpers are not
modified. Set `PITCH_AUGMENTATION=False` to disable it.

Validation/test and scoring never transpose inputs. Training uses note-anchored
crops at lengths 128/256/512/1024/2048/4096, capped by piece length; held-out
crops are deterministic 256-token excerpts by default. Each replacement-sampler
draw receives an independently seeded crop/shift, reproducible across workers,
ranks and mid-epoch resumes.

## Training (not launched)

An existing `latest_checkpoint.pt` resumes with compatibility checks when
`RESUME_TRAINING=True`. If no latest checkpoint exists, an interrupted debug
run can reuse `SAVE_DIR`: orphaned logs, TensorBoard events and best/calibrated
checkpoints move into a unique `previous_runs/interrupted-*` subfolder before
starting fresh. Nothing is deleted or mixed with the new run. `--no-resume`
still refuses to overwrite an existing latest checkpoint.

TensorBoard (`TENSORBOARD=True`) logs accuracy and macro-F1 for all genres,
jazz/blues together, and the remaining known genres under
`train/groups/{all,jazz_blues,remaining}/...` and equivalent `validation/` tags.
Per-genre accuracy, precision, recall, F1 and support are also logged.
Per-genre accuracy means correct predictions among examples of that genre
(equivalent to recall); group F1 averages supported genres' one-vs-rest F1,
including errors across groups. Empty groups' accuracy/F1 are not logged.
Training scores cover the balanced-sampling interval since the last log;
validation scores cover the natural held-out distribution with prior correction.
All ranks' confusion counts are combined before computing scores.

From the repository root in the `improvnet` environment:

```bash
python -m improvnet.metrics.genre_classifier.genre_classifier_train
```

Use `torchrun` for multiple GPUs as for the AR model. `--config` defaults
to the single `genre_classifier_config` module; custom modules may import and
override it. Other overrides: `--manifest`, `--save-dir`, `--batch-size`,
`--accum-steps`, `--steps`, `--workers`, `--device`, `--max-validation-batches`,
`--no-resume`, `--no-calibrate`, `--no-tensorboard`. Architecture/finer controls
belong in the same config file. Reduce warmup too if testing very few steps.

Defaults: 20,000 optimizer steps, batch 8/GPU, accumulation 1, AdamW LR 1e-4
to 1e-5, warmup 1,000, validation/checkpoint every 1,000, validation batch 16,
full validation, BF16 autocast on supporting GPUs. These are starting settings,
not a memory benchmark/tuned training budget. No SLURM submission occurs here.

DDP shards global balanced draws equally, supports accumulation, and evaluates
disjoint unequal validation shards without duplicated examples or DDP-forward
collectives. Checkpoints restore optimizer/scheduler/scaler, per-rank RNG and
sample position; exact resume requires compatible data/settings/world size.
`CHECKPOINT_REQUEST_FILE` requests a safe stop at an optimizer boundary.

CE is unweighted by default: sampling already balances classes. Validation/
scoring subtract the saved log training/held-out prior ratio before temperature
scaling. TensorBoard/JSONL log loss/accuracy/NLL/macro-F1 and per-class precision/
recall/F1. Best checkpoints use mean jazz/blues F1; temperature is fitted on
validation only, never test. Outputs: `latest_checkpoint.pt`,
`best_checkpoint.pt`, optional `calibrated_checkpoint.pt`, in the classifier-only
save directory. No production classifier has been trained by these changes.

## Genre score after training

```python
from improvnet.metrics.genre_classifier.scoring import GenreClassifierMetric

metric = GenreClassifierMetric.from_checkpoint("/path/to/calibrated_checkpoint.pt", device="cuda")
jazz_score = metric.score(finalized_canvas_ids, target_genre="jazz", valid_mask=canvas_mask)
all_scores = metric.probabilities(finalized_canvas_ids, valid_mask=canvas_mask)
batch_scores = metric.score_batch(list_of_canvases, valid_masks=list_of_masks)
```

Vocabulary/label ordering are checked. Padding/noise tokens must be excluded
explicitly; metadata is stripped. Empty/all-rest/terminal-only inputs lack genre
evidence and fail. Long pieces use note-containing windows with length-weighted
probabilities. These are learned genre probabilities, not objective jazz-quality
scores; evaluate piano-only discrimination and generated outputs before RL.
Do not use genre reward for prompt continuation by default. Keep syntax/fidelity
checks separate for all tasks.
