# Contrastive continuity metric

Separate, unconditioned bidirectional encoder. No genre or instrumentation
embeddings, no AR checkpoint, and no changes to the queued genre trainer.
Architecture: 2048 dimensions, 20 layers, 16 query/4 KV heads, RMSNorm, RoPE
and SwiGLU matching the genre/uniform encoders. A two-layer projection produces
256-dimensional L2-normalized embeddings with independent trainable weights.

## Data and objective

Each positive is two **non-overlapping, note-containing excerpts from the same
performance record**, not two performances of the same composition. Training
lengths are independently sampled integers from 128–512, capped by available
length; validation uses exactly 256 real tokens per view. Padding is masked.
Too-short/rest-only records are deterministically resampled within their fold,
with a bounded retry and explicit failure if there are no eligible records.
This slightly changes the eligible sampling distribution, not the source folds.

Each training view gets an **independent** bounded random pitch shift up to
±7 semitones, with drum/percussion pitches unchanged. The full clean piece is
used to resolve instrument roles before each crop; the two shifts need not
match. Independent draws can coincide by chance. Validation/test do not augment.
Genre metadata prefixes are stripped, and unknown/null genres are included.
Default filtering excludes the same Gigamidi `drums-only` paths as AR/flow matching;
normal mixtures with drums remain. `audit --include-drums-only` includes those too.

Symmetric cross-entropy/InfoNCE: compare A against all B views and B against all
A views, dividing cosine logits by fixed `TEMPERATURE=0.1`. The paired view is
the target; other pieces are negatives. Duplicate piece identities are masked
as negatives, not falsely penalized. DDP training gathers embeddings with
autograd so negatives include all GPUs. Accumulation increases optimizer batch
size, **not** the contrastive candidate pool beyond one global forward batch.
There are no same-view A/A or B/B negatives in this cross-view objective.

Splits and negative masking conservatively group `piece_id`/`midi_hash` when
present, otherwise casefolded MIDI filename stems across sources. Positives
always stay in one record. This cannot detect differently named duplicates;
false negatives between related compositions/performances remain possible.

## Files and launch

- `continuity_config.py`: values and paths only, including min/max/eval lengths.
- `continuity_model.py`: model dataclass and trainable encoder/projection.
- `continuity_train.py`: AMP/DDP, accumulation, TensorBoard, console/JSONL,
  atomic checkpoints and deterministic optimizer/RNG/data-position resume.
- `audit.py`, `data.py`, `objective.py`, `scoring.py`: separate supporting logic.

From the repository root, with the `improvnet` environment:

```bash
python -m improvnet.metrics.continuity.audit
torchrun --standalone --nproc_per_node=1 -m improvnet.metrics.continuity.continuity_train
```

The audit scans metadata headers/indexes, not complete token arrays. It writes
a separate continuity manifest that includes unknown genres; do not use the
genre manifest. Existing audit outputs are never overwritten. Training can
also be launched by the absolute script path (absolute imports).
Use `--batch-size`, `--steps`, `--workers`, `--save-dir`, `--manifest`,
`--accum-steps`, `--device`, `--max-validation-batches`, or `--no-tensorboard`.
Short debug runs must set `WARMUP_STEPS < N_STEPS` in a custom config module.
`latest_checkpoint.pt` resumes strictly; absent latest checkpoints allow reuse
of the same debug directory with orphaned outputs archived under `previous_runs`.
`CHECKPOINT_REQUEST_FILE` stops safely at an optimizer boundary after saving.

## Logged scores and reward

Train/validation report contrastive loss, retrieval accuracy, positive/negative
cosine similarity and the positive-minus-hardest-negative margin. Training
numbers are globally aggregated. Validation uses deterministic per-rank local
candidate batches (no uneven-shard forward collectives), then aggregates totals.
Singleton/duplicate-only validation batches are skipped. Compare validation
only with the same batch size, world size, temperature and batch limit: pool
size/composition affects loss/retrieval. Best checkpoint selects lowest
validation loss, not training accuracy.

```python
from improvnet.metrics.continuity.scoring import ContinuityMetric
metric = ContinuityMetric.from_checkpoint("/path/to/best_checkpoint.pt", device="cuda")
score = metric.score(previous_tokens, current_tokens)  # each <=512 tokens
boundary_score = metric.score_boundary(history_tokens, canvas_tokens, window_size=256)
```

The output is cosine similarity in [-1,1], **not** a calibrated probability.
High means same-piece compatibility, not guaranteed musical smoothness; low
need not be negative. This symmetric objective learns compatibility, not
directional next-segment prediction, and can rely on instrumentation/rhythm
even without explicit conditioning. Independent pitch shifts help reduce key
shortcuts but can reduce sensitivity to abrupt key changes. Before RL use,
evaluate real continuations, unrelated excerpts, shuffled/repeated notes and
genre-transfer outputs to check that the metric tracks human judgements.
Keep MIDI syntax as a separate gate; compatibility alone is not syntax quality.
