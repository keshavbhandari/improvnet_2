# Style and beat contrastive conditioning

Current worktree and branch: `improvnet_melody_contrastive`.
Absolute worktree: `/e/home/jusers/bhandari1/jupiter/improvnet_melody_contrastive`.
The previous `improvnet_melody_v6` worktree was renamed in place, preserving
uncommitted changes and generated MIDI examples. No training job was submitted
for this change. The current finetuning protocol is **11**, with new outputs in
`artifacts/melody_refinement_v8/`; previous V6/V7 outputs are retained.

## Two encoders

| Name | Learns from | Positive views | Negatives |
|---|---|---|---|
| `style` | Existing solo and mixture corpus | Solo instrument versus another view of the same piece; same or different excerpts | Different canonical pieces |
| `beat` | Multi-instrument files only | Disjoint nonempty instrument groups from exactly the same time interval | Other pieces and separated regions of the same recording |

The existing trained style weights are retained. Their artifact directory is
now `/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/style/`; the old
`melody_reference_encoder_v6` path is a compatibility symlink. The loader accepts
that checkpoint's legacy format. Its best checkpoint remains step 26,500.
The new beat encoder writes to the sibling `beat/` directory. It has not been
trained yet. An original style checkpoint cannot be substituted for a beat
checkpoint: each frozen loader checks its model kind and tokenizer fingerprint.

Both models share the previous configuration: width 768, 8 bidirectional layers,
12 query heads / 4 KV heads, 256-value unit-normalized embedding, dropout 0.1,
at most 512 tokens per view, 64 anchor pairs/GPU, four GPUs, temperature 0.1,
AdamW (0.9, 0.95), weight decay 0.01, gradient clip 1.0, and 30,000 updates.
Both encoder configs are standalone: they neither import one another nor the
melody config. Each setting has an inline explanation, as does the melody config.
Both encoders specify a complete training schedule: 500 warmup steps
to 0.00015, cosine decay to 0.00001 at step 20,000, then cosine decay to 0.000005
at step 30,000. `DECAY_STEPS=20000` marks the end of the main cosine decay;
`TAIL_END_LR=5e-6` is the final learning rate. Either config can train from scratch
or resume its latest checkpoint. Historical style checkpoint schedule names are
translated when loading without changing the curve or saved weights.
Validate/save every 500 updates.

Independent span masking is unchanged: a 90% chance per view of masking a
uniformly sampled 15–45% of interior tokens; 10% of views remain clean. Spans
cover 4–16 tokens. Pitch/velocity, instrument, onset, duration and `<T>` tokens
are eligible; S/E and padding remain. Validation uses deterministic clean views.
Each pitched view is independently transposed within ±7 semitones and valid MIDI
bounds; drum pitches and timing are unchanged. Beat pairs prefer different
transpositions for the two views, with at least one nonzero shift wherever MIDI
bounds permit. A mixture is transposed consistently across its pitched notes.
This discourages key matching; it does not prove the absence of every shortcut.

Code is shared under `improvnet/model/encoders/`: `style.py`, `beat.py`, their
respective configs, the beat objective, frozen runtime and reference selector.
The shared encoder trainer/backbone are reused, including autograd-enabled
global negatives, AMP, checkpoint/resume, per-rank RNG and data-position restore.

## Beat pairs and temporal negatives

Only candidate headers declaring at least two instrument stems are selected.
Parsed notes must actually support both views; there is no fallback to a solo
piece. The current eligible pool is 109,164 training records and 187 validation
records, with existing composition folds/copy protection retained. ARIA solo
piano records are excluded from this encoder. MIDI instruments are the existing
tokenizer's instrument stems; tracks sharing one instrument class are combined
by that representation.

For each example, randomly partition audible instrument stems into two disjoint,
nonempty groups. All stems can participate. This includes piano versus drums,
piano+drums+violin versus sax, and piano+drums versus violin+guitar+sax. Half the
partition draws favor placing piano and drums on opposite sides when available;
the rest permit them on the same side. Existing piano/drum file preference and
jazz/blues preference match the style sampler.

The two groups share one window start **and** end. Offbeats retain their original
relative positions; views are never independently rebased. Whole onset groups
fit within each view's 512-token cap, and releases are clipped to the common
window end. Both views need at least two attack groups; sparse stems can be
shorter than 512 tokens.

Each anchor pair also samples two views from another window of the same file,
separated by at least 1,000 ms. If no such window exists, a different canonical
piece from the same fold supplies the explicit negative. Other files in the
global GPU batch supply additional negatives. Temporal eligibility is computed
per candidate: overlapping/nearby same-recording windows are blocked, and alternate
recordings of the same canonical piece are not compared using incompatible
absolute clocks. Same-piece distant-window negatives are genuine negatives in
this objective; they are not accidentally removed by the style loss's piece mask.

A beat example encodes four excerpts (two positive views and two explicit-negative
views), while a style example encodes two. Backbone and anchor batch size are
unchanged; the explicit negatives add work. The objective is symmetric in both
positive directions and differentiates through global candidate gathers.

A same-piece negative can repeat a legitimate groove, so inspect held-out
same-window versus different-window retrieval as well as listening results.
Contrastive compatibility does not impose a hard beat grid on generated MIDI.

## Finetuning: both fixed, target excluded

The one existing `improvnet/model/melody_config.py` remains the finetuning config.
It uses the previous 2× non-jazz candidate budget, four jazz/blues passes,
original AR initialization, 24 examples/GPU, accumulation 1, and 20,000 updates.
Three tasks remain: `refine` (20%), `harmonize` (40%), `stem` (40%). Source windows
are capped at 512 tokens; responses at 512 for refinement and 2048 for mix tasks;
left history at 2048; total AR context including conditions stays within 8192.

`STYLE_ENCODER` and `BEAT_ENCODER` are independent frozen checkpoints. Each is
copied to an immutable, content-hashed snapshot inside the finetuning run.
Both run in evaluation mode, without masking or gradients, outside the AR
optimizer. Two separate trainable, initially neutral projections connect their
256-value embeddings to the AR model. Presence masks gate absent conditions to
zero. Resume/inference require the exact pinned encoder identities.

For harmonize and stem, each condition independently has probability
`REFERENCE_PROB=0.75` of being attempted. Its reference is a different neighboring
excerpt, within 30 seconds and at least 250 ms away from the target's complete
sounding interval. That interval includes releases beyond the nominal target
window, across all instruments. Crossing held notes are excluded. No safe
neighbor means no condition; target tokens are never used as a fallback.

The style reference matches the clean target's instrument set: a piano target
gets piano, a mixture gets its corresponding mixture. The beat reference uses
neighboring full-context notes, including drums when available; its cross-track
embedding space supports individual instruments too, so solo target pieces can
still supply piano-only beat context. The two references may overlap each other,
but neither may overlap the target. References are separate encoder inputs and
do not consume the AR token budget.

Refinement receives neither embedding. For all tasks, left history is shown 75%
of the time. Refinement history is monophonic; other tasks use polyphonic target
history. Thirty percent of shown non-refinement histories get small training-only
errors. The previous ornament-first melody corruption, at least 60% harmonization
timing masks, ±50 ms stem timing jitter, skyline-protected 0–5% accompaniment pitch
corruptions within ±7 semitones, and kit-only drum restrictions remain.

A trained beat checkpoint is required before finetuning can start. The trainer
fails clearly if it is absent. Previous protocol-10 fine-tuned checkpoints require
their previous code; new protocol-11 weights are needed for dual conditioning.

## Inference: same or different reference regions

One canonical implementation (`melody_inference.py`) and the thin
`parallel_inference.py` entrypoint share batched refinement → harmonization →
stem addition. The reference MIDI is read once. `--reference-regions same`
(default) uses exactly the same excerpt for both encoders;
`--reference-regions different` uses non-overlapping excerpts of that file,
including release/guard protection. A file too short for different regions
raises a clear error rather than silently reusing the same region.

Style uses the solo-piano view for harmonization and the piano+drums view for
stem addition. Beat uses the chosen piano/drum context embedding for both stages.
Both embeddings are computed once per reference selection and reused across
chunks. Refinement remains unconditioned. Other reference instruments are
excluded, preserving the piano-to-jazz-with-drums focus. Auto-selection uses
training-fold jazz/blues references and prioritizes piano/drum examples.

After training the beat encoder **and then the V8 fine-tuned model**:

```bash
cd /e/home/jusers/bhandari1/jupiter/improvnet_melody_contrastive
python -m improvnet.inference.parallel_inference \
  --model /e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/melody_refinement_v8/best_model.pt \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --output improvnet/inference/clair.mid \
  --genre jazz --task pipeline --instruments "Acoustic Piano" drum \
  --melody-strength .2 --rhythm-strength .9 \
  --style-reference /path/to/jazz_piano_and_drums.mid \
  --reference-regions same
```

Replace `same` with `different` to compare regions. Progress is always shown.
Output names remain short, and JSON records both reference intervals, encoder
identities and each chunk's conditioning flags. Requesting drums runs the stem
stage even when the source MIDI originally contained drums.

## Launchers (prepared, not submitted)

The current scripts follow the supplied single-node launcher structure: one node,
four GPUs, synchronized checkpoint before timeout, optional automatic continuation,
and explicit per-job out/err log folders. To start the **new beat encoder** later:

```bash
sbatch /e/home/jusers/bhandari1/jupiter/improvnet_melody_contrastive/run_scripts/beat_launch.sh
```

- Beat settings: `improvnet/model/encoders/beat/config.py`; artifacts `artifacts/beat/`.
- Existing style settings: `improvnet/model/encoders/style/config.py`; artifacts `artifacts/style/`.
- Finetuning: `run_scripts/melody_launch.sh`; artifacts `artifacts/melody_refinement_v8/`.
- Style training/resume: `run_scripts/style_launch.sh`.
- Slurm out/err: `run_scripts/{beat,style,melody}_logs/slurm-JOBID.{out,err}`.
- TensorBoard: encoder artifact `tensorboard/`; finetuning `runs/melody_refinement_v8/`.

Older versioned launcher copies are retained as historical files; use the three
short current names above. No training or successor job was submitted by these
changes. CPU tests use temporary artifacts only.

## Verification

`python -m pytest -q` runs melody/reference tests including the new beat cases.
Also run `python -m pytest -q tests/test_continuity.py tests/test_continuity_distributed.py
tests/test_continuity_continuation.py tests/test_beat_distributed.py` for the shared
trainer, continuation, and distributed gradient checks. Tests cover aligned clocks,
disjoint stem partitions, key/masking augmentation, explicit temporal negatives,
false-negative masking, frozen gradients, AR initialization/resume, independent
condition gates, batched decoding and both reference-region CLI modes.

The real-corpus CPU audit sampled 32 valid beat pairs (33 attempts), all with
separated same-piece negatives and views within 512 tokens. A tiny new beat model
trained, checkpointed, resumed and loaded through this exact data path. The
renamed existing style weights loaded with the same SHA256 as before the rename.
No GPU/Slurm training was run as part of verification.

Final checks passed: 247 distinct tests plus 3 distributed subtests, including
an end-to-end two-rank beat-encoder training/checkpoint/resume check with the
four-view forward path. Launcher syntax, CLI help, branch/worktree paths and
legacy style-checkpoint resume compatibility were also verified. Production
beat/V8 checkpoints remain absent, consistent with the instruction not to run jobs.

## Encoder source layout and dataset manifests

Each encoder has a dedicated package under `improvnet/model/encoders/`:

```text
encoders/
  style/
    config.py    # Standalone style settings, each setting explained
    data.py      # Piece-level style-pair dataset
    model.py     # StyleEncoder with independently trained parameters
    train.py     # Style training/resume entry point
  beat/
    config.py    # Standalone beat settings, each setting explained
    data.py      # Aligned disjoint-track pairs and temporal negatives
    model.py     # BeatEncoder with independently trained parameters
    train.py     # Beat training/resume entry point
  model.py       # Shared bidirectional transformer backbone
  train.py       # Shared optimizer, validation, distributed and checkpoint engine
  data.py        # Shared piece sampling, masking and pitch augmentation
  objective.py   # Beat contrastive objective
  runtime.py     # Frozen checkpoint loading for finetuning and inference
  reference.py   # Safe reference-window selection
```

Run `python -m improvnet.model.encoders.style.train` or
`python -m improvnet.model.encoders.beat.train`; the existing launchers use these
entry points automatically. The original continuity metric imports remain small
compatibility entry points to the shared model/trainer. Existing checkpoint
parameter names are preserved and historical dataset paths are normalized when
checking resume settings. Artifact paths and batch sizes are unchanged.

`MANIFEST` is the path to a JSON **dataset catalogue**, not a MIDI or checkpoint.
It records source locations/indices, training/validation/test piece assignments,
composition-group and duplicate information, and sampling metadata. Notes are
loaded from the referenced source records, then pairs and augmentations are
constructed dynamically. Both encoders intentionally use `melody_manifest_v5.json`
to preserve the original encoder cohort; melody finetuning uses its own manifest.
The filename is historical and does not select the model version. Beat training
further selects eligible multitrack pieces from that cohort. Keep this catalogue
and its referenced corpus available when reproducing encoder training; it is not
needed to run inference with a trained encoder checkpoint.

## Jazz/blues reference catalogue and trio inference

`improvnet/inference/reference_catalog.json` contains 40 training-fold examples:
8 blues and 32 jazz from Aria, GigaMIDI, Doug McKenzie, JazzSamba, JTD, PiJAMA,
Weimar and Wikifonia. Selection is balanced per genre/source, with solo-piano
and piano/drum mixtures where available. Genre labels are inherited from the
training data; the catalogue is not a manually annotated groove dataset.

Every entry records a readable title (or original filename), original MIDI path,
source, genre, instruments and note counts, plus an exact reproducible reference
excerpt (milliseconds, token count, instruments and selection seed). The record
locator points to the original prepared JSONL row. Automatic selection uses this
catalogue, balances source datasets, and prefers drum-containing excerpts within
a source when available. It never selects held-out validation or test pieces.
If the catalogue is absent and no catalogue selector was requested, the previous
training-manifest selection remains available.

The `rhythm` object describes the selected excerpt: estimated BPM and pulse
confidence, offbeat fraction/evidence, timing spread, piano attack density,
median onset interval/variation, and drum-hit density. Timing summaries group
performed chord attacks within 50 ms. These are browsing estimates, not model
conditions or certified swing/shuffle classifications. Tempo may be half/double
the perceived pulse and low-evidence feel estimates are marked `uncertain`.

`style_bucket` and `beat_bucket` each use four deterministic cosine/spherical
k-means groups from the frozen encoders. Style groups use solo-piano embeddings;
beat groups use the corresponding piano/drum excerpt (or piano alone for solo
recordings). Bucket membership denotes embedding similarity, not an annotated
groove. Instrument content, tempo and source can influence it. Bucket summaries
list their members, genres, sources and estimated tempo range. Checkpoint hashes
ensure the catalogue is not silently reused with different encoder weights.
`--reference-regions different` changes the beat excerpt at generation time;
the saved beat bucket still describes the catalogue's original excerpt.

For pipeline inference, the default requested mix is the input's pitched
instruments plus drums and **Acoustic Bass** (the tokenizer's jazz double-bass
class; MIDI program 32). Explicit `--instruments` overrides that default, so
`--instruments "Acoustic Piano" drum` still produces a duo. The trio stages are:

1. Monophonic melody refinement.
2. Piano rhythm and harmony, excluding instruments scheduled as missing stems.
3. Drum stem addition against the piano mix.
4. Acoustic-bass stem addition against the generated piano/drum mix.

At inference, `--melody-strength 0` bypasses both ornament simplification and
autoregressive refinement. It passes the extracted skyline melody through with
its pitches, onsets, durations and velocities intact, so `refine.mid` matches
`melody.mid`. Positive strengths retain ornament-first corruption and learned
refinement. Training corruption is unchanged: training examples with zero extra
corruption can still contain mandatory ornament merges. Subsequent harmonization
and stem stages still run; a nonzero `--rhythm-strength` changes the final mix's
timing even when melody refinement was bypassed. Chunk reports identify the
bypass with `decoding.end_reason=zero_strength_passthrough`.

The same shared GPU model continuously batches available requests across all
four stages. Bass can start on completed drum chunks before drums finish the
whole piece. Each stage keeps its own generated polyphonic history. Stem stages
retain source notes with the existing bounded timing/accompaniment corrections;
if a drum window produces no hits, its retained piano still flows to bass.
The reference's solo-piano style embedding conditions harmonization; the same
mixture style embedding and beat embedding condition both missing-stem stages.
Existing checkpoints already support Acoustic Bass via the missing-stem task;
no new training protocol or task token is introduced.

Trio outputs use `refine.mid`, `harmonize.mid`, `drums.mid`, `bass.mid`, plus
`final.mid` and `report.json`, all inside the same run folder. For example,
`--output /path/claire_v8.mid` writes the final mix to
`/path/claire_v8/final.mid`; it no longer creates a MIDI alongside that folder.
A single-stem invocation retains its
existing `stem.mid` name. The report distinguishes `stage_order`, individual
stem counts/missing instruments, selected reference ID/buckets and batching
statistics. Adding a requested instrument is learned generation; inspect its
reported count and listen to the result rather than assuming musical quality.

Example with a blues reference from the catalogue:

```bash
python -m improvnet.inference.parallel_inference \
  --model /e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/melody_refinement_v8/best_model.pt \
  --input /e/home/jusers/bhandari1/jupiter/improvnet_melody_contrastive/improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --output /e/home/jusers/bhandari1/jupiter/improvnet_melody_contrastive/improvnet/inference/clair_trio.mid \
  --genre blues --instruments "Acoustic Piano" drum "Acoustic Bass" \
  --melody-strength .1 --rhythm-strength .9 \
  --reference-id blues-gigamidi-7769fe584d
```

Replace `--reference-id` with `--reference-bucket beat_0` to sample a matching
bucket, or omit both to sample from the catalogue. Use `--genre jazz` with a
jazz entry. `--style-reference /path/to/performance.mid` still accepts a custom
MIDI and is mutually exclusive with catalogue ID/bucket selection. No extra
orchestration stage is used.

Rebuild the small catalogue after changing the corpus or encoder weights with:

```bash
python -m improvnet.scripts.melody_style_references --per-source 4 --buckets 4
```

The earlier `style_references.json` contains historical numerical-condition
summaries and is not used by this embedding-based catalogue or inference path.
