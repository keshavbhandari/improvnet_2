# Stage 2: continuous latent diffusion

The trained local VAE is frozen. The `vae176_vector512` pilot defaults generate
one 512-value latent vector over each of **30 adjacent one-second bins**, with up
to **8 preceding clean bins** in the same
canvas. Each bin has its own noise embedding: history stays at zero noise, target
bins share a randomly sampled diffusion time. Loss applies only to valid target
latents, with fractional weights for partially valid seconds (or partially padded
columns for legacy spatial VAEs). The first 5% of
sampled windows start at the beginning of a piece, without history.

Model, training and values-only config are `latent_diffusion_model.py`,
`latent_diffusion_train.py`, and `latent_diffusion_config.py`. Defaults are a
512-wide, 16-layer transformer, cosine Gaussian
diffusion, v-prediction, deterministic DDIM sampling, and EMA inference weights.
This is an adaptation, not a reproduction of Huang's or TD-Paint's architecture.
It does not reuse or modify the AR/discrete-diffusion models.

The vector VAE exposes shape `[512,1,1]`; 1x1 patches yield **one token per second,
38 tokens per window**, rather than the old spatial run's 32,000 tokens. Every
token contains the full second's vector. Musical conditioning tokens are unchanged.
Train/evaluate the new VAE before starting diffusion; both stages use fresh
`latent_diffusion_vae176_vector512_30s_pilot` run paths. Normalization is calibrated afresh for all 512
coordinates. Existing diffusion weights cannot be resumed with this encoder.

The pilot uses `vae176_vector512/best_model.stage2_pilot_30s.pt`, a pinned
snapshot of the current best VAE, so ongoing VAE training cannot replace the
encoder used by diffusion. The snapshot selected for this pilot is step 180,000.
The 30-second target is separate from its up-to-eight-second history: each
training window therefore covers up to 38 seconds. `N_STEPS=100000` remains the
optimizer-update budget; it does not mean audio duration. A shorter pilot budget
can be chosen with `--steps` before starting a fresh run, but the saved schedule
and update budget must match on exact resume. No diffusion training was started
while checking this configuration.

## Musical conditions

- **Soft pitch:** 128 binary values per bin; a pitch is present if active anywhere
  in that second, including held notes. Projected as a conditioning token, never
  an allowed-vocabulary mask. Whole references drop out 10% of examples; training
  also erases 10% of present bits, allowing completion rather than exact copying.
  A presence flag distinguishes absent guidance from genuine silence.
- **Genre:** all supplied training-index tags have stable IDs saved in the
  checkpoint. Multi-tag files sample one of their real tags. Unlabelled files and
  10% conditioning dropout use an unconditioned ID, not an invented genre.
  GigaMIDI uses the supplied `music_styles_curated` metadata, not scraped period,
  language, or finer-style labels. Unlabelled rows remain unconditioned.
- **Instrument:** target GM program and drum flag; MIDI channel is not identity.
  The tokenizer's piano-first target selection remains unchanged. Accordingly,
  learning arbitrary non-piano targets in piano-containing mixtures will require
  a separate target-selection/data experiment later.
- **Accompaniment:** full mode 45%, rhythm-only 45%, absent 10%. Up to three parts
  are selected; a part is `(MIDI track, channel, program)`, so multiple instruments
  in one MIDI container are not collapsed into one part. Training samples parts;
  inference selects the most populated available parts for each window.

### Accompaniment representation

**Full mode:** one token per note, built from separate embeddings for part slot,
program, drum flag, pitch, plus velocity and signed start/end positions on the
shared fine-frame clock. Notes already sounding at a window's start are included.
No flattened vocabulary or interleaved multi-token grammar is required.

The **1024-event budget applies to the entire history + target window**. Pruning
round-robins across part/onset-bin buckets, rather than retaining only the first
1024 notes. Omitted event counts are printed and included in the inference report.
Budgeted events are a lossy condition, not a change to the source accompaniment.

**Rhythm-only mode:** calculate per-track onset count, active-note count and mean
attack velocity at every 10 ms frame, with no pitch values. Counts use `log1p`;
10-frame groups project to one conditioning token (100 ms by default). Track
program/drum identity remains available. Curves use **all selected notes before
event pruning**, so the event budget does not thin the groove. Missing tracks,
events, history bins and trailing frames are masked from attention.

Training accompaniment comes from the same piece as the target. An external jazz
MIDI can supply rhythm at inference, including its piano part. There is **no
automatic beat/tempo alignment**: use a pre-aligned excerpt, or explicit time
scaling/offset. An unrelated groove is a distribution shift; adherence is not
guaranteed. Training reconstruction with native genre tags is also not paired
classical-to-jazz supervision.

## Commands

### Genre-labelled index

Stage 2 uses `midi_index_gigamidi_genres.sqlite`, a separately published copy of
the VAE index with unchanged paths, row IDs, groups and splits. The original VAE
index/checkpoints are untouched. The CSV now lives inside the GigaMIDI scratch
data folder. Curated labels cover **71,791 of 2,136,218 GigaMIDI paths** (3.36%).
No guessed label is assigned to the rest. The dataset authors identify the
[curated column as more reliable than scraped labels](https://huggingface.co/datasets/Metacreation/GigaMIDI).

The training vocabulary has **28 genres**, plus one unconditioned ID:
alternative, ambient, atonal, blues, breakbeat, classical, country, dance, disco,
downtempo, drum&bass, edm, folk, game, house, jazz, latin, metal, pop, punk, ragtime,
rap, reggae, rock, soundtrack, techno, trance, world.
All available labels are eligible; this is not uniform genre-balanced sampling.
Very rare labels have little supervision. Multi-tag rows sample a real tag;
unlabelled rows and genre dropout use ID 0. `game` and `soundtrack` stay distinct
as provided by the source metadata.

The metadata-reader handles the actual extended CSV columns and oversized loop
fields, and supports older genre-column CSVs. To reproduce the refresh with a
fresh output index configured in `latent_diffusion_config.py`:

```bash
python -m improvnet.scripts.update_midi_genres
```

This command refuses to overwrite an existing index. Expanding the genre vocabulary
changes the embedding shape/IDs: old stage-2 debug checkpoints are not exact-resume
compatible. Use a fresh stage-2 SAVE_DIR if one was saved; never delete VAE weights.

### Training and inference

Run from the repository with the `improvnet` environment active. No new MIDI index
is needed; stage 2 uses the separately genre-labelled copy of the VAE index, grouped splits, collection-balanced
sampling, 5% GigaMIDI drum-only subset, and consistent pitch transposition of target,
history and pitched accompaniment. Latents use the VAE posterior **mean**, not a
fresh posterior sample. Mean/std normalization is calibrated on training targets
and saved with the stage-2 checkpoint.

Read-only data/model forward check (no backward, optimizer, training or saved files):

```bash
python -m improvnet.model.latent_diffusion_train --check
```

Training, **only when you choose to start it**:

```bash
torchrun --standalone --nproc_per_node=4 -m improvnet.model.latent_diffusion_train
```

The default batch is 32 windows/GPU, accumulation 1; it is not a measured maximum.
Training logs v-loss and clean-latent normalized RMSE, including low/mid/high
noise groups. Low = t<0.25, mid = 0.25–0.75, high = t>=0.75. These are one-pass
predictions, not full-sampler errors or a VAE robustness verdict. Validation noise
and dataset choices are repeatable. Checkpoints include optimizer, RNG, data
position, EMA, normalization, genre IDs, and frozen VAE SHA-256. Resume requires
matching architecture/data/world/optimizer settings. VAE checkpoints and unrelated
SAVE_DIR checkpoints are protected from overwrite.

The current frozen checkpoint is `vae/vae176_latent4/best_model.pt` (92k steps,
four latent channels), with saved start/onset cutoffs of `.95/.985`, held-note
continuation `.70`, and a 10 ms internal gap allowance selected on validation
sequences. Stage-2 inference applies the same held-note decoding as VAE inference,
after assembling all generated blocks and the explicitly preserved prompt.
Pedal/accompaniment are unaffected, and no reference target informs decoding.
`--note-decoding legacy` restores the earlier assembly; also set onset `.995`
when comparing with the original baseline. The metadata-only calibration changes
the checkpoint file digest but leaves encoder/decoder weights and latents intact.
The fresh run directory is
`artifacts/latent_diffusion_vae176_latent4`; older `latent_diffusion_v1` checkpoints
use a different encoder and remain separate. Latent dimensions are read from the
VAE checkpoint, so no separate eight-to-four channel edit is needed. The 128
normalization samples are training **windows**, containing up to 16,384 target
bins with the current window length; these statistics are recomputed when a new
stage-2 run starts.

To resume with a different per-GPU batch size, explicitly allow the non-exact
continuation (same SAVE_DIR):

```bash
torchrun --standalone --nproc_per_node=1 -m improvnet.model.latent_diffusion_train \
  --batch-size 16 --allow-batch-size-change
```

Model/EMA, optimizer, scheduler, normalization, RNG and step count are retained.
Data restarts at an epoch boundary because the saved batch index no longer maps
to the same examples. Learning rate is not automatically rescaled. This flag
does not permit changes to architecture, world size, data or other optimizer settings.

Transfer, starting without history and preserving the source's soft pitch guide:

```bash
python -m improvnet.inference.latent_diffusion_inference \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --genre jazz --task transfer --passes 48 --noise-strength 0.8 --max-bins 32 \
  --accompaniment-mode rhythm
```

External groove (use a pre-aligned jazz excerpt; scale/offset are in seconds):

```bash
python -m improvnet.inference.latent_diffusion_inference \
  --input piece.mid --genre jazz --accompaniment-mode rhythm \
  --accompaniment-midi jazz_groove.mid --reference-time-scale 1.0 \
  --reference-offset-seconds 0.0 --max-bins 32
```

Reconstruction diagnostics: use the source's actual genre (or `--genre none` if
unknown), and repeat with strengths such as 0.2, 0.5, 0.8 and 1.0:

```bash
python -m improvnet.inference.latent_diffusion_inference \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --task reconstruct --genre classical --noise-strength 0.2 --passes 48
```

Novel continuation, without source pitch guidance or source-latent initialization:

```bash
python -m improvnet.inference.latent_diffusion_inference \
  --input piece.mid --task continue --prompt-bins 4 --noise-strength 1 \
  --genre jazz --accompaniment-mode none --max-bins 32
```

`--checkpoint` defaults to the latest stage-2 checkpoint; EMA is used unless
`--raw-weights` is supplied. The associated VAE is hash-checked. `--target-channel`
and `--target-program` select the source part. `--max-bins` counts generated bins,
excluding a preserved prompt. Transfer/reconstruction stop at source exhaustion;
continuation can extend it. History uses committed generated latents directly,
avoiding repeated decode/re-encode degradation. MIDI decoding assembles the entire
roll continuously, so sustained notes are not split at bin boundaries.

Inference restores window lengths, patch sizes, width, depth, heads and latent
normalization from the selected diffusion checkpoint, rather than the current
training config. The retired spatial `[4,32,25]` VAE with 1x1 patches used 800
tokens per second and 32,000 tokens per window. The vector `[512,1,1]` pilot uses
one token per second and 38 tokens per window. Generation advances by 30 target
bins and retains the last 8 generated latent bins as history. Short final windows
are padded and masked. `--max-bins` controls total generated length independently
of the trained window size, so longer outputs use multiple windows.
The report records the restored settings under `diffusion_architecture`.
Conflicting saved model/window settings are rejected before sampling. Changing
the live config cannot turn an older checkpoint into a new architecture; choose
a checkpoint trained with the desired settings using `--checkpoint`.

## Reading the small report

Outputs go to a fresh scratch evaluation directory, or `--output-dir` (must be
empty/new). Four MIDI files let you hear the source target, VAE reconstruction,
generated target, and generated target with the **original source accompaniment**.
External conditioning MIDI is deliberately not mixed automatically into output.

The concise `report.json` separates:

- `vae_only`: stage-1 reconstruction floor: activity/onset/note F1, extra/missed
  notes, velocity/pedal errors, and matched note timing errors.
- `generated_vs_source`: the same measures after stage 2. For transfer or novel
  continuation these are **reference distances**, not necessarily mistakes.
- `latent_nrmse_vs_source`: final latent RMSE divided by training channel std.
  For reconstruction this is a useful stage-2 residual diagnostic. It is not the
  fraction of corrupted notes, nor the maximum allowed forward diffusion noise.
- Reconstruction only: `vae_jitter_at_measured_residual`, one isotropic decoder
  stress test at that RMSE. Actual diffusion residuals may be structured; this
  comparison cannot prove that jitter training will help.

A good VAE baseline but poor reconstruction sampling points first to stage 2.
A poor baseline indicates stage-1 limitations. Strong jitter sensitivity warrants
closer decoder robustness testing, not automatic VAE retraining. Pitch/genre/style
changes are not encoded in a perfectly disentangled latent space.

Tests (synthetic, no training):

```bash
python -m unittest discover -s tests -p 'test_latent_diffusion.py' -v
```

References: [latent diffusion](https://arxiv.org/abs/2112.10752),
[v-prediction and deterministic sampling](https://arxiv.org/abs/2202.00512),
[variable-noise inpainting](https://arxiv.org/abs/2410.09306).
