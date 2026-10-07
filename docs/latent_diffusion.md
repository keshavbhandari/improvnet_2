# Stage 2: continuous latent diffusion

The existing local VAE is frozen. Stage 2 generates its spatial latents over
**8 adjacent one-second bins**, with up to **4 preceding clean bins** in the same
canvas. Each bin has its own noise embedding: history stays at zero noise, target
bins share a randomly sampled diffusion time. Loss applies only to valid target
latents, with fractional weights for partially padded columns. The first 5% of
sampled windows start at the beginning of a piece, without history.

Model, training and values-only config are `latent_diffusion_model.py`,
`latent_diffusion_train.py`, and `latent_diffusion_config.py`. Defaults are a
512-wide, 12-layer transformer with spatial latent patches, cosine Gaussian
diffusion, v-prediction, deterministic DDIM sampling, and EMA inference weights.
This is an adaptation, not a reproduction of Huang's or TD-Paint's architecture.
It does not reuse or modify the AR/discrete-diffusion models.

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

The **256-event budget applies to the entire history + target window**. Pruning
round-robins across part/onset-bin buckets, rather than retaining only the first
256 notes. Omitted event counts are printed and included in the inference report.
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

The default batch is 8 windows/GPU, accumulation 1; it is not a measured maximum.
Training logs v-loss and clean-latent normalized RMSE, including low/mid/high
noise groups. Low = t<0.25, mid = 0.25–0.75, high = t>=0.75. These are one-pass
predictions, not full-sampler errors or a VAE robustness verdict. Validation noise
and dataset choices are repeatable. Checkpoints include optimizer, RNG, data
position, EMA, normalization, genre IDs, and frozen VAE SHA-256. Resume requires
matching architecture/data/world/optimizer settings. VAE checkpoints and unrelated
SAVE_DIR checkpoints are protected from overwrite.

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
