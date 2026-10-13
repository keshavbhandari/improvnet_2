# Masked diffusion on AR music tokens

This replaces the old uniform-noise architecture, SFT trainer and evaluator.
The VAE experiment stays on `hierarchical-pianoroll-tokenizer`; this work is on
`master`. No training job has been started.

## Representation and model

Use the unchanged `AbsTokenizer` and `ProcessData` normalization from AR.
Each note is four consecutive tokens: **instrument, combined pitch/velocity,
onset, duration**. There are no separate pitch and velocity tokens in this
vocabulary. Structure and pedal events retain their existing representation.

The new transformer uses AR's width, depth, grouped-query attention, RMSNorm,
RoPE and SwiGLU. Defaults are 2048 width, 20 layers, 16 query heads and four KV
heads, with at most 8192 music tokens. Attention is bidirectional, including
padding masks, and has no autoregressive KV cache. Four condition embeddings
carry genre, instrument multihot, abstract rhythm and elasticity. A small role
embedding identifies each masked attribute; it never exposes the hidden value.
Grouped-query attention is retained from AR, unlike LLaDA's particular 8B model.

The [LLaDA paper](https://arxiv.org/abs/2502.09992) motivates the objective:
sample `t ~ Uniform(0,1)`, replace eligible tokens with `<MASK>` with probability
`t`, and optimize masked-token cross entropy weighted by `1/t`. Normalize by
all valid nonpadding tokens, not just the masked subset. Structural tokens,
including `<T>`, `<D>`, `<E>` and `<S>`, are masked and reconstructed at the same
rate as attributes. Padding is excluded. Four-token blank targets share one masking draw, so they are
hidden as a complete group. Half of examples also mask real notes as whole
groups; the other half mask real attributes independently. This prevents the
four-mask pattern from identifying blank targets by itself, while providing
training examples for partial attribute editing. Every eligible token retains
masking probability `t`, so the inverse-rate objective stays unchanged. This
is a music adaptation of the absorbing-mask objective, not uniform replacement
noise, an AR auxiliary loss or a reproduction of LLaDA's language-training recipe.

## Elasticity and blanks

Each training excerpt samples elasticity `e` from `[0, ELASTICITY_MAX]`. Insert
`floor(e * number_of_notes)` empty slots at randomly chosen complete note
boundaries. Each target slot contains **four consecutive `<BLANK>` tokens**.
The same masking process also hides real notes, teaching recovery of valid
four-attribute notes as well as empty slots.

At inference the inserted slots begin as four `<MASK>` tokens. The instrument
prediction decides whether the slot is blank or occupied. Blank slots commit
all four blanks together; occupied slots predict legal instrument, note, onset
and duration values together. There are no partial notes or mixed blank groups.
Generated notes keep the source's instrument palette.
Unfilled blank groups are removed before MIDI decoding. Elasticity controls
**capacity**, not a guaranteed number of generated notes or a forced density.
Inference reports both the budget and the realized note-count change.

For existing notes:

- `timing`: mask onset and duration for the selected fraction of notes;
  instrument and pitch/velocity stay fixed.
- `pitch-velocity`: mask the combined note token for the selected fraction;
  instrument, onset and duration stay fixed.
- `notes`: mask all four attributes for the selected fraction. Up to
  `floor(e * source_notes)` selected original slots may also become blank,
  allowing deletion. At `e=0` original slots cannot become blank.

Fractions are in `[0,1]`; selection is seeded and rounded up to a whole number
of notes per window. At zero mask percentage, only any added slots are edited.
At zero elasticity no notes are added or deleted. Long sources are processed
in complete-event windows, reserving space for added slots. Existing `<T>`
markers are preserved, so the time grid and overall source timeline remain.
Windows do not share attention history. Extra slots are fully editable even
when only timing or pitch/velocity is edited in existing notes.

## Abstract rhythm condition

Compute 14 features from clean training excerpts, before corruption or blanks:

1. Six-bin histogram of **positive gaps between distinct note onsets**.
2. Six-bin histogram of **note durations**.
3. Fraction of notes sharing an onset with another note.
4. Fraction of durations longer than the median positive onset gap.

Both histograms divide times by that median gap; cut points are
`0.25, 0.5, 1, 2, 4`. If there is only one distinct onset, median duration
provides the scale. Fractions are rounded to two decimals. The condition contains
no pitch, velocity, instrument identity, note count, absolute tempo, onset order,
beat-phase template or individual timing values. It is invariant to global tempo
scaling and intentionally cannot prescribe an exact swing pattern. Global
statistics can still correlate with musical content; this limits detail rather
than proving zero information leakage. Rhythm conditioning drops out on 10% of
training examples. At inference the same extractor accepts any donor MIDI through
`--rhythm-reference`; absent a donor, it uses the source piece's statistics.
The donor is not aligned, copied or mixed into the generated MIDI.

## Data and training

`masked_diffusion_config.py` reuses AR data paths, split seed/ratios, drum-only
exclusion, architecture and optimizer schedule. Its extra controls are
`ELASTICITY_MAX`, `JAZZ_BLUES_WEIGHT=4` and `RHYTHM_DROP_PROB=.1`, plus run path,
seed and worker count. Each jazz/blues-labelled training piece has four times
the sampling weight of another piece. A deterministic shared weighted draw is
sharded over GPUs. Validation is held out and is not oversampled. The first
dataset construction scans metadata headers (not the large note arrays) and
caches genre weights alongside the JSONL;
cache identity includes source metadata, split offsets and the genre weight.
Repeated draws receive different deterministic crops/augmentations. Pitch
augmentation stays within the AR vocabulary and does not transpose drums.

Existing tokenized JSONL files and their `_index.pkl` byte indexes are required;
raw MIDI folders do not substitute for these. The existing files are under
`data_old`, and the shared AR config now points to those verified paths.
Check the paths before launching:

```bash
python -m improvnet.model.masked_diffusion_train --check
# These commands start training; they were not executed during implementation.
python -m improvnet.model.masked_diffusion_train
torchrun --standalone --nproc_per_node=4 -m improvnet.model.masked_diffusion_train
```

The sibling `run_scripts/improvnet/masked_multi_node_launch.sh` replaces the
uniform launcher, including its coordinated checkpoint request. The model starts
fresh in `artifacts/masked_diffusion_v1`; AR weights and runs are not overwritten.
Resume restores optimizer, scheduler, scaler, per-rank RNG and weighted data
position. Architecture, sampling/optimization settings, world size and data
identity must match. Validation reports the fixed-noise objective and masked
accuracy; neither metric proves successful classical-to-jazz transfer.

## Inference

```bash
python -m improvnet.inference.masked_diffusion_inference \
  --input improvnet/inference/examples/debussy-clair-de-lune_original.mid \
  --output /path/to/classical_as_jazz.mid --genre jazz \
  --rhythm-reference /path/to/jazz.mid \
  --strategy timing --mask-percentage .5 --elasticity .2 --steps 32
```

`--model` defaults to the new run's `best_model.pt`. Use `--strategy pitch-velocity`
for pitch/velocity edits, or `notes` for complete-note edits. Each pass predicts
remaining masks; high-confidence note groups commit first and low-confidence
groups stay masked for the next pass. Source and committed attributes are never
remasked. Predictions are constrained to the role's vocabulary and positive
velocity/duration. Whole-note blank decisions commit atomically. The JSON beside
the output records rhythm features, budgets and actual added/deleted notes.
`--steps` defaults to 32 and accepts any positive integer, such as 64 or 128.
It is an upper bound on passes per window: decoding ends when all editable note
groups are resolved, and each pass commits at least one remaining group.
Structural tokens remain fixed for these style-transfer inference strategies,
even though their reconstruction is learned during training.

Focused checks: `python -m unittest discover -s tests -p test_masked_diffusion.py`.
They cover masking/loss, rhythm invariance, typed edits, blanks and MIDI decoding,
bidirectional/padded attention, fourfold genre weighting and checkpoint resume.
