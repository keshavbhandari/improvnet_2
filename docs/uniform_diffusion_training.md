# Uniform diffusion music SFT

This is an independent training pipeline for `UniformDiffusionModel`, initialized
from the existing AR checkpoint. It does not use or alter the old TwoTower
trainer, inference, corruption rules, AR weights on disk, or shared preprocessing.
No production training job has been launched as part of implementation.

## Recipe

The uniform corruption follows [DiffusionGemma section 4](https://arxiv.org/html/2608.00146v1#S4).
The joint objective and 50% self-conditioning follow its
[open-source downstream SFT recipe, section 8](https://arxiv.org/html/2608.00146v1#S8).
This is full-weight adaptation of our dense music backbone, not Gemma's MoE,
LoRA, sampler distillation, or reinforcement learning.

For each training example:

1. Normalize the source using the same AR tokenizer, including split drum events.
   Select a nonempty target of up to 256 tokens first, then up to 4096 preceding
   tokens. Select the sequence-ending window with probability 5%; otherwise
   choose a token-level start, not a multiple-of-256 grid. Independently drop the
   musical prompt with probability 5%. Short sources may naturally have no prompt.
2. Keep genre and multi-hot instrument conditioning in every example, including
   promptless ones. Instrument labels come from the selected clean history and
   target; genre labels remain the dataset's true genre, not forcibly jazz.
3. Draw one continuous `t ~ Uniform[0,1]` per canvas. Independently replace each
   valid token with probability `t` by a token drawn uniformly from the **entire**
   vocabulary. The random token may equal the original. `t=0` is clean and `t=1`
   is uniform noise. Structural tokens, note fields and timing fields all follow
   the same rule. There is no special MASK operation, typed replacement schedule,
   four-draft trajectory, or fixed 16-step training grid.
4. Encode the clean history plus target causally for the auxiliary AR objective.
   Before denoising, restrict cache visibility to the preceding history and
   controls. The clean target is hidden in BOTH self-conditioning and denoising
   passes. Promptless examples expose controls only.
5. For 50% of examples, use a detached initial prediction of the **same noisy
   canvas** as soft self-conditioning; use zeros for the remainder. The bootstrap
   itself has no gradients. The final pass backpropagates through both the
   original context cache and the shared weights. Self-conditioning temperature
   is explicitly configured as `SELF_CONDITIONING_TEMPERATURE = 1.0` during SFT.
   The paper does not separately prescribe an SFT temperature: this value is
   verified against the released adapter, which passes raw first-pass logits to
   [`encode_logits`](https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/gemma4/_modules.py)
   and applies softmax without temperature scaling. The paper's 0.8-to-0.4
   annealing schedule applies to inference, including inference self-conditioning.
6. Minimize `mean_canvas_CE + AR_LOSS_WEIGHT * mean_next_token_CE` (default weight
   1). Both terms average valid tokens within each example, then examples within
   the batch. Every valid canvas position has equal weight, including unchanged
   tokens. AR logits at musical position `j` predict `j+1`; the first musical
   token has no AR label from a preceding musical position. Pads and conditioning
   tokens do not contribute target labels.

Sampling a target-first local window is our music-data adaptation rather than
the report's fixed `P + K*C` sequence layout with uniformly selected canvas index.
It retains variable history, endpoint coverage, and one supervised canvas per
example without empty blocks. A short target is padded to 256 with an explicit
mask. A valid position randomly replaced by the PAD ID is **not** padding and
still attends/receives loss.

This first baseline is **rigid length**: elasticity is explicitly zero. It does
not create BLANK targets or remove instrument stems. Those require a separate
music-specific insertion objective and should not be silently inherited from
the old absorbing-mask model. Noise can still randomly sample BLANK/MASK/PAD as
ordinary vocabulary entries; those are not desired clean targets.

The genre controls and partial corruption provide the basis for later style
transfer experiments. This is not paired classical-to-jazz supervision, and
does not by itself guarantee preservation of the original composition. Later
inference must specify which source fields are preserved and how to initialize
the noisy canvas. The old masked-diffusion sampler is not appropriate here.

## Files and defaults

- `improvnet/model/uniform_diffusion_train_config.py`: paths and training settings.
- `improvnet/model/uniform_diffusion_train.py`: single-device / torchrun trainer.
- `improvnet/utils/uniform_diffusion_data.py`: indexed dataset and padding.
- `improvnet/utils/uniform_diffusion_corruption.py`: forward noise and CE.

Starting defaults are batch 24/GPU, accumulation 1, validation batch 16, up to 100
validation batches **per rank**, 20,000 optimizer updates, LR `2e-5`, 500 warmup
updates, cosine decay to `2e-6`. The batch was checked on a GH200; the learning
rate/update budget remain starting points, not a convergence guarantee.
The effective batch is `24 * world_size` (192 across eight GPUs). Unlike the earlier TwoTower model,
the context encoder now needs backward activations; its batch 64 should not be
assumed to fit here. Review learning curves before setting a final budget.

The trainer uses gradient checkpointing, CUDA BF16 where supported (otherwise
FP16 with scaling), accumulated gradients, clipping, and chunked/checkpointed
fp32 CE. Validation releases training gradients first, disables autograd, uses
its own batch limit and repeats fixed windows/noise/self-conditioning choices
for an unchanged validation configuration. It does not advance training RNG.

TensorBoard and stdout report total loss, separate denoising/AR losses, accuracy
over all valid positions, accuracy over **actually changed** positions, and mean
noise level. The changed-position accuracy helps expose a low loss obtained
mostly by copying already-clean tokens. Best checkpoints use denoising loss,
not the sum of two objectives.

## Launching later

From the repository root, after activating the `improvnet` environment:

```bash
python -m improvnet.model.uniform_diffusion_train --help
# Single GPU (starts training when explicitly run):
python -m improvnet.model.uniform_diffusion_train
# Four GPUs on one allocated node:
torchrun --standalone --nproc_per_node=4 -m improvnet.model.uniform_diffusion_train
```

For multiple allocated nodes, use normal torchrun rendezvous/node-rank options
and this **new module**, not the old TwoTower launcher unchanged. This change
does not submit jobs or modify existing SLURM scripts. CLI overrides include
`--ar-model`, `--save-dir`, `--batch-size`, `--accum-steps`, `--steps`, and
`--config` (an importable settings module).

Fresh runs strictly copy all compatible AR backbone weights and initialize only
the new diffusion modules. All model parameters then train. The default output
directory is `artifacts/uniform_diffusion_v1`, separate from AR and TwoTower.
The trainer rejects using the source AR directory as its destination.

If `latest_checkpoint.pt` exists, resume restores model, optimizer, scheduler,
scaler, per-rank RNG, epoch, and next local batch. Saves are atomic and occur at
optimizer boundaries. With `--no-resume`, an existing latest checkpoint causes
an error instead of being overwritten. Exact resume currently requires the same
world size, batch/accumulation, data settings and optimization schedule; changes
raise an error. It is not an automatic warm-start/rescale feature. Data windows
and pitch augmentation are deterministic by example and epoch, independent of
worker count and prefetching. Dedicated loader RNGs keep resumed noise/dropout
streams aligned. Bitwise GPU reproducibility still depends on kernel behaviour.
Data file handles are also reopened after a worker fork, including when a parent
inspected examples before constructing its DataLoader.

For compatible SLURM preemption handling, set `CHECKPOINT_REQUEST_FILE`; once that
file exists on any rank, all ranks save at the next optimizer boundary and exit.
The launcher must clear/use a fresh request file before restarting. The trainer
does not install SLURM signal handlers or delete request files itself.

## Verification

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_uniform_diffusion*.py' -v
```

Tests use tiny CPU models and temporary synthetic datasets/checkpoints. They
cover uniform-noise endpoints and frequencies, identity replacements, drum
normalization, genre/instruments, window endpoints, promptless/partial canvases,
padding and all-token CE gradients, clean-target leakage, self-conditioning,
checkpointed shared-backbone backward, deterministic validation, exact resumed
versus uninterrupted updates across epochs, and two-process CPU DDP with unequal
validation shards. They do not establish production GPU memory usage or musical
generation quality. Inference and sampler implementation remain deferred.

## GH200 memory audit

The full AR-initialized model was tested on the shared debug GH200 (95 GiB),
with about 27 GiB already occupied. These are disposable optimizer steps, not a
production run: no checkpoint was written, no job submitted, and no other
process was modified. Tests used BF16 autocast, FP32 trainable parameters/AdamW
states, checkpointed backward, 4096 history + 256 target tokens, and 100%
self-conditioning to stress the extra pass. Optimizer states were included by
running multiple updates, followed by validation.

Before the causal-padding fast-path optimization:

| Batch/GPU | Worst training allocated | Worst training reserved | Validation allocated |
| --- | --- | --- | --- |
| 8 | 24.72 GiB | 26.70 GiB | 14.32 GiB |
| 16 | 36.93 GiB | 41.20 GiB | 18.06 GiB |
| 20 | 43.01 GiB | 49.08 GiB | 19.93 GiB |
| 24 | 49.04 GiB | 51.88 GiB | 21.74 GiB |

Batch 24 used `PYTORCH_ALLOC_CONF=expandable_segments:True` to avoid fragmented
reservations under a deliberately restricted 52 GiB allocator budget. Earlier
24-example attempts reached the artificial 48/52 GiB limits despite physical
device memory remaining free; those are not measurements of the GH200's maximum
batch. The benchmark stops instead of exhausting the shared device.

Right-padded causal prefill now uses the causal fast path: valid queries precede
all padding keys, so causality already hides them. Ignored padded-query outputs
stay excluded from loss and decoder cache visibility. Interior padding and
incremental appends retain explicit masks. CPU equivalence tests cover both.
After this optimization, a repeated batch-24 test peaked at **47.77 GiB allocated
/ 51.80 GiB reserved** in training and **21.31 GiB allocated** in validation.
The padded case dropped from approximately 7.77 to 3.92 seconds/update (three
updates, same shared GPU; indicative timings, not a dedicated throughput study).

The test does not measure multi-node communication or DDP memory buffers. Batch
24 deliberately leaves room for those on dedicated 95 GiB GPUs; 32 or higher
has not been validated. Use batch 16 for additional margin on the shared debug
GPU, and recheck free memory before profiling.

The reusable bounded benchmark is:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 \
python -m improvnet.scripts.benchmark_uniform_diffusion \
  --batch-sizes 16 24 --steps 3 --memory-limit-gib 52 --headroom-gib 12
```

A read-only check of 64 records across both actual JSONL files also passed
tokenization, drum normalization, genre/instrument conditioning, collation and
target validity checks. This was a small prefix sample, not a representative
genre-balance audit or evidence of generation quality.
