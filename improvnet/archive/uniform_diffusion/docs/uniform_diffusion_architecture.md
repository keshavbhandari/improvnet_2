# Shared-weight music diffusion architecture

This implements a DiffusionGemma-style music model.
It is a new model, separate from the existing TwoTower training and inference.
The uniform-noise SFT trainer is described in [uniform_diffusion_training.md](uniform_diffusion_training.md).
The reverse-diffusion sampler is not implemented yet.

## Model structure

`improvnet.archive.uniform_diffusion.model.uniform_diffusion_model.UniformDiffusionModel` owns **one** set
of embeddings, attention projections, transformer blocks, final normalization,
and output head. Its two execution modes are:

- `encode_context`: causal attention over clean musical history and conditioning
  tokens; produces a per-layer KV cache and optional next-token logits.
- `denoise`: bidirectional attention within the current canvas, together with
  attention to the clean prefix cache; produces same-position clean-token
  predictions. It never writes canvas keys or values into the context cache.

Repeated denoising calls use the same canvas positions. Only after a sampler
finalizes the canvas should it be passed to `encode_context(..., cache=...)`.
The old four-snapshot staircase, AdaLN, and timestep embeddings are absent.
Noise levels belong to the corruption/sampling code, matching the
reference decoder's lack of an explicit timestep input.

The context and denoising paths share parameters by construction, not by copying
weights into separately registered modules. Sharing the paths does **not** tie
the input embedding to the output head: those remain distinct as in our AR
checkpoint. The checkpoint's dense SwiGLU, GQA, Q/K normalization, RMSNorm, and
1D RoPE are preserved instead of importing Gemma's MoE-specific layers.

## Self-conditioning

Previous logits are converted into soft embeddings using
`softmax(previous_logits / temperature) @ token_embedding.weight`. A learned
pre-norm gated MLP transforms them; its output is added to the noisy token
embedding and normalized with scale-free RMSNorm. This follows the reference
implementation's input-side self-conditioning, using our backbone's SwiGLU.

Previous logits are detached by default. Current embedding and self-conditioning
parameters still receive gradients. A boolean `self_conditioning_mask` of shape
`[batch]` enables per-example self-conditioning dropout. No supplied logits means
a zero soft signal. The trainer produces these logits with a no-gradient model
pass on the same noisy canvas, not from ground-truth labels. Half the examples
use that signal; the other half use zero. The bootstrap reads a detached cache,
but the final decoder pass reads the original differentiable encoder cache.

## Music conditioning and padding

The first context pass prepends the existing AR genre embedding and multi-hot
instrument projection. Denoising reads these controls through the shared cache.
A promptless example is a `[batch, 0]` musical prompt: its context contains just
the controls. Empty history therefore does not remove genre information.

An optional `[batch]` elasticity fraction adds a third learned prefix vector.
This is a music-specific extension, not part of DiffusionGemma. Omitting it
preserves the original two-control AR computation exactly; passing zero explicitly
means rigid generation in a run trained with the extra control. Use a consistent
control layout within a diffusion training run. With DDP, always supplying
elasticity (including zeros) also ensures its parameters participate in forward;
otherwise the trainer must handle unused parameters.

Padding is controlled by explicit boolean masks. Do not infer it from the token
value: a uniform-noise process may sample the PAD vocabulary ID at a valid
position. Padding is excluded from key visibility and position counting. Padded
output logits are zero; the training loss explicitly excludes padded targets.

Both variable prompt lengths and partial final canvases are supported. The
default canvas limit is 256. Causal appends use the bottom-right mask alignment
needed for unequal query/key lengths. Attention uses PyTorch SDPA with GQA; it
does not require the external flash-attn package. Right-padded causal prefill
uses the fused causal path because valid queries cannot see later padding keys.
Interior padding and cache appends retain explicit attention masks; decoder
cache visibility always excludes padding and hidden clean targets.

## Checkpoint initialization and minimal API

```python
import torch
from improvnet.archive.uniform_diffusion.model.uniform_diffusion_model import UniformDiffusionModel

model = UniformDiffusionModel.from_ar_checkpoint(ar_checkpoint_path).to(device)
model.eval()

with torch.no_grad():
    context = model.encode_context(
        prompt_ids,                       # [batch, prompt_length]; may be empty
        attention_mask=prompt_mask,
        genre=genre_ids,                   # [batch]
        multi_hot=instrument_controls,     # [batch, 41]
        return_logits=False,
    )
    logits = model.denoise(
        noisy_canvas_ids,                  # [batch, <=256]
        cache=context.cache,
        attention_mask=canvas_mask,
    )
    # The future sampler constructs next_canvas_ids from logits.
    next_logits = model.denoise(
        next_canvas_ids,
        cache=context.cache,
        attention_mask=canvas_mask,
        self_conditioning_logits=logits,
    )
    # After the future sampler finalizes this canvas:
    context = model.encode_context(
        finalized_canvas_ids,
        attention_mask=finalized_canvas_mask,
        cache=context.cache,
        return_logits=False,
    )
```

Controls are already cached; omit them on appends. Re-encode the prefix when
changing genre, instruments, or elasticity. Remove finalized BLANK placeholders
before encoding committed musical history; that policy belongs to sampling.

`initialize_from_ar` accepts a raw state dictionary or a training checkpoint's
`model_state_dict`, with or without DDP's `module.` prefix. It validates every
backbone key and shape before copying any weights. Only self-conditioning and
elasticity modules remain newly initialized. Use normal strict `load_state_dict`
for later diffusion checkpoints. AR and diffusion token/genre/instrument index
orderings must match; tensor shape checks alone cannot detect reordered labels.

## Training integration boundary

`model(prefix_ids, canvas_ids, ...)` is a combined forward suitable for DDP. It
returns denoising logits plus a `ContextOutput`; request `return_context_logits`
only when an auxiliary AR objective needs them. Gradients flow through the shared
encoder and decoder, including context KV. `ContextCache.detach()` is explicit;
the model never silently freezes its context path. Gradient checkpointing is
enabled by default during training.

Pass only clean tokens preceding the target canvas as the prefix. Alternatively,
pass the full clean sequence with `prefix_lengths` to `forward`: it applies
`cache.prefix(prefix_lengths)` before both bootstrap and final denoising. This
creates a visibility-limited view for a chosen target. Lengths exclude controls
and padding. This view masks the target and all future keys and resets the
canvas's starting positions accordingly. Rebuild caches after optimizer updates;
they contain representations computed with the earlier parameter values.

Future work: elastic/stem-insertion training extensions, source-token
preservation policies, the entropy-bounded reverse sampler, and MIDI integration.
The old masked-diffusion sampler is not a sampler for this new architecture.

## References and checks

- [DiffusionGemma report, sections 3–4](https://arxiv.org/pdf/2608.00146).
- [Reference architecture and self-conditioning](https://github.com/huggingface/transformers/blob/main/src/transformers/models/diffusion_gemma/modular_diffusion_gemma.py).
- `improvnet/archive/uniform_diffusion/tests/test_uniform_diffusion_model.py`: AR parity, checkpoint transfer,
  incremental encoding, padded versus unpadded equivalence, bidirectional versus
  causal visibility, immutable decoder cache use, leakage-safe prefix views,
  promptless conditioning, self-conditioning gradients, and checkpointed backward.

The configured production AR checkpoint was checked with meta tensors: all
backbone keys and shapes match. This checks compatibility without allocating the
full model on the GPU; it is not a trained-generation quality test.
