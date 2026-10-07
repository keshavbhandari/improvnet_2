"""Uniform categorical forward diffusion and unweighted reconstruction loss.

q(x_t | x_0) = (1-t) delta(x_0) + t Uniform(vocabulary), t ~ Uniform[0,1].
Noise can sample ANY vocabulary entry, including the original token and PAD.
Consequently validity must come from the clean example's explicit padding mask.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class CorruptedCanvas:
    tokens: torch.Tensor
    timesteps: torch.Tensor
    replaced: torch.Tensor
    changed: torch.Tensor


def uniform_corrupt(clean, valid_mask, vocab_size, *, timesteps=None, generator=None):
    if clean.ndim != 2 or clean.dtype != torch.long or valid_mask.shape != clean.shape:
        raise ValueError("Expected long [batch, canvas] IDs and matching validity mask")
    if vocab_size < 1 or bool(((clean < 0) | (clean >= vocab_size)).any()):
        raise ValueError("Clean token IDs must belong to the vocabulary")
    valid_mask = valid_mask.to(device=clean.device, dtype=torch.bool)
    batch = clean.shape[0]
    if timesteps is None:
        timesteps = torch.rand(batch, device=clean.device, generator=generator)
    else:
        timesteps = torch.as_tensor(timesteps, device=clean.device, dtype=torch.float32)
    if timesteps.shape != (batch,) or not bool(torch.isfinite(timesteps).all()) or bool(((timesteps < 0) | (timesteps > 1)).any()):
        raise ValueError("timesteps must be finite [batch] values in [0, 1]")
    replaced = (torch.rand(clean.shape, device=clean.device, generator=generator) < timesteps[:, None]) & valid_mask
    noise = torch.randint(vocab_size, clean.shape, device=clean.device, generator=generator)
    tokens = torch.where(replaced, noise, clean)
    return CorruptedCanvas(tokens, timesteps, replaced, (tokens != clean) & valid_mask)


def reconstruction_losses(logits, targets, valid_mask, chunk_size=256):
    """Mean CE per example, over ALL valid positions, including unchanged ones.

    Chunk and checkpoint the fp32 CE to avoid retaining full-sequence fp32
    softmax activations. An empty row contributes a differentiable zero; the
    data sampler never creates empty denoising targets.
    """
    if logits.shape[:-1] != targets.shape or targets.shape != valid_mask.shape or chunk_size < 1:
        raise ValueError("Invalid loss shapes or chunk size")
    sums = logits.new_zeros(targets.shape[0], dtype=torch.float32)

    def chunk_loss(scores, labels, mask):
        safe_labels = labels.masked_fill(~mask, 0)
        ce = F.cross_entropy(scores.float().transpose(1, 2), safe_labels, reduction="none")
        return (ce * mask).sum(-1)

    valid_mask = valid_mask.bool()
    for start in range(0, targets.shape[1], chunk_size):
        args = (logits[:, start:start + chunk_size], targets[:, start:start + chunk_size],
                valid_mask[:, start:start + chunk_size])
        if torch.is_grad_enabled() and logits.requires_grad:
            sums = sums + checkpoint(chunk_loss, *args, use_reentrant=False)
        else:
            sums = sums + chunk_loss(*args)
    return sums / valid_mask.sum(-1).clamp_min(1)
