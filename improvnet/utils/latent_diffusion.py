"""Continuous cosine diffusion and frozen-VAE window encoding (no training side effects)."""
import hashlib
import math

import torch


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def encode_windows(vae, batch, batch_size=16):
    """Only encode real bins; an absent history bin cannot enter the VAE."""
    if batch_size < 1:
        raise ValueError("VAE batch size must be positive")
    b, n, _, _, f = batch["roll"].shape
    shape = vae.config.latent_shape
    valid = batch["valid_mask"].reshape(b * n, f)
    ids = valid.any(-1).nonzero().flatten()
    if not len(ids):
        raise ValueError("Window contains no valid bins")
    z = torch.zeros((b * n, *shape), device=valid.device, dtype=torch.float32)
    weights = torch.zeros((b * n, 1, 1, shape[-1]), device=valid.device)
    roll = batch["roll"].reshape(b * n, 2, 128, f)
    pedal = batch["pedal"].reshape(b * n, 1, f)
    vae.eval()
    for chunk in ids.split(batch_size):
        p = vae.encode(roll[chunk], pedal[chunk], valid[chunk])
        z[chunk], weights[chunk] = p.mean, p.weights
    if not bool(torch.isfinite(z).all()):
        raise FloatingPointError("Non-finite VAE encoding")
    return z.reshape(b, n, *shape), weights.reshape(b, n, 1, 1, shape[-1])


class LatentNormalizer:
    """Per-channel mean/std of clean posterior means, fitted on training targets only."""
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        if self.mean.ndim != 1 or self.std.shape != self.mean.shape or not bool(
                torch.isfinite(self.mean).all() & torch.isfinite(self.std).all() & (self.std > 0).all()):
            raise ValueError("Invalid latent normalization")

    def state_dict(self):
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    def normalize(self, z, weights):
        m, s = (v.to(z.device)[None, None, :, None, None] for v in (self.mean, self.std))
        return ((z - m) / s).masked_fill(~(weights > 0), 0)

    def denormalize(self, z, weights):
        m, s = (v.to(z.device)[None, None, :, None, None] for v in (self.mean, self.std))
        return (z * s + m).masked_fill(~(weights > 0), 0)


def coefficients(t):
    return (t * (math.pi / 2)).cos(), (t * (math.pi / 2)).sin()


def noise_target(clean, known, weights, t, epsilon):
    """History is clean, targets noisy; v = alpha*epsilon - sigma*x0."""
    if t.shape != (len(clean),) or epsilon.shape != clean.shape or not bool(((t >= 0) & (t <= 1)).all()):
        raise ValueError("Invalid timestep/noise shape or range")
    levels = t[:, None] * (~known)
    a, s = (v[..., None, None, None] for v in coefficients(levels))
    noisy = (a * clean + s * epsilon).masked_fill(~(weights > 0), 0)
    target = (a * epsilon - s * clean).masked_fill(~(weights > 0), 0)
    return noisy, target, levels


def clean_from_velocity(noisy, velocity, levels):
    a, s = (v[..., None, None, None] for v in coefficients(levels))
    return a * noisy - s * velocity


def weighted_loss(prediction, target, weights, known):
    w = weights * (~known)[..., None, None, None]
    # Average each example separately so shorter examples are not underweighted.
    denominator = w.expand_as(prediction).flatten(1).sum(1)
    if bool((denominator <= 0).any()):
        raise ValueError("Every window needs a nonempty target")
    errors = (prediction.float() - target.float()).square().masked_fill(~(w > 0), 0)
    return (errors * w).flatten(1).sum(1) / denominator


@torch.inference_mode()
def sample(model, source, known, weights, conditioning, *, steps=48, strength=1., generator=None, callback=None):
    """Deterministic DDIM updates with v prediction; exactly reclamp clean prefix each pass."""
    if steps < 1 or not math.isfinite(strength) or not 0 < strength <= 1:
        raise ValueError("Positive pass count and noise strength in (0,1] required")
    if not bool(((weights > 0) & (~known)[..., None, None, None]).flatten(1).any(1).all()):
        raise ValueError("Sampling needs at least one target bin per example")
    epsilon = torch.randn(source.shape, device=source.device, generator=generator)
    start = source.new_full((len(source),), strength)
    z, _, _ = noise_target(source, known, weights, start, epsilon)
    # At strength=1 there is no dependence on source target values, even numerically.
    if strength == 1:
        z = torch.where(known[..., None, None, None], source, epsilon).masked_fill(~(weights > 0), 0)
    grid = torch.linspace(strength, 0, steps + 1, device=source.device)
    for index in range(steps):
        levels = grid[index].expand(len(z), len(known[0])) * (~known)
        v = model(z, levels, known, weights, conditioning)
        if not bool(torch.isfinite(v).all()):
            raise FloatingPointError("Non-finite diffusion prediction")
        a, s = (q[..., None, None, None] for q in coefficients(levels))
        clean, eps = a * z - s * v, s * z + a * v
        next_levels = grid[index + 1].expand_as(levels) * (~known)
        an, sn = (q[..., None, None, None] for q in coefficients(next_levels))
        z = torch.where(known[..., None, None, None], source, an * clean + sn * eps)
        z = z.masked_fill(~(weights > 0), 0)
        if callback:
            callback(index + 1, float(grid[index + 1]), z)
    return z
