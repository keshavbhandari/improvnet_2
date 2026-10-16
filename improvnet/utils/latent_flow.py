"""Straight-path conditional flow matching and frozen-VAE window encoding (no training side effects)."""
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


def flow_target(clean, known, weights, t, epsilon):
    """t=0 is noise, t=1 is data; clean history stays at t=1."""
    if t.shape != (len(clean),) or epsilon.shape != clean.shape or not bool(torch.isfinite(t).all() & ((t >= 0) & (t <= 1)).all()):
        raise ValueError("Invalid flow time/noise shape or range")
    levels = torch.where(known, torch.ones_like(t[:, None]), t[:, None])
    time = levels[..., None, None, None]
    state = ((1 - time) * epsilon + time * clean).masked_fill(~(weights > 0), 0)
    velocity = (clean - epsilon).masked_fill(known[..., None, None, None] | ~(weights > 0), 0)
    return state, velocity, levels


def clean_from_velocity(state, velocity, levels):
    """Straight-path endpoint estimate, used only as a training diagnostic."""
    return state + (1 - levels[..., None, None, None]) * velocity


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
    """Euler integration from t=1-strength to t=1; reclamp history every step.

    strength=1 starts from pure noise; lower strengths interpolate the source
    with noise before following the learned vector field toward the data end.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1 or not math.isfinite(strength) or not 0 < strength <= 1:
        raise ValueError("Positive step count and strength in (0,1] required")
    if not bool(((weights > 0) & (~known)[..., None, None, None]).flatten(1).any(1).all()):
        raise ValueError("Sampling needs at least one target bin per example")
    epsilon = torch.randn(source.shape, device=source.device, generator=generator)
    start = source.new_full((len(source),), 1 - strength)
    if strength == 1:
        z = torch.where(known[..., None, None, None], source, epsilon).masked_fill(~(weights > 0), 0)
    else:
        z, _, _ = flow_target(source, known, weights, start, epsilon)
    grid = torch.linspace(1 - strength, 1, steps + 1, device=source.device)
    for index in range(steps):
        levels = torch.where(known, torch.ones_like(known, dtype=z.dtype), grid[index].expand_as(known))
        velocity = model(z, levels, known, weights, conditioning)
        if not bool(torch.isfinite(velocity).all()):
            raise FloatingPointError("Non-finite flow velocity")
        z = torch.where(known[..., None, None, None], source, z + (grid[index + 1] - grid[index]) * velocity)
        z = z.masked_fill(~(weights > 0), 0)
        if not bool(torch.isfinite(z).all()):
            raise FloatingPointError("Non-finite flow integration state")
        if callback:
            callback(index + 1, float(grid[index + 1]), z)
    return z
