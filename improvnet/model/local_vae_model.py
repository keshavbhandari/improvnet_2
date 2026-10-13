"""Unconditional local VAE for complete fine-frame patches, not coarse summaries.

Velocity/onset use a 2-D encoder; pedal uses a separate 1-D encoder. Fusion is in
hidden features only. A global vector or a spatial grid forms the bottleneck.
No encoder-to-decoder skip connections can bypass latents.
The model config class lives here; local_vae_config.py contains only values.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from improvnet.utils.local_vae_corruption import jitter_latent


@dataclass(frozen=True)
class LocalVAEConfig:
    frames_per_bin: int = 100
    base_channels: int = 96
    channel_multipliers: tuple = (1, 2, 4)
    latent_channels: int = 8
    res_blocks: int = 2
    latent_layout: str = "spatial"
    vector_projection_channels: int = 4

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in
               (self.frames_per_bin, self.base_channels, self.latent_channels, self.res_blocks,
                self.vector_projection_channels)):
            raise ValueError("VAE dimensions must be positive integers")
        if self.latent_layout not in ("spatial", "vector"):
            raise ValueError("Latent layout must be spatial or vector")
        if len(self.channel_multipliers) < 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in self.channel_multipliers):
            raise ValueError("Use at least two positive channel multipliers")
        if 128 % self.factor:
            raise ValueError("Pitch dimension must divide by the downsampling factor")

    @property
    def factor(self):
        return 2 ** (len(self.channel_multipliers) - 1)

    @property
    def latent_shape(self):
        if self.latent_layout == "vector":
            # Singleton spatial axes keep the encoder/decoder/diffusion API shared.
            return (self.latent_channels, 1, 1)
        return (self.latent_channels, 128 // self.factor, math.ceil(self.frames_per_bin / self.factor))


def norm(channels):
    return nn.GroupNorm(math.gcd(channels, 8), channels)


class Residual(nn.Module):
    def __init__(self, channels, dim=2):
        super().__init__()
        conv = nn.Conv2d if dim == 2 else nn.Conv1d
        self.body = nn.Sequential(norm(channels), nn.SiLU(), conv(channels, channels, 3, padding=1),
                                  norm(channels), nn.SiLU(), conv(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.body(x)


@dataclass
class Posterior:
    mean: torch.Tensor
    logvar: torch.Tensor
    weights: torch.Tensor  # [B,1,1,latent_time], fraction of valid fine frames.

    def sample(self, generator=None):
        noise = torch.randn(self.mean.shape, device=self.mean.device, dtype=self.mean.dtype, generator=generator)
        return (self.mean + torch.exp(.5 * self.logvar) * noise) * (self.weights > 0)

    def mode(self):
        return self.mean

    def kl(self):
        terms = .5 * (self.mean.float().square() + self.logvar.float().exp() - 1 - self.logvar.float())
        return (terms * self.weights).flatten(1).sum(1) / (self.weights.flatten(1).sum(1) * self.mean.shape[1] * self.mean.shape[2]).clamp_min(1)


@dataclass
class Reconstruction:
    activity_logits: torch.Tensor  # [B,128,F]
    onset_logits: torch.Tensor
    velocity: torch.Tensor         # Normalized attack velocity, before activity gating.
    pedal: torch.Tensor            # [B,1,F], normalized CC64.

    def tensors(self, valid_mask, pedal_enabled, *, activity_threshold=.5, onset_threshold=.5):
        """Quantized arrays for assembling complete pieces before MIDI decoding.

        Never insert an onset at a patch start: a note may carry from the prior
        bin. Final MIDI assembly must decode all bins continuously.
        """
        active = (self.activity_logits.sigmoid() >= activity_threshold) & valid_mask[:, None, :]
        velocity = (self.velocity.clamp(0, 1) * 127).round().clamp(1, 127).to(torch.uint8) * active
        onset = ((self.onset_logits.sigmoid() >= onset_threshold) & active).to(torch.uint8)
        pedal = (self.pedal[:, 0].clamp(0, 1) * 127).round().to(torch.uint8)
        pedal = pedal * valid_mask * pedal_enabled[:, None]
        return dict(velocity=velocity, onset=onset, pedal=pedal)


@dataclass
class VAEOutput:
    reconstruction: Reconstruction
    posterior: Posterior
    latent: torch.Tensor
    denoising: "VAEOutput | None" = None
    jittered: "VAEOutput | None" = None


class LocalVAE(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or LocalVAEConfig()
        c = self.config
        widths = [c.base_channels * m for m in c.channel_multipliers]
        roll, pedal = [nn.Conv2d(3, widths[0], 3, padding=1)], [nn.Conv1d(2, widths[0], 3, padding=1)]
        for index, width in enumerate(widths):
            roll.extend(Residual(width) for _ in range(c.res_blocks))
            pedal.append(Residual(width, dim=1))
            if index + 1 < len(widths):
                roll.append(nn.Conv2d(width, widths[index + 1], 3, stride=2, padding=1))
                pedal.append(nn.Conv1d(width, widths[index + 1], 3, stride=2, padding=1))
        self.roll_encoder, self.pedal_encoder = nn.Sequential(*roll), nn.Sequential(*pedal)
        if c.latent_layout == "vector":
            # Learn a global code from the complete ordered pitch/time grid.
            # Projection before flattening avoids a huge dense feature matrix.
            projected = c.vector_projection_channels
            self.grid_shape = (projected, 128 // c.factor, math.ceil(c.frames_per_bin / c.factor))
            grid_values = math.prod(self.grid_shape)
            self.posterior_head = nn.Conv2d(2 * widths[-1], projected, 1)
            self.vector_posterior = nn.Linear(grid_values, 2 * c.latent_channels)
            self.vector_decoder = nn.Linear(c.latent_channels, grid_values)
            decoder_channels = projected
        else:
            self.posterior_head = nn.Conv2d(2 * widths[-1], 2 * c.latent_channels, 1)
            decoder_channels = c.latent_channels
        decoder = [nn.Conv2d(decoder_channels, widths[-1], 3, padding=1)]
        pedal_decoder = [nn.Conv1d(decoder_channels, widths[-1], 3, padding=1)]
        for index in reversed(range(len(widths))):
            width = widths[index]
            decoder.extend(Residual(width) for _ in range(c.res_blocks))
            pedal_decoder.append(Residual(width, dim=1))
            if index:
                decoder.extend((nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(width, widths[index - 1], 3, padding=1)))
                pedal_decoder.extend((nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv1d(width, widths[index - 1], 3, padding=1)))
        self.roll_decoder, self.pedal_decoder = nn.Sequential(*decoder), nn.Sequential(*pedal_decoder)
        self.roll_head = nn.Conv2d(widths[0], 3, 1)  # Activity, onset, nonzero velocity.
        self.pedal_head = nn.Conv1d(widths[0], 1, 1)
        with torch.no_grad():
            self.roll_head.bias[0] = -3
            self.roll_head.bias[1] = -4

    def encode(self, roll, pedal, valid_mask):
        c = self.config
        if roll.ndim != 4 or roll.shape[1:] != (2, 128, c.frames_per_bin):
            raise ValueError("roll must be [batch,2,128,frames_per_bin]")
        if pedal.shape != (len(roll), 1, c.frames_per_bin) or valid_mask.shape != (len(roll), c.frames_per_bin):
            raise ValueError("Invalid pedal/mask shapes")
        if valid_mask.dtype != torch.bool or not bool(valid_mask.any(1).all()):
            raise ValueError("Every patch needs at least one valid frame")
        if bool(((~valid_mask[:, :-1]) & valid_mask[:, 1:]).any()):
            raise ValueError("Patch padding must be trailing")
        missing = (-c.frames_per_bin) % c.factor
        mask = F.pad(valid_mask.float(), (0, missing))[:, None]
        roll = F.pad(roll.masked_fill(~valid_mask[:, None, None], 0), (0, missing))
        pedal = F.pad(pedal.masked_fill(~valid_mask[:, None], 0), (0, missing))
        h = self.roll_encoder(torch.cat((roll, mask[:, :, None].expand(-1, -1, 128, -1)), dim=1))
        p = self.pedal_encoder(torch.cat((pedal, mask), dim=1))
        features = self.posterior_head(torch.cat((h, p[:, :, None].expand_as(h)), dim=1))
        if c.latent_layout == "vector":
            # Mask fully padded feature columns before the global projection.
            grid_valid = F.avg_pool1d(mask, c.factor, c.factor) > 0
            features = features.masked_fill(~grid_valid[:, :, None], 0)
            parameters = self.vector_posterior(features.flatten(1))[:, :, None, None]
            weights = valid_mask.float().mean(-1)[:, None, None, None]
        else:
            parameters = features
            weights = F.avg_pool1d(mask, c.factor, c.factor)[:, :, None]
        mean, logvar = parameters.float().chunk(2, dim=1)
        return Posterior(mean * (weights > 0), logvar.clamp(-20, 10) * (weights > 0), weights)

    def decode(self, latent):
        if latent.ndim != 4 or tuple(latent.shape[1:]) != self.config.latent_shape:
            raise ValueError("Latent shape does not match local VAE config")
        if self.config.latent_layout == "vector":
            latent = self.vector_decoder(latent.flatten(1)).reshape(len(latent), *self.grid_shape)
        f = self.config.frames_per_bin
        r = self.roll_head(self.roll_decoder(latent))[:, :, :, :f]
        p = self.pedal_head(self.pedal_decoder(latent.mean(dim=2)))[:, :, :f]
        return Reconstruction(r[:, 0], r[:, 1], r[:, 2].sigmoid(), p.sigmoid())

    def forward(self, roll, pedal, valid_mask, *, sample=True, generator=None,
                corrupted_roll=None, latent_jitter_std=0., fixed_jitter=False):
        # A single forward/graph includes all enabled branches, including under
        # DDP. No robustness-specific parameters/buffers; legacy inference uses
        # its saved architecture configuration and the same state keys.
        count = len(roll)
        if corrupted_roll is None:
            posterior = self.encode(roll, pedal, valid_mask)
            noisy_posterior = None
        else:
            if corrupted_roll.shape != roll.shape:
                raise ValueError("Corrupted roll must match the clean roll shape")
            joint = self.encode(torch.cat((roll, corrupted_roll)), torch.cat((pedal, pedal)),
                                torch.cat((valid_mask, valid_mask)))
            posterior, noisy_posterior = [Posterior(*(x[part] for x in
                (joint.mean, joint.logvar, joint.weights))) for part in (slice(0, count), slice(count, None))]
        latent = posterior.sample(generator) if sample else posterior.mode()
        latents = [latent]
        if noisy_posterior is not None:
            latents.append(noisy_posterior.sample(generator) if sample else noisy_posterior.mode())
        if latent_jitter_std:
            latents.append(jitter_latent(latent, posterior, latent_jitter_std,
                                         generator=generator, fixed=fixed_jitter))
        joint_reconstruction = self.decode(torch.cat(latents))
        reconstructions = [Reconstruction(*(x[i * count:(i + 1) * count] for x in
            (joint_reconstruction.activity_logits, joint_reconstruction.onset_logits,
             joint_reconstruction.velocity, joint_reconstruction.pedal))) for i in range(len(latents))]
        output = VAEOutput(reconstructions[0], posterior, latent)
        if noisy_posterior is not None:
            output.denoising = VAEOutput(reconstructions[1], noisy_posterior, latents[1])
        if latent_jitter_std:
            output.jittered = VAEOutput(reconstructions[-1], posterior, latents[-1])
        return output


def balanced_bce(logits, targets, mask, *, positive_fraction=.5):
    if not 0 < positive_fraction < 1:
        raise ValueError("Positive class fraction must be in (0,1)")
    # Truly exclude padding, even if a caller's padded tensors contain NaNs.
    logits = logits.float().masked_fill(~mask, 0)
    targets = targets.float().masked_fill(~mask, 0)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pos, neg = mask & targets.bool(), mask & ~targets.bool()
    def average(selection):
        return (loss * selection).flatten(1).sum(1) / selection.flatten(1).sum(1).clamp_min(1)
    # Average over classes that are actually present, including silent patches.
    classes = (positive_fraction * pos.flatten(1).any(1).float() +
               (1 - positive_fraction) * neg.flatten(1).any(1).float())
    return (positive_fraction * average(pos) + (1 - positive_fraction) * average(neg)) / classes.clamp_min(1e-8)


def reconstruction_losses(output, batch, *, activity_positive_fraction=.5, onset_positive_fraction=.5):
    """Per-example float32 losses; no genre/metadata conditioning or pad targets."""
    r, roll = output.reconstruction, batch["roll"].float()
    mask = batch["valid_mask"][:, None].expand_as(roll[:, 0])
    active = (roll[:, 0] > 0) & mask
    def masked_mean(values, selection):
        return values.float().masked_fill(~selection, 0).flatten(1).sum(1) / selection.flatten(1).sum(1).clamp_min(1)
    pedal_mask = batch["valid_mask"][:, None] & batch["pedal_enabled"][:, None, None]
    return dict(activity=balanced_bce(r.activity_logits, active, mask, positive_fraction=activity_positive_fraction),
                onset=balanced_bce(r.onset_logits, roll[:, 1], mask, positive_fraction=onset_positive_fraction),
                velocity=masked_mean((r.velocity.float() - roll[:, 0]).abs(), active),
                pedal=masked_mean((r.pedal.float() - batch["pedal"]).abs(), pedal_mask),
                consistency=masked_mean(r.onset_logits.float().sigmoid() * (1 - r.activity_logits.float().sigmoid()), mask),
                kl=output.posterior.kl())
