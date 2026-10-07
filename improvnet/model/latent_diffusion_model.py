"""Multi-bin continuous latent diffusion with clean-prefix and soft musical conditions.

Local VAE latents are patchified spatially; bin/pitch/fine-time positions remain
distinct. Bidirectional self attention couples the complete generated window.
There is no causal/AR tower and no pitch mask applied to output predictions.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class LatentDiffusionConfig:
    latent_channels: int = 8
    latent_pitch: int = 32
    latent_time: int = 25
    frames_per_bin: int = 100
    history_bins: int = 4
    target_bins: int = 8
    patch_pitch: int = 4
    patch_time: int = 5
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    ff_multiplier: int = 4
    num_genres: int = 2
    max_tracks: int = 3
    max_events: int = 256
    rhythm_frames_per_token: int = 10
    gradient_checkpointing: bool = True

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "gradient_checkpointing":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "history_bins" else 1):
                raise ValueError(f"Invalid {name}")
        if self.d_model % self.n_heads or self.latent_pitch % self.patch_pitch or self.max_tracks > 3:
            raise ValueError("Invalid head/patch dimensions or track limit")
        if self.frames_per_bin % self.rhythm_frames_per_token:
            raise ValueError("Rhythm token width must divide frames_per_bin")

    @property
    def bins(self):
        return self.history_bins + self.target_bins

    @property
    def spatial_grid(self):
        return self.latent_pitch // self.patch_pitch, math.ceil(self.latent_time / self.patch_time)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.width = heads, dim // heads
        self.q, self.k, self.v, self.out = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, query, context, valid):
        def heads(x):
            return x.reshape(len(x), -1, self.heads, self.width).transpose(1, 2)
        attended = F.scaled_dot_product_attention(heads(self.q(query)), heads(self.k(context)),
            heads(self.v(context)), attn_mask=valid[:, None, None, :], dropout_p=0.)
        return self.out(attended.transpose(1, 2).reshape_as(query))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(c.d_model) for _ in range(3))
        self.self_attention = Attention(c.d_model, c.n_heads)
        self.cross_attention = Attention(c.d_model, c.n_heads)
        self.ff = nn.Sequential(nn.Linear(c.d_model, c.ff_multiplier * c.d_model), nn.GELU(),
                                nn.Linear(c.ff_multiplier * c.d_model, c.d_model))

    def forward(self, x, context, valid, context_valid):
        y = self.norm1(x)
        x = x + self.self_attention(y, y, valid)
        x = x + self.cross_attention(self.norm2(x), context, context_valid)
        return (x + self.ff(self.norm3(x))).masked_fill(~valid[..., None], 0)


class LatentDiffusionModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or LatentDiffusionConfig()
        gh, gw = c.spatial_grid
        self.patch_dim = c.latent_channels * c.patch_pitch * c.patch_time
        self.input = nn.Linear(self.patch_dim, c.d_model)
        self.bin_position = nn.Embedding(c.bins, c.d_model)
        self.pitch_position = nn.Embedding(gh, c.d_model)
        self.time_position = nn.Embedding(gw, c.d_model)
        self.noise_embedding = nn.Sequential(nn.Linear(64, c.d_model), nn.SiLU(), nn.Linear(c.d_model, c.d_model))
        self.known_embedding = nn.Embedding(2, c.d_model)
        self.genre_embedding = nn.Embedding(c.num_genres, c.d_model)
        self.program_embedding = nn.Embedding(128, c.d_model)
        self.drum_embedding = nn.Embedding(2, c.d_model)
        self.mode_embedding = nn.Embedding(3, c.d_model)
        self.pitch_condition = nn.Sequential(nn.Linear(128, c.d_model), nn.SiLU(), nn.Linear(c.d_model, c.d_model))
        self.event_pitch = nn.Embedding(128, c.d_model)
        self.track_embedding = nn.Embedding(c.max_tracks, c.d_model)
        self.event_numbers = nn.Sequential(nn.Linear(3, c.d_model), nn.SiLU(), nn.Linear(c.d_model, c.d_model))
        self.rhythm_projection = nn.Linear(3 * c.rhythm_frames_per_token, c.d_model)
        self.rhythm_position = nn.Embedding(c.frames_per_bin // c.rhythm_frames_per_token, c.d_model)
        self.null_condition = nn.Parameter(torch.randn(1, 1, c.d_model) * .02)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output_norm = nn.LayerNorm(c.d_model)
        self.output = nn.Linear(c.d_model, self.patch_dim)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def patchify(self, z):
        c = self.config
        b, n, channels, height, width = z.shape
        gh, gw = c.spatial_grid
        z = F.pad(z, (0, gw * c.patch_time - width))
        return z.reshape(b, n, channels, gh, c.patch_pitch, gw, c.patch_time).permute(0, 1, 3, 5, 2, 4, 6).reshape(b, n, gh, gw, self.patch_dim)

    def unpatchify(self, tokens):
        c = self.config
        gh, gw = c.spatial_grid
        return tokens.reshape(len(tokens), c.bins, gh, gw, c.latent_channels, c.patch_pitch, c.patch_time).permute(
            0, 1, 4, 2, 5, 3, 6).reshape(len(tokens), c.bins, c.latent_channels, c.latent_pitch, gw * c.patch_time)[..., :c.latent_time]

    def conditions(self, condition, bin_valid):
        c = self.config
        b = len(bin_valid)
        positions = self.bin_position.weight[None]
        pitch = self.pitch_condition(condition["pitch"].masked_fill(~condition["pitch_present"][..., None], 0)) + positions
        pitch_mask = condition["pitch_present"] & bin_valid
        event_mask = condition["event_mask"] & (condition["mode"][:, None] == 1)
        # Sanitize padded fields BEFORE embedding lookups; even invalid padded IDs must be harmless.
        e = condition["events"].masked_fill(~event_mask[..., None], 0)
        p = condition["event_positions"].masked_fill(~event_mask[..., None], 0) / (c.bins * c.frames_per_bin)
        numerical = torch.cat((p.clamp(-2, 2), e[..., 4:5].float() / 127), dim=-1)
        events = self.track_embedding(e[..., 0]) + self.program_embedding(e[..., 1]) + self.drum_embedding(e[..., 2]) + self.event_pitch(e[..., 3]) + self.event_numbers(numerical)
        groups = c.frames_per_bin // c.rhythm_frames_per_token
        group_valid = condition["valid_mask"].reshape(b, c.bins, groups, c.rhythm_frames_per_token).any(-1)
        rhythm_mask = (condition["track_mask"][:, :, None, None] & group_valid[:, None]
                       & (condition["mode"][:, None, None, None] == 2))
        fine_mask = (rhythm_mask.repeat_interleave(c.rhythm_frames_per_token, dim=-1)
                     & condition["valid_mask"][:, None])
        r = condition["rhythm"].masked_fill(~fine_mask[..., None], 0)
        rhythm = self.rhythm_projection(r.reshape(b, c.max_tracks, c.bins, groups, -1))
        rhythm = rhythm + positions[:, None, :, None] + self.rhythm_position.weight[None, None, None]
        rhythm = rhythm + self.track_embedding.weight[None, :, None, None]
        meta_mask = condition["track_mask"] & (condition["mode"][:, None] == 2)
        programs = condition["accomp_programs"].masked_fill(~meta_mask, 0)
        drums = condition["accomp_drums"].masked_fill(~meta_mask, 0)
        rhythm = rhythm + (self.program_embedding(programs) + self.drum_embedding(drums))[:, :, None, None]
        context = torch.cat((self.null_condition.expand(b, -1, -1), pitch, events, rhythm.flatten(1, 3)), dim=1)
        valid = torch.cat((torch.ones(b, 1, dtype=torch.bool, device=bin_valid.device), pitch_mask, event_mask, rhythm_mask.flatten(1)), dim=1)
        return context.masked_fill(~valid[..., None], 0), valid

    def forward(self, noisy, noise_levels, known, weights, condition):
        c = self.config
        expected = (c.bins, c.latent_channels, c.latent_pitch, c.latent_time)
        if noisy.ndim != 5 or tuple(noisy.shape[1:]) != expected:
            raise ValueError(f"Expected latent windows [batch,{expected}]")
        if noise_levels.shape != noisy.shape[:2] or known.shape != noise_levels.shape or known.dtype != torch.bool:
            raise ValueError("Invalid bin timestep/history shapes")
        if weights.shape != (len(noisy), c.bins, 1, 1, c.latent_time):
            raise ValueError("Invalid latent padding weights")
        if condition["events"].shape[1] > c.max_events:
            raise ValueError("Accompaniment event budget exceeded")
        if not bool(torch.isfinite(noise_levels).all() & (noise_levels >= 0).all() & (noise_levels <= 1).all()) or bool((noise_levels[known] != 0).any()):
            raise ValueError("Clean history must have timestep zero; target timesteps must lie in [0,1]")
        gh, gw = c.spatial_grid
        w = F.pad(weights[..., 0, 0, :], (0, gw * c.patch_time - c.latent_time))
        valid = w.reshape(len(noisy), c.bins, gw, c.patch_time).any(-1)[:, :, None].expand(-1, -1, gh, -1)
        bin_valid = weights.flatten(2).any(-1)
        if not bool(bin_valid.any(-1).all()):
            raise ValueError("Every canvas needs valid tokens")
        x = self.input(self.patchify(noisy.masked_fill(~(weights > 0), 0)))
        frequency = torch.exp(torch.arange(32, device=x.device).float() * (-math.log(10000) / 31))
        phase = noise_levels[..., None].float() * 1000 * frequency
        time = self.noise_embedding(torch.cat((phase.sin(), phase.cos()), -1))
        global_condition = (self.genre_embedding(condition["genre"]) + self.program_embedding(condition["program"])
                            + self.drum_embedding(condition["is_drum"].long()) + self.mode_embedding(condition["mode"]))
        x = x + (self.bin_position.weight[None] + time + self.known_embedding(known.long()) + global_condition[:, None])[:, :, None, None]
        x = x + self.pitch_position.weight[None, None, :, None] + self.time_position.weight[None, None, None]
        x = x.flatten(1, 3).masked_fill(~valid.flatten(1)[..., None], 0)
        context, context_valid = self.conditions(condition, bin_valid)
        for block in self.blocks:
            if c.gradient_checkpointing and self.training:
                x = checkpoint(block, x, context, valid.flatten(1), context_valid, use_reentrant=False)
            else:
                x = block(x, context, valid.flatten(1), context_valid)
        output = self.unpatchify(self.output(self.output_norm(x)))
        return output.masked_fill(~(weights > 0), 0)
