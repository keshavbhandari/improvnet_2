"""DiffusionGemma-style shared architecture adapted to the music AR checkpoint.

One set of transformer weights runs in causal encoder or bidirectional canvas
mode. Decoder passes read the clean prefix cache without updating it. Only a
finalized canvas should be encoded into the history. Corruption, losses and
sampling are deliberately outside this architecture module.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from improvnet.model.ar_model import RMSNorm, RotaryEmbedding, SwiGLU
from improvnet.model.uniform_diffusion_config import UniformDiffusionConfig


@dataclass(frozen=True)
class ContextCache:
    # Layout matches the existing AR cache: [batch, sequence, KV heads, head dim].
    key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    attention_mask: torch.Tensor
    next_positions: torch.Tensor
    control_tokens: int = 2

    def detach(self):
        """Explicit opt-out from gradients through the context encoder."""
        return ContextCache(
            tuple((k.detach(), v.detach()) for k, v in self.key_values),
            self.attention_mask, self.next_positions, self.control_tokens,
        )

    def prefix(self, music_lengths: torch.Tensor):
        """Expose only these many clean musical tokens in each batch row.

        Useful when an AR pass encoded a full clean training sequence: hide the
        target canvas AND all future tokens from the denoiser. Lengths exclude
        conditioning tokens and padding. The encoder itself must be causal.
        """
        music_lengths = torch.as_tensor(music_lengths, device=self.attention_mask.device)
        if music_lengths.shape != self.next_positions.shape or music_lengths.is_floating_point():
            raise ValueError("music_lengths must be an integer vector of shape [batch]")
        lengths = music_lengths + self.control_tokens
        if bool(((music_lengths < 0) | (lengths > self.next_positions)).any()):
            raise ValueError("Requested prefix exceeds the available clean history")
        ranks = self.attention_mask.long().cumsum(-1)
        mask = self.attention_mask & (ranks <= lengths[:, None])
        return ContextCache(self.key_values, mask, lengths, self.control_tokens)


@dataclass
class ContextOutput:
    logits: torch.Tensor | None
    cache: ContextCache


@dataclass
class DiffusionOutput:
    logits: torch.Tensor
    context: ContextOutput


class SharedAttention(nn.Module):
    """Same AR projection/norm/RoPE parameters; attention visibility is an input."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.k_proj = nn.Linear(config.embed_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.embed_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = config.dropout

    def forward(self, x, positions, attention_mask, is_causal, past_kv=None):
        batch, length, width = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim)
        q = self.rope(self.q_norm(q), positions)
        k = self.rope(self.k_norm(k), positions)
        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=1)
            v = torch.cat((past_kv[1], v), dim=1)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attention_mask, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.n_heads != self.n_kv_heads,
        )
        out = self.o_proj(attended.transpose(1, 2).reshape(batch, length, width))
        return out, k, v


class SharedBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.embed_dim)
        self.norm2 = RMSNorm(config.embed_dim)
        self.attn = SharedAttention(config)
        self.ffn = SwiGLU(config.embed_dim, config.ffn_mult, config.dropout)

    def forward(self, x, positions, attention_mask, is_causal, past_kv=None):
        attended, k, v = self.attn(self.norm1(x), positions, attention_mask, is_causal, past_kv)
        x = x + attended
        return x + self.ffn(self.norm2(x)), k, v


class SelfConditioning(nn.Module):
    """Reference-style pre-norm gated MLP, addition, then scale-free RMSNorm.

    Uses the AR backbone's SwiGLU activation in place of Gemma's gated MLP.
    Zero input contributes exactly zero through the bias-free MLP.
    """

    def __init__(self, config):
        super().__init__()
        self.pre_norm = RMSNorm(config.embed_dim)
        self.ffn = SwiGLU(config.embed_dim, config.ffn_mult, dropout=0.0)

    def forward(self, inputs, soft_embeddings):
        combined = inputs + self.ffn(self.pre_norm(soft_embeddings))
        normalized = combined.float() * torch.rsqrt(combined.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(combined.dtype)


class UniformDiffusionModel(nn.Module):
    """A single trainable backbone shared by context encoding and denoising.

    AR parameter names are preserved for strict checkpoint transfer. The AR
    input embedding and output head remain distinct, as in the source model;
    each is shared BETWEEN encoder and decoder modes. No timestep embedding,
    AdaLN, separate cross-attention weights, or concatenated draft trajectory.
    """

    def __init__(self, config: UniformDiffusionConfig | None = None):
        super().__init__()
        self.config = config or UniformDiffusionConfig()
        c = self.config
        self.token_emb = nn.Embedding(c.vocab_size, c.embed_dim)
        self.genre_emb = nn.Embedding(c.num_genres, c.embed_dim)
        self.multihot_proj = nn.Linear(c.num_instruments, c.embed_dim, bias=False)
        self.layers = nn.ModuleList(SharedBlock(c) for _ in range(c.n_layers))
        self.out_norm = RMSNorm(c.embed_dim)
        self.lm_head = nn.Linear(c.embed_dim, c.vocab_size, bias=False)
        self.self_conditioning = SelfConditioning(c)
        # Music-specific optional third prefix control. Omit it to exactly
        # reproduce AR prefill. Supply zero explicitly for rigid diffusion data.
        self.elasticity_emb = nn.Sequential(
            nn.Linear(1, c.embed_dim), nn.SiLU(), nn.Linear(c.embed_dim, c.embed_dim)
        )
        scale = 1.0 / math.sqrt(2.0 * c.n_layers)
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1:
                residual_weight = any(part in name for part in (
                    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                    "w1.weight", "w2.weight", "w3.weight",
                ))
                nn.init.normal_(parameter, std=0.02 * scale if residual_weight else 0.02)
            elif name.endswith("bias"):
                nn.init.zeros_(parameter)

    def initialize_from_ar(self, state: Mapping):
        """Copy all AR parameters; reject missing, extra or mismatched backbone weights.

        Accept a raw state_dict or the existing {'model_state_dict': ...} format,
        including DDP's module. prefix. This copies tensors; it never aliases an
        existing AR model's parameters. Diffusion-specific modules remain fresh.
        """
        state = state.get("model_state_dict", state)
        state = {name.removeprefix("module."): value for name, value in state.items()}
        own = self.state_dict()
        expected = {name for name in own if not name.startswith(("self_conditioning.", "elasticity_emb."))}
        missing, extra = expected - state.keys(), state.keys() - expected
        mismatched = [name for name in expected & state.keys() if state[name].shape != own[name].shape]
        if missing or extra or mismatched:
            raise ValueError(
                f"Incompatible AR checkpoint: missing={sorted(missing)}, "
                f"extra={sorted(extra)}, shape_mismatches={sorted(mismatched)}"
            )
        return self.load_state_dict(state, strict=False)

    @classmethod
    def from_ar_checkpoint(cls, path: str | Path, config: UniformDiffusionConfig | None = None):
        """Load a trusted local AR checkpoint on CPU; caller chooses device/dtype."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(config)
        model.initialize_from_ar(state)
        return model

    @staticmethod
    def _mask(tokens, attention_mask):
        if tokens.ndim != 2:
            raise ValueError("Token IDs must have shape [batch, length]")
        # Do not infer padding from token IDs: uniform corruption may sample
        # the PAD vocabulary entry at a VALID canvas position.
        if attention_mask is None:
            return torch.ones_like(tokens, dtype=torch.bool)
        if attention_mask.shape != tokens.shape:
            raise ValueError("attention_mask must match token IDs")
        return attention_mask.to(device=tokens.device, dtype=torch.bool)

    def _validate_cache(self, cache, batch, device):
        if not isinstance(cache, ContextCache):
            raise TypeError("Use a ContextCache produced by this architecture")
        if len(cache.key_values) != len(self.layers) or cache.attention_mask.ndim != 2:
            raise ValueError("Invalid cache layer count or mask")
        if cache.attention_mask.shape[0] != batch or cache.attention_mask.device != device:
            raise ValueError("Cache batch/device does not match inputs")
        if cache.next_positions.shape != (batch,):
            raise ValueError("Cache next_positions must have shape [batch]")
        shape = (batch, cache.attention_mask.shape[1], self.config.n_kv_heads,
                 self.config.embed_dim // self.config.n_heads)
        for k, v in cache.key_values:
            if k.shape != shape or v.shape != shape or k.device != device or v.device != device:
                raise ValueError("Invalid cached KV shape/device")

    def _run_layers(self, hidden, positions, valid_mask, cache, causal):
        old_length = 0 if cache is None else cache.attention_mask.shape[1]
        key_mask = valid_mask if cache is None else torch.cat((cache.attention_mask, valid_mask), dim=1)
        length = hidden.shape[1]
        # Right-padded prefill can also use fused causal attention: every valid
        # query precedes all padding keys, so causality already excludes them.
        # Padded-query outputs are ignored and padded KV stay masked in cache.
        # Interior padding and causal cache appends need explicit alignment (not SDPA's
        # top-left is_causal alignment for unequal Q/K lengths).
        right_padded = causal and old_length == 0 and not bool((~key_mask[:, :-1] & key_mask[:, 1:]).any())
        fused_causal = right_padded
        attention_mask = None if fused_causal or bool(key_mask.all()) else key_mask[:, None, None, :]
        if causal and not fused_causal:
            query_indices = old_length + torch.arange(length, device=hidden.device)
            key_indices = torch.arange(key_mask.shape[1], device=hidden.device)
            attention_mask = key_mask[:, None, None, :] & (key_indices[None, :] <= query_indices[:, None])[None, None]
        key_values = []
        for index, layer in enumerate(self.layers):
            past = None if cache is None else cache.key_values[index]
            args = (hidden, positions, attention_mask, fused_causal, past)
            if self.training and self.config.gradient_checkpointing and torch.is_grad_enabled():
                hidden, k, v = checkpoint(layer, *args, use_reentrant=False)
            else:
                hidden, k, v = layer(*args)
            if causal:
                key_values.append((k, v))
        return self.out_norm(hidden), tuple(key_values), key_mask

    def encode_context(
        self, input_ids, *, genre=None, multi_hot=None, attention_mask=None,
        cache: ContextCache | None = None, elasticity=None, return_logits=True,
    ) -> ContextOutput:
        """Causal prefill or append, with gradients enabled during training.

        Empty musical prompts are supported: the initial cache then contains
        only genre/instrument controls (and optional elasticity). Controls are
        immutable while appending; re-encode to change them. Returned logits
        align with input musical tokens and predict their NEXT tokens.
        """
        valid = self._mask(input_ids, attention_mask)
        batch, length = input_ids.shape
        device = input_ids.device
        hidden = self.token_emb(input_ids)
        control_tokens = 0
        if cache is None:
            genre = torch.zeros(batch, dtype=torch.long, device=device) if genre is None else genre.to(device)
            multi_hot = torch.zeros(batch, self.config.num_instruments, device=device) if multi_hot is None else multi_hot.to(device)
            if genre.shape != (batch,) or multi_hot.shape != (batch, self.config.num_instruments):
                raise ValueError("Invalid genre or instrument conditioning shape")
            controls = [self.genre_emb(genre).unsqueeze(1), self.multihot_proj(multi_hot.to(hidden.dtype)).unsqueeze(1)]
            if elasticity is not None:
                elasticity = torch.as_tensor(elasticity, device=device, dtype=hidden.dtype)
                if elasticity.shape != (batch,) or not bool(torch.isfinite(elasticity).all()) or bool(((elasticity < 0) | (elasticity > 1)).any()):
                    raise ValueError("elasticity must be a finite [batch] fraction in [0, 1]")
                controls.append(self.elasticity_emb(elasticity[:, None]).unsqueeze(1))
            control_tokens = len(controls)
            hidden = torch.cat((*controls, hidden), dim=1)
            valid = torch.cat((torch.ones(batch, control_tokens, device=device, dtype=torch.bool), valid), dim=1)
            offsets = torch.zeros(batch, device=device, dtype=torch.long)
        else:
            self._validate_cache(cache, batch, device)
            if genre is not None or multi_hot is not None or elasticity is not None:
                raise ValueError("Conditioning is already cached; omit controls when appending")
            if length == 0:
                logits = hidden.new_empty(batch, 0, self.config.vocab_size) if return_logits else None
                return ContextOutput(logits, cache)
            offsets = cache.next_positions
        positions = offsets[:, None] + (valid.long().cumsum(-1) - 1).clamp_min(0)
        hidden, key_values, key_mask = self._run_layers(hidden, positions, valid, cache, causal=True)
        new_cache = ContextCache(
            key_values, key_mask, offsets + valid.sum(-1),
            control_tokens if cache is None else cache.control_tokens,
        )
        logits = self.lm_head(hidden[:, control_tokens:]) if return_logits else None
        if logits is not None:
            logits = logits.masked_fill(~valid[:, control_tokens:, None], 0.0)
        return ContextOutput(logits, new_cache)

    def denoise(
        self, canvas_ids, *, cache: ContextCache, attention_mask=None,
        self_conditioning_logits=None, self_conditioning_mask=None,
        self_conditioning_temperature=1.0, detach_self_conditioning=True,
    ) -> torch.Tensor:
        """Predict original tokens at SAME positions using bidirectional attention.

        Each call reads only clean prefix KV plus the current canvas. Repeated
        calls reuse the same position range and NEVER append noisy KV to cache.
        Previous-pass logits are detached by default; the current soft embedding
        and self-conditioning module still receive gradients.
        """
        valid = self._mask(canvas_ids, attention_mask)
        batch, length = canvas_ids.shape
        if not 1 <= length <= self.config.canvas_size:
            raise ValueError("Canvas length must be between 1 and canvas_size")
        self._validate_cache(cache, batch, canvas_ids.device)
        if not math.isfinite(self_conditioning_temperature) or self_conditioning_temperature <= 0:
            raise ValueError("self_conditioning_temperature must be finite and positive")
        hidden = self.token_emb(canvas_ids)
        if self_conditioning_logits is None:
            soft_embeddings = torch.zeros_like(hidden)
        else:
            expected = (batch, length, self.config.vocab_size)
            if self_conditioning_logits.shape != expected:
                raise ValueError(f"self_conditioning_logits must have shape {expected}")
            previous = self_conditioning_logits.detach() if detach_self_conditioning else self_conditioning_logits
            probabilities = torch.softmax(previous.float() / self_conditioning_temperature, dim=-1)
            soft_embeddings = probabilities.to(self.token_emb.weight.dtype) @ self.token_emb.weight
        if self_conditioning_mask is not None:
            if self_conditioning_mask.shape != (batch,):
                raise ValueError("self_conditioning_mask must have shape [batch]")
            soft_embeddings = soft_embeddings * self_conditioning_mask.to(hidden.device)[:, None, None]
        soft_embeddings = soft_embeddings.masked_fill(~valid[:, :, None], 0.0)
        hidden = self.self_conditioning(hidden, soft_embeddings)
        positions = cache.next_positions[:, None] + (valid.long().cumsum(-1) - 1).clamp_min(0)
        hidden, _, _ = self._run_layers(hidden, positions, valid, cache, causal=False)
        return self.lm_head(hidden).masked_fill(~valid[:, :, None], 0.0)

    def forward(
        self, input_ids, canvas_ids, *, genre=None, multi_hot=None,
        context_attention_mask=None, canvas_attention_mask=None, elasticity=None,
        self_conditioning_logits=None, self_conditioning_mask=None,
        return_context_logits=False, prefix_lengths=None,
        bootstrap_self_conditioning=False, self_conditioning_temperature=1.0,
    ) -> DiffusionOutput:
        """Combined forward for DDP: causal context, then the shared denoiser.

        Normally input_ids contains ONLY the prefix. For joint AR/denoising
        training it may contain the full clean sequence, provided prefix_lengths
        explicitly limits the denoiser's cache to tokens BEFORE its target.
        Bootstrap self-conditioning uses a detached prediction of the SAME noisy
        canvas, never clean labels; the final pass still backpropagates through
        the original encoder cache and shared backbone.
        """
        if bootstrap_self_conditioning and self_conditioning_logits is not None:
            raise ValueError("Cannot bootstrap and supply self-conditioning logits together")
        if bootstrap_self_conditioning and self_conditioning_mask is None:
            raise ValueError("Bootstrapping requires an explicit per-example mask")
        if not math.isfinite(self_conditioning_temperature) or self_conditioning_temperature <= 0:
            raise ValueError("self_conditioning_temperature must be finite and positive")
        context = self.encode_context(
            input_ids, genre=genre, multi_hot=multi_hot,
            attention_mask=context_attention_mask, elasticity=elasticity,
            return_logits=return_context_logits,
        )
        denoising_cache = context.cache if prefix_lengths is None else context.cache.prefix(prefix_lengths)
        if bootstrap_self_conditioning and bool(self_conditioning_mask.any()):
            with torch.no_grad():
                self_conditioning_logits = self.denoise(
                    canvas_ids, cache=denoising_cache.detach(),
                    attention_mask=canvas_attention_mask,
                )
        logits = self.denoise(
            canvas_ids, cache=denoising_cache, attention_mask=canvas_attention_mask,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_mask=self_conditioning_mask,
            self_conditioning_temperature=self_conditioning_temperature,
        )
        return DiffusionOutput(logits, context)
