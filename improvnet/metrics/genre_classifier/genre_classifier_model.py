"""Bidirectional genre encoder with the uniform-diffusion backbone architecture.

Independent parameters and local primitives: importing this module does not
import/change AR or diffusion training configs. No conditioning or LM head.
"""

from dataclasses import dataclass, field
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import genre_classifier_config as defaults


def _vocab_size():
    from improvnet.tokenizer.absolute import AbsTokenizer
    return AbsTokenizer().vocab_size


@dataclass(frozen=True)
class GenreClassifierConfig:
    vocab_size: int = field(default_factory=_vocab_size)
    num_genres: int = 12  # Replaced with the audited label count by the trainer.
    pad_id: int = 2
    embed_dim: int = defaults.EMBED_DIM
    n_heads: int = defaults.N_HEADS
    n_kv_heads: int = defaults.N_KV_HEADS
    n_layers: int = defaults.N_LAYERS
    ffn_mult: float = defaults.FFN_MULT
    dropout: float = defaults.DROPOUT
    gradient_checkpointing: bool = defaults.GRADIENT_CHECKPOINTING
    max_length: int = defaults.MAX_LENGTH

    @classmethod
    def from_settings(cls, settings, *, vocab_size, num_genres, pad_id):
        """Build checkpoint-compatible model settings from values-only config."""
        return cls(
            vocab_size=vocab_size, num_genres=num_genres, pad_id=pad_id,
            embed_dim=settings.EMBED_DIM, n_heads=settings.N_HEADS,
            n_kv_heads=settings.N_KV_HEADS, n_layers=settings.N_LAYERS,
            ffn_mult=settings.FFN_MULT, dropout=settings.DROPOUT,
            gradient_checkpointing=settings.GRADIENT_CHECKPOINTING,
            max_length=settings.MAX_LENGTH,
        )

    def __post_init__(self):
        for name in ("vocab_size", "num_genres", "embed_dim", "n_heads", "n_kv_heads", "n_layers", "max_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_genres < 2 or self.embed_dim % self.n_heads or (self.embed_dim // self.n_heads) % 2:
            raise ValueError("Need at least two genres and an even, integral attention head dimension")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        if not math.isfinite(self.ffn_mult) or self.ffn_mult <= 0:
            raise ValueError("ffn_mult must be finite and positive")
        if not 0 <= self.pad_id < self.vocab_size or not 0 <= self.dropout < 1:
            raise ValueError("Invalid padding ID or dropout")
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing must be boolean")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        values = x.float()
        return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps) * self.scale.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=500000.):
        super().__init__()
        self.register_buffer("inv_freq", base ** (-torch.arange(0, head_dim, 2).float() / head_dim), persistent=False)

    def forward(self, x, positions):
        # Same split-half RoPE convention and base as the uniform backbone.
        phases = torch.einsum("bt,f->btf", positions.float(), self.inv_freq)
        phases = torch.cat((phases, phases), dim=-1).unsqueeze(2)
        first, second = x.chunk(2, dim=-1)
        return x * phases.cos().to(x.dtype) + torch.cat((-second, first), dim=-1) * phases.sin().to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, width, mult, dropout):
        super().__init__()
        hidden = int(math.ceil(width * mult / 64) * 64)
        self.w1 = nn.Linear(width, hidden, bias=False)
        self.w2 = nn.Linear(width, hidden, bias=False)
        self.w3 = nn.Linear(hidden, width, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class EncoderAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads, self.n_kv_heads = config.n_heads, config.n_kv_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.k_proj = nn.Linear(config.embed_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.embed_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = config.dropout

    def forward(self, x, positions, mask):
        batch, length, width = x.shape
        q = self.rope(self.q_norm(self.q_proj(x).view(batch, length, self.n_heads, self.head_dim)), positions)
        k = self.rope(self.k_norm(self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim)), positions)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask[:, None, None, :], is_causal=False,
            dropout_p=self.dropout if self.training else 0.,
            enable_gqa=self.n_heads != self.n_kv_heads,
        )
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, width))


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(config.embed_dim), RMSNorm(config.embed_dim)
        self.attn = EncoderAttention(config)
        self.ffn = SwiGLU(config.embed_dim, config.ffn_mult, config.dropout)

    def forward(self, x, mask, positions):
        x = x + self.attn(self.norm1(x), positions, mask)
        return (x + self.ffn(self.norm2(x))).masked_fill(~mask[..., None], 0.)


class GenreClassifier(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or GenreClassifierConfig()
        c = self.config
        self.token_emb = nn.Embedding(c.vocab_size, c.embed_dim, padding_idx=c.pad_id)
        self.layers = nn.ModuleList(EncoderBlock(c) for _ in range(c.n_layers))
        self.out_norm = RMSNorm(c.embed_dim)
        self.head = nn.Linear(c.embed_dim, c.num_genres)
        scale = 1. / math.sqrt(2. * c.n_layers)
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1:
                residual = any(part in name for part in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "w1.weight", "w2.weight", "w3.weight"))
                nn.init.normal_(parameter, std=.02 * scale if residual else .02)
            elif name.endswith("bias"):
                nn.init.zeros_(parameter)
        with torch.no_grad():
            self.token_emb.weight[c.pad_id].zero_()

    def forward(self, ids, attention_mask=None, *, return_embedding=False):
        if ids.ndim != 2 or ids.shape[0] < 1 or not 1 <= ids.shape[1] <= self.config.max_length:
            raise ValueError("Need nonempty [batch, length] IDs within the configured limit")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Token IDs must be integers")
        mask = ids.ne(self.config.pad_id) if attention_mask is None else attention_mask
        if mask.shape != ids.shape or mask.dtype != torch.bool or mask.device != ids.device:
            raise ValueError("attention_mask must be boolean and match IDs in shape/device")
        if not bool(mask.any(-1).all()):
            raise ValueError("Cannot classify an empty/all-padded example")
        if bool(((ids[mask] < 0) | (ids[mask] >= self.config.vocab_size)).any()):
            raise ValueError("Valid IDs are outside the classifier vocabulary")
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        x = self.token_emb(ids.masked_fill(~mask, self.config.pad_id)).masked_fill(~mask[..., None], 0.)
        for layer in self.layers:
            if self.training and self.config.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(layer, x, mask, positions, use_reentrant=False)
            else:
                x = layer(x, mask, positions)
        x = self.out_norm(x).masked_fill(~mask[..., None], 0.)
        pooled = x.float().sum(1) / mask.sum(-1, keepdim=True)
        logits = self.head(pooled.to(x.dtype))
        return (logits, pooled) if return_embedding else logits
