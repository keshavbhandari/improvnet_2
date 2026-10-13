"""Bidirectional AR-sized mask predictor, with four conditioning tokens."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from improvnet.model import masked_diffusion_config as defaults
from improvnet.model.ar_model import RMSNorm, SwiGLU, RotaryEmbedding

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


@dataclass(frozen=True)
class MaskedDiffusionConfig:
    vocab_size: int = defaults.VOCAB_SIZE
    embed_dim: int = defaults.EMBED_DIM
    n_heads: int = defaults.N_HEADS
    n_kv_heads: int = defaults.N_KV_HEADS
    n_layers: int = defaults.N_LAYERS
    num_genres: int = defaults.NUM_GENRES
    num_instruments: int = 41
    rhythm_dim: int = 14
    seq_len: int = defaults.SEQ_LEN
    pad_id: int = defaults.PAD_ID
    gradient_checkpointing: bool = True

    def __post_init__(self):
        for name in ("vocab_size", "embed_dim", "n_heads", "n_kv_heads", "n_layers",
                     "num_genres", "num_instruments", "rhythm_dim", "seq_len"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"Invalid {name}")
        if self.embed_dim % self.n_heads or self.n_heads % self.n_kv_heads or (self.embed_dim // self.n_heads) % 2:
            raise ValueError("Invalid attention dimensions")


def model_config(c):
    return MaskedDiffusionConfig(vocab_size=c.VOCAB_SIZE, embed_dim=c.EMBED_DIM,
        n_heads=c.N_HEADS, n_kv_heads=c.N_KV_HEADS, n_layers=c.N_LAYERS,
        num_genres=c.NUM_GENRES, seq_len=c.SEQ_LEN, pad_id=c.PAD_ID)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.kv_heads, self.width = c.n_heads, c.n_kv_heads, c.embed_dim // c.n_heads
        self.q = nn.Linear(c.embed_dim, c.embed_dim, bias=False)
        self.k = nn.Linear(c.embed_dim, self.kv_heads * self.width, bias=False)
        self.v = nn.Linear(c.embed_dim, self.kv_heads * self.width, bias=False)
        self.out = nn.Linear(c.embed_dim, c.embed_dim, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.width), RMSNorm(self.width)
        self.rope = RotaryEmbedding(self.width)

    def forward(self, x, positions, valid):
        b, n, _ = x.shape
        q = self.rope(self.q_norm(self.q(x).reshape(b, n, self.heads, self.width)), positions)
        k = self.rope(self.k_norm(self.k(x).reshape(b, n, self.kv_heads, self.width)), positions)
        v = self.v(x).reshape(b, n, self.kv_heads, self.width)
        if flash_attn_varlen_func is not None and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            # Padded music must not enter attention; use the native GQA varlen path.
            lengths = valid.sum(-1, dtype=torch.int32)
            cu = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
            indices = valid.flatten().nonzero().flatten()
            q, k, v = (a.flatten(0, 1)[indices] for a in (q, k, v))
            maximum = int(lengths.max())
            attended = flash_attn_varlen_func(q, k, v, cu, cu, maximum, maximum,
                                             dropout_p=0., causal=False)
            result = x.new_zeros((b * n, self.heads, self.width)).index_copy(0, indices, attended)
            result = result.reshape(b, n, -1)
        else:
            attended = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                v.transpose(1, 2), attn_mask=valid[:, None, None], dropout_p=0.,
                is_causal=False, enable_gqa=self.heads != self.kv_heads)
            result = attended.transpose(1, 2).reshape(b, n, -1)
        return self.out(result)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.embed_dim), RMSNorm(c.embed_dim)
        self.attention = Attention(c)
        self.ffn = SwiGLU(c.embed_dim, dropout=0.)

    def forward(self, x, positions, valid):
        x = x + self.attention(self.norm1(x), positions, valid)
        return (x + self.ffn(self.norm2(x))).masked_fill(~valid[..., None], 0)


class MaskedDiffusionModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or MaskedDiffusionConfig()
        self.token_emb = nn.Embedding(c.vocab_size, c.embed_dim)
        # 0=structure, 1=instrument, 2=pitch/velocity, 3=onset, 4=duration.
        self.role_emb = nn.Embedding(5, c.embed_dim)
        self.genre_emb = nn.Embedding(c.num_genres, c.embed_dim)
        self.instrument_proj = nn.Linear(c.num_instruments, c.embed_dim, bias=False)
        self.rhythm_proj = nn.Sequential(nn.Linear(c.rhythm_dim, c.embed_dim), nn.SiLU(),
                                         nn.Linear(c.embed_dim, c.embed_dim))
        self.elasticity_proj = nn.Sequential(nn.Linear(1, c.embed_dim), nn.SiLU(),
                                             nn.Linear(c.embed_dim, c.embed_dim))
        self.layers = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.out_norm = RMSNorm(c.embed_dim)
        self.lm_head = nn.Linear(c.embed_dim, c.vocab_size, bias=False)
        for name, parameter in self.named_parameters():
            if parameter.ndim > 1:
                std = .02 / math.sqrt(2 * c.n_layers) if name.endswith(("out.weight", "w3.weight")) else .02
                nn.init.normal_(parameter, std=std)

    def forward(self, tokens, roles, valid, genre, multi_hot, rhythm, elasticity):
        if tokens.ndim != 2 or roles.shape != tokens.shape or valid.shape != tokens.shape:
            raise ValueError("Tokens, roles and validity must have matching [batch,length] shapes")
        b, n = tokens.shape
        c = self.config
        if n > c.seq_len or valid.dtype != torch.bool or not bool(valid.any(-1).all()):
            raise ValueError("Invalid sequence length/validity")
        if rhythm.shape != (b, c.rhythm_dim) or multi_hot.shape != (b, c.num_instruments):
            raise ValueError("Invalid conditioning shapes")
        if elasticity.shape != (b,) or not bool(torch.isfinite(elasticity).all() & (elasticity >= 0).all() & (elasticity <= 1).all()):
            raise ValueError("Elasticity must lie in [0,1]")
        condition = torch.stack((self.genre_emb(genre), self.instrument_proj(multi_hot),
            self.rhythm_proj(rhythm), self.elasticity_proj(elasticity[:, None])), 1)
        x = self.token_emb(tokens.masked_fill(~valid, c.pad_id)) + self.role_emb(roles.masked_fill(~valid, 0))
        x = torch.cat((condition, x), 1)
        valid = torch.cat((torch.ones(b, 4, dtype=torch.bool, device=x.device), valid), 1)
        positions = torch.arange(n + 4, device=x.device)[None].expand(b, -1)
        for layer in self.layers:
            if self.training and c.gradient_checkpointing:
                x = checkpoint(layer, x, positions, valid, use_reentrant=False)
            else:
                x = layer(x, positions, valid)
        return self.lm_head(self.out_norm(x[:, 4:]))
