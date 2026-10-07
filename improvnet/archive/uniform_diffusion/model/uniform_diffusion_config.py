"""Architecture settings for the AR-initialized, shared-weight diffusion model.

This module intentionally has no training paths or CUDA-dependent batch settings.
The defaults match the existing music AR backbone.
"""

from dataclasses import dataclass, field

from improvnet.tokenizer.absolute import AbsTokenizer


@dataclass(frozen=True)
class UniformDiffusionConfig:
    vocab_size: int = field(default_factory=lambda: AbsTokenizer().vocab_size)
    embed_dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4
    n_layers: int = 20
    num_genres: int = 4
    num_instruments: int = 41
    canvas_size: int = 256
    ffn_mult: float = 8 / 3
    dropout: float = 0.1
    gradient_checkpointing: bool = True

    def __post_init__(self):
        for name in (
            "vocab_size", "embed_dim", "n_heads", "n_kv_heads", "n_layers",
            "num_genres", "num_instruments", "canvas_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.embed_dim % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("embed_dim must divide into heads and query heads into KV groups")
        if (self.embed_dim // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not 0 <= self.dropout < 1 or self.ffn_mult <= 0:
            raise ValueError("Invalid dropout or feed-forward multiplier")
