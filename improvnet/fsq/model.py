import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from improvnet.fsq.config import (
    VOCAB_SIZES, NUM_ATTRS, PATCH_SIZE, EMBED_DIM, LATENT_DIM, LEVELS, NUM_QUANTIZERS,
    N_HEADS, N_KV_HEADS, N_LAYERS, FFN_MULT, DROPOUT, USE_CKPT,
    DEFAULT_MRA_BASE_VALUES
)

# ---------------------------------------------------------------------------
# Embedding & Normalization
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.scale

class CompoundInputEmbed(nn.Module):
    """Embeds 5 discrete attributes and concatenates them to EMBED_DIM"""
    def __init__(self, embed_dim, vocab_sizes):
        super().__init__()
        self.attr_emb_dim = embed_dim // len(vocab_sizes)
        self.embeds = nn.ModuleList([
            nn.Embedding(vs, self.attr_emb_dim) for vs in vocab_sizes
        ])

    def forward(self, x):
        # x shape: [..., 5]
        embs = [self.embeds[i](x[..., i]) for i in range(len(self.embeds))]
        return torch.cat(embs, dim=-1)

# ---------------------------------------------------------------------------
# Multidimensional Relative Attention (MRA) for Encoder
# ---------------------------------------------------------------------------

class MRAGroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = dropout

        self.num_groups = NUM_ATTRS
        self.heads_per_group_q = n_heads // self.num_groups
        self.heads_per_group_kv = n_kv_heads // self.num_groups

        for i in range(self.num_groups):
            base = DEFAULT_MRA_BASE_VALUES[i]
            inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer(f'inv_freq_{i}', inv_freq, persistent=False)

        self.q_proj = nn.Linear(embed_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_mra_rotation(self, tensor, attributes, heads_per_group):
        B, T, _, D = tensor.shape
        rotated_list = []
        for g in range(self.num_groups):
            start_head = g * heads_per_group
            end_head = (g + 1) * heads_per_group
            tensor_group = tensor[:, :, start_head:end_head, :]

            pos_vals = attributes[:, :, g].float()
            inv_freq = getattr(self, f'inv_freq_{g}')

            sinusoid_inp = torch.einsum("b t, d -> b t d", pos_vals, inv_freq)
            sin = torch.sin(sinusoid_inp).unsqueeze(2).repeat_interleave(2, dim=-1)
            cos = torch.cos(sinusoid_inp).unsqueeze(2).repeat_interleave(2, dim=-1)

            rotated = (tensor_group * cos) + (self._rotate_half(tensor_group) * sin)
            rotated_list.append(rotated)

        return torch.cat(rotated_list, dim=2)

    def forward(self, x, attributes):
        B, T, _ = x.shape
        H, Hkv, D = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, Hkv, D)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)

        q = self._apply_mra_rotation(q, attributes, self.heads_per_group_q).transpose(1, 2)
        k = self._apply_mra_rotation(k, attributes, self.heads_per_group_kv).transpose(1, 2)

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, Hkv, self.n_rep, T, D).reshape(B, H, T, D)
            v = v.unsqueeze(2).expand(B, Hkv, self.n_rep, T, D).reshape(B, H, T, D)

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out)

# ---------------------------------------------------------------------------
# Standard RoPE & GQA for PatchDecoder (Non-autoregressive)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: int = 10_000):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        if seq_len > self.cos_cache.shape[0]:
            self._build_cache(seq_len * 2)
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin

class StandardGroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, max_seq_len=256, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, Hkv, D = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)

        q = self.rope(q, T)
        k = self.rope(k, T)

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, Hkv, self.n_rep, T, D).reshape(B, H, T, D)
            v = v.unsqueeze(2).expand(B, Hkv, self.n_rep, T, D).reshape(B, H, T, D)

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out)

# ---------------------------------------------------------------------------
# Transformer Blocks
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, embed_dim: int, mult: float = 8/3, dropout: float = 0.0):
        super().__init__()
        hidden = int(math.ceil(embed_dim * mult / 64) * 64)
        self.w1 = nn.Linear(embed_dim, hidden, bias=False)
        self.w2 = nn.Linear(embed_dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class MRAEncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, ffn_mult, dropout):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.attn  = MRAGroupedQueryAttention(embed_dim, n_heads, n_kv_heads, dropout)
        self.ffn   = SwiGLU(embed_dim, mult=ffn_mult, dropout=dropout)

    def forward(self, x, attributes):
        x = x + self.attn(self.norm1(x), attributes)
        x = x + self.ffn(self.norm2(x))
        return x

class StandardDecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, max_seq_len, ffn_mult, dropout):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.attn  = StandardGroupedQueryAttention(embed_dim, n_heads, n_kv_heads, max_seq_len, dropout)
        self.ffn   = SwiGLU(embed_dim, mult=ffn_mult, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# ---------------------------------------------------------------------------
# FSQ
# ---------------------------------------------------------------------------

class FSQ(nn.Module):
    def __init__(self, levels: int, latent_dim: int):
        super().__init__()
        self.levels = levels
        self.latent_dim = latent_dim
        self.register_buffer("lvls", torch.linspace(-1.0, 1.0, levels))

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        z_exp = z.unsqueeze(-1)
        lvls = self.lvls.view(*([1] * z.dim()), -1)
        indices = (z_exp - lvls).abs().argmin(dim=-1)
        return self.lvls[indices]

    def forward(self, z: torch.Tensor):
        # iFSQ
        z_b = 2.0 * torch.sigmoid(1.6 * z) - 1.0
        z_q = self.quantize(z_b)
        z_q_ste = z_b + (z_q - z_b).detach()
        indices = ((z_q + 1.0) / 2.0 * (self.levels - 1)).round().long()
        return z_q_ste, indices

    @property
    def codebook_size(self) -> int:
        return self.levels ** self.latent_dim


class ResidualFSQ(nn.Module):
    def __init__(self, levels: int, latent_dim: int, num_quantizers: int = 4):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.levels = levels
        self.latent_dim = latent_dim

        # Create multiple FSQ stages
        self.quantizers = nn.ModuleList([FSQ(levels, latent_dim) for _ in range(num_quantizers)])

        # Scaling factor to stretch the residual back to the [-1, 1] grid
        self.residual_scale = levels - 1.0

    def forward(self, z: torch.Tensor):
        z_q_total = 0
        all_indices = []

        # FSQ bounds the initial continuous vector to [-1, 1]
        current_z = torch.tanh(z)
        current_scale = 1.0

        for i in range(self.num_quantizers):
            # 1. Quantize the current state
            z_q = self.quantizers[i].quantize(current_z)

            # 2. Straight-Through Estimator (STE) for backprop
            z_q_ste = current_z + (z_q - current_z).detach()

            # 3. Add to the total reconstruction, scaling down based on the stage depth
            z_q_total = z_q_total + (z_q_ste / current_scale)

            # 4. Calculate integer indices for this stage
            indices = ((z_q + 1.0) / 2.0 * (self.levels - 1)).round().long()
            all_indices.append(indices)

            # 5. Compute the residual error and scale it UP for the next quantizer
            residual = current_z - z_q
            current_z = residual * self.residual_scale
            current_scale *= self.residual_scale

        # Returns summed latents [B, P, D] and stacked indices [B, P, Num_Quantizers, D]
        return z_q_total, torch.stack(all_indices, dim=-2)

    @property
    def codebook_size(self) -> int:
        return (self.levels ** self.latent_dim) ** self.num_quantizers

# ---------------------------------------------------------------------------
# Encoder & Decoder Architecture
# ---------------------------------------------------------------------------

class PatchEncoder(nn.Module):
    def __init__(self, vocab_sizes, patch_size, embed_dim, latent_dim, n_heads, n_kv_heads, n_layers, ffn_mult, dropout, use_ckpt):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.tok_emb = CompoundInputEmbed(embed_dim, vocab_sizes)
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.layers = nn.ModuleList([
            MRAEncoderBlock(embed_dim, n_heads, n_kv_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, latent_dim, bias=False)
        nn.init.trunc_normal_(self.cls_tok, std=0.02)

    def _run_layer(self, layer, x, attr):
        if self.use_ckpt and self.training:
            return checkpoint(layer, x, attr, use_reentrant=False)
        return layer(x, attr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, T, num_attr = x.shape
        x_flat = x.view(B * P, T, num_attr)

        # Add a dummy zero attribute for the CLS token to safely pass through MRA
        cls_attr = torch.zeros((B * P, 1, num_attr), device=x.device, dtype=x.dtype)
        attributes = torch.cat([cls_attr, x_flat], dim=1) # [B*P, T+1, 5]

        h = self.tok_emb(x_flat)
        cls = self.cls_tok.expand(B * P, -1, -1)
        h = torch.cat([cls, h], dim=1) # [B*P, T+1, E]

        for layer in self.layers:
            h = self._run_layer(layer, h, attributes)

        h = self.norm(h)
        cls_out = h[:, 0]
        z = self.proj(cls_out)
        return z.view(B, P, -1)

class PatchDecoder(nn.Module):
    def __init__(self, vocab_sizes, patch_size, embed_dim, latent_dim, n_heads, n_kv_heads, n_layers, ffn_mult, dropout, use_ckpt):
        super().__init__()
        self.patch_size = patch_size
        self.vocab_sizes = vocab_sizes
        self.use_ckpt = use_ckpt

        self.latent_proj = nn.Linear(latent_dim, embed_dim, bias=False)
        self.pos_queries = nn.Parameter(torch.zeros(1, patch_size, embed_dim))

        self.layers = nn.ModuleList([
            StandardDecoderBlock(embed_dim, n_heads, n_kv_heads, patch_size, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(embed_dim)

        # 5 output heads
        self.out_heads = nn.ModuleList([
            nn.Linear(embed_dim, vs, bias=False) for vs in vocab_sizes
        ])
        nn.init.trunc_normal_(self.pos_queries, std=0.02)

    def _run_layer(self, layer, x):
        if self.use_ckpt and self.training:
            return checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    def forward(self, z_q: torch.Tensor):
        B, P, D = z_q.shape
        T = self.patch_size

        z_flat = z_q.view(B * P, D)
        lat = self.latent_proj(z_flat)

        queries = self.pos_queries.expand(B * P, -1, -1)
        h = lat.unsqueeze(1) + queries

        for layer in self.layers:
            h = self._run_layer(layer, h)

        h = self.norm(h) # [B*P, T, E]

        # Process individual heads
        logits = [head(h).view(B, P, T, -1) for head in self.out_heads]
        return logits

# ---------------------------------------------------------------------------
# FSQ Autoencoder
# ---------------------------------------------------------------------------

class FSQAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        common = dict(
            vocab_sizes=VOCAB_SIZES, patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM, latent_dim=LATENT_DIM,
            n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
            n_layers=N_LAYERS, ffn_mult=FFN_MULT,
            dropout=DROPOUT, use_ckpt=USE_CKPT,
        )
        self.encoder = PatchEncoder(**common)
        self.fsq = ResidualFSQ(LEVELS, LATENT_DIM, NUM_QUANTIZERS)
        self.decoder = PatchDecoder(**common)

        # [Instrument, Pitch, Velocity, Onset, Duration]
        self.register_buffer("loss_weights", torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        z_q, indices = self.fsq(z)
        logits_list = self.decoder(z_q)
        return logits_list, z_q, indices

    def loss(self, logits_list, x: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for i in range(NUM_ATTRS):
            # Calculate cross entropy for this attribute
            ce_loss = F.cross_entropy(
                logits_list[i].reshape(-1, logits_list[i].shape[-1]),
                x[..., i].reshape(-1)
            )
            # Multiply by its specific weight
            total_loss += ce_loss * self.loss_weights[i]

        return total_loss

    def calculate_accuracy(self, logits_list, x):
        """Calculates percentage of sequences where ALL 5 attributes match exactly"""
        preds = torch.stack([logits.argmax(-1) for logits in logits_list], dim=-1)
        exact_matches = (preds == x).all(dim=-1) # True if all 5 attributes match for a token
        return exact_matches.float().mean().item()

    # --- Downstream Encode/Decode API ---
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns integer indices [B, P, D] for Discrete Transformer modeling"""
        _, indices = self.fsq(self.encoder(x))
        return indices

    def encode_to_floats(self, x: torch.Tensor) -> torch.Tensor:
        """Returns continuous latent floats [B, P, D] for Flow/Diffusion modeling"""
        z_q, _ = self.fsq(self.encoder(x))
        return z_q

    def decode_from_indices(self, indices: torch.Tensor):
        """Converts indices back to grid floats, then decodes to logits and tokens."""
        z_q = self.fsq.lvls[indices]
        logits_list = self.decoder(z_q)
        tokens = torch.stack([logits.argmax(-1) for logits in logits_list], dim=-1)
        return logits_list, tokens

    def decode_from_floats(self, z: torch.Tensor, snap_to_grid: bool = True):
        """Decodes continuous latents. Snaps to grid if it originated from a continuous generator."""
        if snap_to_grid:
            z = self.fsq.quantize(torch.tanh(z))
        logits_list = self.decoder(z)
        tokens = torch.stack([logits.argmax(-1) for logits in logits_list], dim=-1)
        return logits_list, tokens

    @property
    def codebook_size(self) -> int:
        return self.fsq.codebook_size