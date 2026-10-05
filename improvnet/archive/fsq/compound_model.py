import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from improvnet.fsq.compound_config import (
    VOCAB_SIZES, NUM_ATTRS, SEQ_LEN, EMBED_DIM, LATENT_DIM, LEVELS, NUM_QUANTIZERS,
    N_HEADS, N_KV_HEADS, N_LAYERS, FFN_MULT, USE_CKPT,
    DEFAULT_MRA_BASE_VALUES, IS_CAUSAL, ATTN_DROPOUT, FFN_DROPOUT
)

# ---------------------------------------------------------------------------
# SOTA Utilities (RMSNorm, SwiGLU, RoPE, Embeddings)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.scale

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

class CompoundInputEmbed(nn.Module):
    def __init__(self, embed_dim, vocab_sizes):
        super().__init__()
        self.attr_emb_dim = embed_dim // len(vocab_sizes)
        self.embeds = nn.ModuleList([
            nn.Embedding(vs, self.attr_emb_dim) for vs in vocab_sizes
        ])

    def forward(self, x):
        embs = [self.embeds[i](x[..., i]) for i in range(len(self.embeds))]
        return torch.cat(embs, dim=-1)

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

# ---------------------------------------------------------------------------
# Attention Modules (MRA for Encoder, Standard RoPE for Decoder)
# ---------------------------------------------------------------------------

class MRARotationCache(nn.Module):
    def __init__(self, head_dim, num_groups=5):
        super().__init__()
        self.head_dim = head_dim
        self.num_groups = num_groups
        for i in range(num_groups):
            base = DEFAULT_MRA_BASE_VALUES[i]
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
            self.register_buffer(f'inv_freq_{i}', inv_freq, persistent=False)

    def forward(self, attributes):
        # Calculates sine/cosine for all 5 groups exactly once per step
        mra_cos = []
        mra_sin = []
        for g in range(self.num_groups):
            pos_vals = attributes[:, :, g].float()
            inv_freq = getattr(self, f'inv_freq_{g}')
            sinusoid_inp = torch.einsum("b t, d -> b t d", pos_vals, inv_freq)
            sin = torch.sin(sinusoid_inp).unsqueeze(2).repeat_interleave(2, dim=-1)
            cos = torch.cos(sinusoid_inp).unsqueeze(2).repeat_interleave(2, dim=-1)
            mra_cos.append(cos)
            mra_sin.append(sin)
        return mra_cos, mra_sin


class MRAGroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, dropout, is_causal):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = dropout
        self.is_causal = is_causal

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

    def _apply_mra_rotation(self, tensor, mra_cos, mra_sin, heads_per_group):
        B, T, _, D = tensor.shape
        rotated_list = []
        for g in range(self.num_groups):
            start_head = g * heads_per_group
            end_head = (g + 1) * heads_per_group
            tensor_group = tensor[:, :, start_head:end_head, :]

            # Fetch the precomputed rotations!
            cos = mra_cos[g]
            sin = mra_sin[g]

            rotated = (tensor_group * cos) + (self._rotate_half(tensor_group) * sin)
            rotated_list.append(rotated)
        return torch.cat(rotated_list, dim=2)

    def forward(self, x, mra_cos, mra_sin):
        B, T, _ = x.shape
        H, Hkv, D = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, Hkv, D)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)

        # Pass the precomputed matrices instead of the raw attributes
        q = self._apply_mra_rotation(q, mra_cos, mra_sin, self.heads_per_group_q).transpose(1, 2)
        k = self._apply_mra_rotation(k, mra_cos, mra_sin, self.heads_per_group_kv).transpose(1, 2)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=self.is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out)

class StandardGroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, max_seq_len, dropout, is_causal):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = dropout
        self.is_causal = is_causal

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
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=self.is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out)

# ---------------------------------------------------------------------------
# SOTA Transformer Blocks
# ---------------------------------------------------------------------------
class MRAEncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, ffn_mult, attn_dropout, ffn_dropout, is_causal):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.attn  = MRAGroupedQueryAttention(embed_dim, n_heads, n_kv_heads, attn_dropout, is_causal)
        self.ffn   = SwiGLU(embed_dim, mult=ffn_mult, dropout=ffn_dropout)

    def forward(self, x, mra_cos, mra_sin):
        x = x + self.attn(self.norm1(x), mra_cos, mra_sin)
        x = x + self.ffn(self.norm2(x))
        return x

class StandardTransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, max_seq_len, ffn_mult, attn_dropout, ffn_dropout, is_causal):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.attn  = StandardGroupedQueryAttention(embed_dim, n_heads, n_kv_heads, max_seq_len, attn_dropout, is_causal)
        self.ffn   = SwiGLU(embed_dim, mult=ffn_mult, dropout=ffn_dropout)

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

class TokenEncoder(nn.Module):
    def __init__(self, vocab_sizes, embed_dim, latent_dim, n_heads, n_kv_heads, n_layers, ffn_mult, attn_dropout, ffn_dropout, use_ckpt, is_causal):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.tok_emb = CompoundInputEmbed(embed_dim, vocab_sizes) 
        
        # Add the centralized cache generator here!
        self.mra_cache = MRARotationCache(embed_dim // n_heads)
        
        self.layers = nn.ModuleList([
            MRAEncoderBlock(embed_dim, n_heads, n_kv_heads, ffn_mult, attn_dropout, ffn_dropout, is_causal) 
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, latent_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tok_emb(x)
        
        # Calculate Trigonometry ONCE per sequence
        mra_cos, mra_sin = self.mra_cache(x)
        
        for layer in self.layers:
            if self.training and self.use_ckpt:
                # Pass the precomputed matrices safely through gradient checkpointing
                h = checkpoint(layer, h, mra_cos, mra_sin, use_reentrant=False, preserve_rng_state=False)
            else:
                h = layer(h, mra_cos, mra_sin)
                
        return self.proj(self.norm_f(h))

class TokenDecoder(nn.Module):
    def __init__(self, vocab_sizes, max_seq_len, embed_dim, latent_dim, n_heads, n_kv_heads, n_layers, ffn_mult, attn_dropout, ffn_dropout, use_ckpt, is_causal):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.latent_proj = nn.Linear(latent_dim, embed_dim, bias=False)
        
        self.layers = nn.ModuleList([
            StandardTransformerBlock(embed_dim, n_heads, n_kv_heads, max_seq_len, ffn_mult, attn_dropout, ffn_dropout, is_causal) 
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(embed_dim)
        
        self.out_heads = nn.ModuleList([
            nn.Linear(embed_dim, vs, bias=False) for vs in vocab_sizes
        ])

    def forward(self, z_q: torch.Tensor):
        h = self.latent_proj(z_q)
        for layer in self.layers:
            if self.training and self.use_ckpt:
                h = checkpoint(layer, h, use_reentrant=False, preserve_rng_state=False)
            else:
                h = layer(h)
                
        h = self.norm_f(h)
        logits = [head(h) for head in self.out_heads] 
        return logits

# ---------------------------------------------------------------------------
# FSQ Autoencoder
# ---------------------------------------------------------------------------

class FSQAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        common = dict(
            vocab_sizes=VOCAB_SIZES, embed_dim=EMBED_DIM, latent_dim=LATENT_DIM,
            n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, n_layers=N_LAYERS, 
            ffn_mult=FFN_MULT, attn_dropout=ATTN_DROPOUT, ffn_dropout=FFN_DROPOUT, 
            use_ckpt=USE_CKPT, is_causal=IS_CAUSAL
        )
        
        # Encoder uses MRA (no max_seq_len needed)
        self.encoder = TokenEncoder(**common)
        
        self.fsq = ResidualFSQ(LEVELS, LATENT_DIM, NUM_QUANTIZERS)
        
        # Decoder uses standard RoPE (needs max_seq_len)
        self.decoder = TokenDecoder(**common, max_seq_len=SEQ_LEN)

        self.register_buffer("loss_weights", torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        z_q, indices = self.fsq(z)
        logits_list = self.decoder(z_q)
        return logits_list, z_q, indices

    def loss(self, logits_list, x: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for i in range(len(VOCAB_SIZES)):
            ce_loss = F.cross_entropy(
                logits_list[i].reshape(-1, logits_list[i].shape[-1]),
                x[..., i].reshape(-1)
            )
            total_loss += ce_loss * self.loss_weights[i]
        return total_loss

    def calculate_accuracy(self, logits_list, x):
        preds = torch.stack([logits.argmax(-1) for logits in logits_list], dim=-1)
        exact_matches = (preds == x).all(dim=-1) 
        return exact_matches.float().mean().item()

    # --- DOWNSTREAM INFERENCE API ---
    @torch.no_grad()
    def encode_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        z = self.encoder(x)
        z_q, _ = self.fsq(z)
        return z_q

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        z = self.encoder(x)
        _, indices = self.fsq(z)
        return indices

    @torch.no_grad()
    def decode_from_latents(self, z_q: torch.Tensor, snap_to_grid: bool = True) -> torch.Tensor:
        self.eval()
        if snap_to_grid:
            z_q = self.fsq.quantize(torch.tanh(z_q))
        logits_list = self.decoder(z_q)
        predicted_tokens = torch.stack([logits.argmax(-1) for logits in logits_list], dim=-1)
        return predicted_tokens
        
    @torch.no_grad()
    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        self.eval()
        B, T, num_q = indices.shape
        D = self.fsq.latent_dim
        
        z_q_total = torch.zeros((B, T, D), device=indices.device)
        current_scale = 1.0
        
        for q_idx in range(num_q):
            lvls = self.fsq.quantizers[q_idx].lvls
            z_q_i = lvls[indices[:, :, q_idx]]
            z_q_total += z_q_i / current_scale
            current_scale *= self.fsq.residual_scale
            
        logits_list = self.decoder(z_q_total)
        predicted_tokens = torch.stack([logits.argmax(-1) for logits in logits_list], dim=-1)
        return predicted_tokens