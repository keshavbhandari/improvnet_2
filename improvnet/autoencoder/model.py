import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from improvnet.autoencoder.config import (
    VOCAB_SIZES, NUM_ATTRS, SEQ_LEN, EMBED_DIM, LATENT_DIM,
    N_HEADS, N_KV_HEADS, N_LAYERS, FFN_MULT, USE_CKPT,
    DEFAULT_MRA_BASE_VALUES, IS_CAUSAL, ATTN_DROPOUT, FFN_DROPOUT, PATCH_SIZE
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
# Encoder & Decoder Architecture
# ---------------------------------------------------------------------------

class TokenEncoder(nn.Module):
    def __init__(self, vocab_sizes, embed_dim, latent_dim, patch_size, n_heads, n_kv_heads, n_layers, ffn_mult, attn_dropout, ffn_dropout, use_ckpt, is_causal):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.patch_size = patch_size
        self.tok_emb = CompoundInputEmbed(embed_dim, vocab_sizes) 
        self.mra_cache = MRARotationCache(embed_dim // n_heads)
        
        self.layers = nn.ModuleList([
            MRAEncoderBlock(embed_dim, n_heads, n_kv_heads, ffn_mult, attn_dropout, ffn_dropout, is_causal) 
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(embed_dim)
        
        # PROJECTION CHANGE: It now maps (Patch_Size * Embed_Dim) down to Latent_Dim
        self.proj = nn.Linear(patch_size * embed_dim, latent_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.tok_emb(x)
        mra_cos, mra_sin = self.mra_cache(x)
        
        for layer in self.layers:
            if self.training and self.use_ckpt:
                h = checkpoint(layer, h, mra_cos, mra_sin, use_reentrant=False, preserve_rng_state=False)
            else:
                h = layer(h, mra_cos, mra_sin)
                
        h = self.norm_f(h)
        
        # INTERNAL PATCHING: Group the sequence into patches
        # [B, T, Embed_Dim] -> [B, T // Patch_Size, Patch_Size * Embed_Dim]
        num_patches = T // self.patch_size
        h_patches = h.view(B, num_patches, self.patch_size * h.shape[-1])
        
        # 1. Project to Latent Dim
        z_raw = self.proj(h_patches)
        
        return z_raw

class TokenDecoder(nn.Module):
    def __init__(self, vocab_sizes, max_seq_len, embed_dim, latent_dim, patch_size, n_heads, n_kv_heads, n_layers, ffn_mult, attn_dropout, ffn_dropout, use_ckpt, is_causal):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.patch_size = patch_size
        
        # PROJECTION CHANGE: Unpacks Latent_Dim back to (Patch_Size * Embed_Dim)
        self.latent_proj = nn.Linear(latent_dim, patch_size * embed_dim, bias=False)
        
        self.layers = nn.ModuleList([
            StandardTransformerBlock(embed_dim, n_heads, n_kv_heads, max_seq_len, ffn_mult, attn_dropout, ffn_dropout, is_causal) 
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(embed_dim)
        
        self.out_heads = nn.ModuleList([
            nn.Linear(embed_dim, vs, bias=False) for vs in vocab_sizes
        ])

    def forward(self, z: torch.Tensor):
        B, num_patches, _ = z.shape
        
        # Unpack the continuous latent patch
        h_patches = self.latent_proj(z)
        
        # INTERNAL UNPATCHING: Stretch back to a flat 1:1 sequence
        # [B, Num_Patches, Patch_Size * Embed_Dim] -> [B, T, Embed_Dim]
        h_seq = h_patches.view(B, num_patches * self.patch_size, -1)
        
        for layer in self.layers:
            if self.training and self.use_ckpt:
                h_seq = checkpoint(layer, h_seq, use_reentrant=False, preserve_rng_state=False)
            else:
                h_seq = layer(h_seq)
                
        h_seq = self.norm_f(h_seq)
        logits = [head(h_seq) for head in self.out_heads] 
        return logits

# ---------------------------------------------------------------------------
# Master Continuous Autoencoder 
# ---------------------------------------------------------------------------
class ContinuousAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        common = dict(
            vocab_sizes=VOCAB_SIZES, embed_dim=EMBED_DIM, latent_dim=LATENT_DIM,
            patch_size=PATCH_SIZE, # Pass the patch size!
            n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, n_layers=N_LAYERS, 
            ffn_mult=FFN_MULT, attn_dropout=ATTN_DROPOUT, ffn_dropout=FFN_DROPOUT, use_ckpt=USE_CKPT, is_causal=IS_CAUSAL
        )
        
        self.encoder = TokenEncoder(**common)
        self.fc_mu = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.decoder = TokenDecoder(**common, max_seq_len=SEQ_LEN)
        self.register_buffer("loss_weights", torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """The core of the VAE. Allows backpropagation through randomness."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu # During inference, we just use the pure mean!

    def forward(self, x: torch.Tensor):
        # 1. Get raw hidden state from encoder
        hidden = self.encoder(x)
        
        # 2. Project to Gaussian parameters
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        # Clamp logvar to prevent extreme float16/bfloat16 explosions early in training
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        
        # 3. Sample the latents! (No more tanh, no more manual noise injection)
        z = self.reparameterize(mu, logvar)
        
        # 4. Decode
        logits_list = self.decoder(z)
        
        return logits_list, mu, logvar

    def loss(self, 
             logits_list: list[torch.Tensor], 
             target_tensor: torch.Tensor, 
             mu: torch.Tensor,
             logvar: torch.Tensor,
             beta: float = 0.05) -> dict: # Beta controls KL strength
        """
        Calculates Cross Entropy + KL Divergence.
        """
        criterion = nn.CrossEntropyLoss(ignore_index=2)
        ce_loss = 0.0
        
        # 1. Standard Reconstruction Loss
        for i, logits in enumerate(logits_list):
            logits_reshaped = logits.transpose(1, 2)
            targets = target_tensor[:, :, i]
            # Upcast to float32 for safety!
            ce_loss += criterion(logits_reshaped.float(), targets)
            
        # 2. KL Divergence (Forces latents into a standard Gaussian sphere)
        # --- THE FIX: Upcast to float32 to prevent bfloat16 roundoff errors ---
        mu_fp32 = mu.float()
        logvar_fp32 = logvar.float()
        
        # Exact Mathematical Formula
        kl_raw = -0.5 * torch.sum(1 + logvar_fp32 - mu_fp32.pow(2) - logvar_fp32.exp())
        
        # --- Hard clamp to 0.0 to destroy floating point ghosts ---
        kl_clamped = torch.clamp(kl_raw, min=0.0)
        
        # Average KL loss across the batch so it scales cleanly
        kl_loss = kl_clamped / mu.shape[0] 
                
        # Total Loss (Beta-VAE configuration)
        total_loss = ce_loss + (beta * kl_loss)
        
        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "kl_loss": kl_loss
        }

    def calculate_accuracy(self, logits_list, x):
        # 1. Get predictions for all 5 streams
        preds = torch.stack([logits.argmax(-1) for logits in logits_list], dim=-1)
        
        # 2. Check if all 5 streams match exactly for each token
        exact_matches = (preds == x).all(dim=-1) 
        
        # --- THE FIX: Mask out the Padding Tokens ---
        # Assuming if stream 0 is padding (ID 2), the whole token is padding
        valid_mask = (x[..., 0] != 2)
        
        # 3. Extract only the matches that are NOT padding
        valid_matches = exact_matches[valid_mask]
        
        # 4. Safely calculate the mean (avoid division by zero if a batch is only padding)
        if valid_matches.numel() == 0:
            return 0.0
            
        return valid_matches.float().mean().item()

    # --- DOWNSTREAM INFERENCE API ---
    @torch.no_grad()
    def encode_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Used by the Flow Matching model to get the target latents."""
        self.eval()
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        # We drop the variance and the tanh! The Flow model gets the pure Gaussian Mean.
        return mu 

    @torch.no_grad()
    def decode_from_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Used by the Flow Matching model to turn generated latents into music."""
        self.eval()
        logits_list = self.decoder(z)
        predicted_tokens = torch.stack([logits.argmax(-1) for logits in logits_list], dim=-1)
        return predicted_tokens