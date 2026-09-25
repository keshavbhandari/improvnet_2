import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from improvnet.model.twotower_config import *

HAS_FLASH = False

try:
    from flash_attn import flash_attn_func

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()

        # FlashAttention v2 requires Ampere (SM80+) or newer.
        HAS_FLASH = major >= 8
except ImportError:
    HAS_FLASH = False

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.scale.float()).type_as(x)

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

class AdaLN(nn.Module):
    """
    Adaptive Layer Norm. 
    Upgraded to accept Sequence-Wise Timestep Embeddings [B, T, D]
    so each draft block receives its own distinct diffusion time modulation.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, 2 * embed_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        # Chunking across the feature dimension perfectly broadcasts across the sequence length [B, T, D]
        scale, shift = self.linear(t_emb).chunk(2, dim=-1)
        return x_norm * (1.0 + scale) + shift

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.freq_dim = freq_dim

    def forward(self, t):
        # t shape: [B, T]
        t_scaled = t * 1000.0
        half_dim = self.freq_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t_scaled.unsqueeze(-1) * emb.view(1, 1, -1)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)

class ElasticityEmbedder(nn.Module):
    """Embed the requested fraction of output positions reserved for blanks."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, elasticity):
        # Normalize the configured generation range to approximately [0, 1].
        scale = max(float(ELASTICITY_MAX_RATIO), 1e-8)
        return self.mlp((elasticity.float() / scale).unsqueeze(-1))

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=500000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, qk, seq_pos):
        freqs = torch.einsum("bt,f->btf", seq_pos.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(2).to(dtype=qk.dtype)
        sin = emb.sin().unsqueeze(2).to(dtype=qk.dtype)
        return (qk * cos) + (rotate_half(qk) * sin)

def sdpa_attn_func(q, k, v, dropout_p=0.0, causal=False, attn_mask=None):
    """flash_attn_func-compatible SDPA fallback.

    Inputs use FlashAttention layout: q is (B, Tq, Hq, D), while k and v
    are (B, Tk, Hkv, D). PyTorch SDPA handles grouped-query attention without
    explicitly repeating the cached KV heads.
    """
    hq = q.shape[2]
    hkv = k.shape[2]

    q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))

    if hq % hkv != 0:
        raise ValueError(f"Query heads ({hq}) must be divisible by KV heads ({hkv})")

    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
        enable_gqa=(hq != hkv),
    )
    return output.transpose(1, 2).contiguous()

class HybridGroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = embed_dim // n_heads
        
        self.q_proj = nn.Linear(embed_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, embed_dim, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)
        self.attn_fn = flash_attn_func if HAS_FLASH else sdpa_attn_func

    def _attend(self, q, k, v, dropout_p, key_mask=None):
        if key_mask is None:
            return self.attn_fn(q, k, v, dropout_p=dropout_p, causal=False)

        # FlashAttention's dense API has no key-padding-mask argument. SDPA
        # accepts a broadcastable boolean mask and still supports GQA.
        sdpa_mask = key_mask[:, None, None, :].to(device=q.device, dtype=torch.bool)
        return sdpa_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            causal=False,
            attn_mask=sdpa_mask,
        )

    def forward(
        self,
        x,
        seq_pos,
        context_kv=None,
        draft_size=0,
        attention_mask=None,
        context_attention_mask=None,
    ):
        B, T, C = x.shape

        if attention_mask is not None and attention_mask.shape != (B, T):
            raise ValueError(
                f"attention_mask must have shape ({B}, {T}), got {tuple(attention_mask.shape)}"
            )
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope(q, seq_pos)
        k = self.rope(k, seq_pos)

        # Concatenate the frozen AR KV cache with the active Denoiser KV cache
        if context_kv is not None:
            k_ctx, v_ctx = context_kv
            if (
                context_attention_mask is not None
                and context_attention_mask.shape != (B, k_ctx.shape[1])
            ):
                raise ValueError(
                    "context_attention_mask must have shape "
                    f"({B}, {k_ctx.shape[1]}), got {tuple(context_attention_mask.shape)}"
                )
            k = torch.cat([k_ctx, k], dim=1)
            v = torch.cat([v_ctx, v], dim=1)
            prefix_len = k_ctx.shape[1]
        else:
            if context_attention_mask is not None:
                raise ValueError("context_attention_mask requires context_kv")
            prefix_len = 0

        if attention_mask is None and context_attention_mask is None:
            # Preserve the fast FlashAttention path for callers that do not
            # need padding masks (for example, full-length inference blocks).
            key_mask = None
        else:
            active_key_mask = (
                attention_mask.to(device=x.device, dtype=torch.bool)
                if attention_mask is not None
                else torch.ones(B, T, device=x.device, dtype=torch.bool)
            )
            if prefix_len:
                prefix_key_mask = (
                    context_attention_mask.to(device=x.device, dtype=torch.bool)
                    if context_attention_mask is not None
                    else torch.ones(B, prefix_len, device=x.device, dtype=torch.bool)
                )
                key_mask = torch.cat([prefix_key_mask, active_key_mask], dim=1)
            else:
                key_mask = active_key_mask

        dropout_p = self.dropout.p if self.training else 0.0

        # -------------------------------------------------------------------
        # THE STAIRCASE ROUTER (Macro-Causal, Micro-Bidirectional)
        # -------------------------------------------------------------------
        if draft_size > 0 and T > 1:
            q_drafts = []
            curr_idx = 0
            
            while curr_idx < T:
                next_idx = min(curr_idx + draft_size, T)
                
                # Queries for THIS draft chunk
                q_d = q[:, curr_idx:next_idx]
                
                # Keys/Values visible up to the end of THIS draft chunk (+ the frozen prefix!)
                k_d = k[:, :prefix_len + next_idx] 
                v_d = v[:, :prefix_len + next_idx]
                
                # causal=False triggers bidirectional visibility WITHIN the visible window
                out_d = self._attend(
                    q_d,
                    k_d,
                    v_d,
                    dropout_p=dropout_p,
                    key_mask=(
                        key_mask[:, :prefix_len + next_idx]
                        if key_mask is not None
                        else None
                    ),
                )
                q_drafts.append(out_d)
                
                curr_idx = next_idx
                
            attn_out = torch.cat(q_drafts, dim=1)
        else:
            # Fallback for inference (1 token)
            attn_out = self._attend(q, k, v, dropout_p=dropout_p, key_mask=key_mask)

        out = self.o_proj(attn_out.reshape(B, T, C))
        return out

class DenoiserTransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, ffn_mult, dropout):
        super().__init__()
        self.norm1 = AdaLN(embed_dim)
        self.norm2 = AdaLN(embed_dim)
        self.attn  = HybridGroupedQueryAttention(embed_dim, n_heads, n_kv_heads, dropout)
        self.ffn = SwiGLU(embed_dim, mult=ffn_mult, dropout=dropout)

    def forward(
        self,
        x,
        t_emb,
        seq_pos,
        context_kv=None,
        draft_size=0,
        attention_mask=None,
        context_attention_mask=None,
    ):
        h = self.norm1(x, t_emb)
        attn_out = self.attn(
            h,
            seq_pos=seq_pos,
            context_kv=context_kv,
            draft_size=draft_size,
            attention_mask=attention_mask,
            context_attention_mask=context_attention_mask,
        )
        x = x + attn_out
        
        h = self.norm2(x, t_emb)
        x = x + self.ffn(h)
        return x

class TwoTowerDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.token_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.time_emb = TimestepEmbedder(EMBED_DIM)
        self.elasticity_emb = ElasticityEmbedder(EMBED_DIM)
        self.genre_emb = nn.Embedding(NUM_GENRES, EMBED_DIM)
        self.multihot_proj = nn.Linear(NUM_INSTRUMENTS, EMBED_DIM, bias=False)
        
        self.layers = nn.ModuleList([
            DenoiserTransformerBlock(
                EMBED_DIM, N_HEADS, N_KV_HEADS, ffn_mult=8/3, dropout=0.1
            ) for _ in range(N_LAYERS)
        ])
        
        self.out_norm = AdaLN(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)
        self._init_weights()

    def _init_weights(self):
        scale_factor = 1.0 / math.sqrt(2.0 * len(self.layers))
        for name, p in self.named_parameters():
            if p.dim() <= 1: continue
            if "o_proj.weight" in name or "w3.weight" in name:
                nn.init.normal_(p, mean=0.0, std=0.02 * scale_factor)
            elif any(k in name for k in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "w1.weight", "w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 * scale_factor)
            elif "inv_freq" in name: pass
            else:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        noisy_target,
        timestep,
        seq_offset=0,
        context_kv_cache=None,
        attention_mask=None,
        context_attention_mask=None,
        draft_size=0,
        elasticity=None,
        genre=None,
        multi_hot=None,
    ):
        """
        noisy_target: [Batch, Sequence_Length] (e.g., Draft 1 <SEP> Draft 2 <SEP> Draft 3)
        timestep: [Batch, Sequence_Length] (Distinct t-values for each token)
        seq_offset: scalar or [Batch] true prefix lengths, including AR controls
        attention_mask: [Batch, Sequence_Length] valid denoiser tokens
        context_attention_mask: [Batch, Cached_Length] valid Tower A cache entries
        draft_size: Determines the staircase slicing logic (e.g., BLOCK_SIZE + 1)
        elasticity: [Batch] requested <BLANK> fraction; 0 disables elasticity
        genre: [Batch] target genre, supplied directly to Tower B
        multi_hot: [Batch, NUM_INSTRUMENTS] desired output instrumentation
        """
        B, T = noisy_target.shape
        device = noisy_target.device

        x = self.token_emb(noisy_target) 
        
        # Both diffusion time and the requested sequence elasticity modulate
        # every denoiser block. Omitting the control means rigid generation.
        if elasticity is None:
            elasticity = torch.zeros(B, device=device, dtype=torch.float32)
        elif elasticity.ndim != 1 or elasticity.shape[0] != B:
            raise ValueError(f"elasticity must have shape ({B},), got {tuple(elasticity.shape)}")
        elasticity = elasticity.to(device=device, dtype=torch.float32)

        if genre is None:
            genre = torch.zeros(B, device=device, dtype=torch.long)
        elif genre.ndim != 1 or genre.shape[0] != B:
            raise ValueError(f"genre must have shape ({B},), got {tuple(genre.shape)}")
        genre = genre.to(device=device, dtype=torch.long)

        if multi_hot is None:
            multi_hot = torch.zeros(B, NUM_INSTRUMENTS, device=device, dtype=torch.float32)
        elif multi_hot.shape != (B, NUM_INSTRUMENTS):
            raise ValueError(
                f"multi_hot must have shape ({B}, {NUM_INSTRUMENTS}), "
                f"got {tuple(multi_hot.shape)}"
            )
        multi_hot = multi_hot.to(device=device, dtype=torch.float32)

        t_emb = self.time_emb(timestep)
        t_emb = t_emb + self.elasticity_emb(elasticity).unsqueeze(1).to(t_emb.dtype)
        t_emb = t_emb + self.genre_emb(genre).unsqueeze(1).to(t_emb.dtype)
        t_emb = t_emb + self.multihot_proj(multi_hot).unsqueeze(1).to(t_emb.dtype)
        
        # Continuous RoPE coordinates for the entire concatenated trajectory
        if torch.is_tensor(seq_offset):
            if seq_offset.ndim != 1 or seq_offset.shape[0] != B:
                raise ValueError(f"seq_offset must be scalar or have shape ({B},)")
            offsets = seq_offset.to(device=device, dtype=torch.long).unsqueeze(1)
            seq_pos = offsets + torch.arange(T, device=device).unsqueeze(0)
        else:
            seq_pos = torch.arange(seq_offset, seq_offset + T, device=device).unsqueeze(0).expand(B, -1)
            
        h = x
        for i, layer in enumerate(self.layers):
            layer_context_kv = context_kv_cache[i] if context_kv_cache is not None else None
            
            if self.training:
                h = checkpoint(
                    layer,
                    h,
                    t_emb,
                    seq_pos,
                    layer_context_kv,
                    draft_size,
                    attention_mask,
                    context_attention_mask,
                    use_reentrant=False,
                )
            else:
                h = layer(
                    h,
                    t_emb=t_emb,
                    seq_pos=seq_pos,
                    context_kv=layer_context_kv,
                    draft_size=draft_size,
                    attention_mask=attention_mask,
                    context_attention_mask=context_attention_mask,
                )

        h = self.out_norm(h, t_emb)
        logits = self.lm_head(h)

        return logits
