import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func
from improvnet.model.omni_config import *

NUM_INSTRUMENTS = 41 # Based on ProcessData.INSTRUMENT_CLASSES

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
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
        t_scaled = t * 1000.0
        half_dim = self.freq_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t_scaled.unsqueeze(-1) * emb.view(1, 1, -1)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# ---------------------------------------------------------------------------
# STANDARD 1D SEQUENCE ROPE
# ---------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000.0):
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


# ---------------------------------------------------------------------------
# OMNI-DIRECTIONAL ATTENTION ROUTER
# ---------------------------------------------------------------------------
class OmniGroupedQueryAttention(nn.Module):
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

    def forward(self, x, seq_pos, causal_prefix_len=0, draft_size=0, use_cache=False, kv_cache=None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope(q, seq_pos)
        k = self.rope(k, seq_pos)

        if use_cache:
            if kv_cache is not None:
                past_k, past_v = kv_cache
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        dropout_p = self.dropout.p if self.training else 0.0
        
        total_q_len = q.shape[1]
        total_k_len = k.shape[1]

        # -------------------------------------------------------------------
        # THE ATTENTION ROUTER
        # -------------------------------------------------------------------
        if total_q_len == 1:
            # Inference Step: Pure token-by-token generation. It naturally attends to all past context.
            attn_out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=False)
            
        elif causal_prefix_len > 0 and causal_prefix_len < total_q_len and draft_size > 0:
            # Training / Parallel Block Step: MACRO-CAUSAL, MICRO-BIDIRECTIONAL
            
            # 1. Evaluate Prefix (Strictly Causal)
            q_prefix = q[:, :causal_prefix_len]
            k_prefix = k[:, :causal_prefix_len]
            v_prefix = v[:, :causal_prefix_len]
            out_prefix = flash_attn_func(q_prefix, k_prefix, v_prefix, dropout_p=dropout_p, causal=True)
            
            # 2. Evaluate Successive Drafts (Bidirectional internally, Causal externally)
            q_drafts = []
            curr_idx = causal_prefix_len
            
            while curr_idx < total_q_len:
                next_idx = min(curr_idx + draft_size, total_q_len)
                
                q_d = q[:, curr_idx:next_idx]
                k_d = k[:, :next_idx]  # Keys strictly bounded to the end of the CURRENT draft
                v_d = v[:, :next_idx]
                
                # causal=False triggers full bidirectional visibility within the sliced window!
                out_d = flash_attn_func(q_d, k_d, v_d, dropout_p=dropout_p, causal=False)
                q_drafts.append(out_d)
                
                curr_idx = next_idx
                
            attn_out = torch.cat([out_prefix] + q_drafts, dim=1)
            
        else:
            # Fallback for standard sequence generation (e.g., evaluating a pure causal prefix)
            attn_out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=True)

        out = self.o_proj(attn_out.reshape(B, T, C))
        return out, new_kv_cache


class OmniTransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, ffn_mult, dropout):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.attn  = OmniGroupedQueryAttention(embed_dim, n_heads, n_kv_heads, dropout)
        self.ffn = SwiGLU(embed_dim, mult=ffn_mult, dropout=dropout)

    def forward(self, x, seq_pos, causal_prefix_len=0, draft_size=0, use_cache=False, kv_cache=None):
        attn_out, new_kv_cache = self.attn(
            self.norm1(x), seq_pos=seq_pos, 
            causal_prefix_len=causal_prefix_len, draft_size=draft_size,
            use_cache=use_cache, kv_cache=kv_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        if use_cache: return x, new_kv_cache
        return x


class CaDDiModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.token_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.time_emb = TimestepEmbedder(EMBED_DIM)
        
        # Omni-CaDDi Conditional Embeddings
        self.genre_emb = nn.Embedding(NUM_GENRES, EMBED_DIM)
        self.mode_emb = nn.Embedding(2, EMBED_DIM)       # 0: STRICT, 1: EDIT
        self.len_emb = nn.Embedding(2, EMBED_DIM)        # 0: FIXED, 1: ELASTIC
        self.multihot_proj = nn.Linear(NUM_INSTRUMENTS, EMBED_DIM, bias=False)
        
        self.layers = nn.ModuleList([
            OmniTransformerBlock(
                EMBED_DIM, N_HEADS, N_KV_HEADS, ffn_mult=8/3, dropout=0.1
            ) for _ in range(N_LAYERS)
        ])
        
        self.out_norm = RMSNorm(EMBED_DIM)
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
        self, target, timestep, genre, 
        mode=None, length_ctrl=None, multi_hot=None,
        causal_prefix_len=0, draft_size=0,
        seq_offset=0, use_cache=False, past_key_values=None
    ):
        B, T = target.shape
        device = target.device

        # Handle optional controls gracefully
        if mode is None: mode = torch.zeros(B, dtype=torch.long, device=device)
        if length_ctrl is None: length_ctrl = torch.zeros(B, dtype=torch.long, device=device)
        if multi_hot is None: multi_hot = torch.zeros((B, NUM_INSTRUMENTS), dtype=torch.float32, device=device)

        x = self.token_emb(target) 
        
        # Inject dynamic Timestep embedding smoothly across the continuous sequence
        e_time = self.time_emb(timestep)
        x = x + e_time
        
        is_first_step = past_key_values is None
        if is_first_step:
            # Prepend the 4 structural condition vectors
            e_genre = self.genre_emb(genre).unsqueeze(1)
            e_mode = self.mode_emb(mode).unsqueeze(1)
            e_len = self.len_emb(length_ctrl).unsqueeze(1)
            e_mh = self.multihot_proj(multi_hot).unsqueeze(1)
            
            x = torch.cat([e_genre, e_mode, e_len, e_mh, x], dim=1) 
            
            # The 4 control tokens count as the first 4 indices, sequence shifts right by 4
            seq_pos = torch.arange(seq_offset, seq_offset + x.size(1), device=device).unsqueeze(0).expand(B, -1)
            
            # Update the causal router boundary to include the 4 new control tokens
            actual_prefix_len = causal_prefix_len + 4 if causal_prefix_len > 0 else 4
        else:
            seq_pos = torch.arange(seq_offset, seq_offset + T, device=device).unsqueeze(0).expand(B, -1)
            actual_prefix_len = 0 # Pure inference loop, handled by KV cache
            
        h = x
        next_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            current_kv = past_key_values[i] if not is_first_step else None
            
            kwargs = {
                "seq_pos": seq_pos,
                "causal_prefix_len": actual_prefix_len,
                "draft_size": draft_size,
                "use_cache": use_cache,
                "kv_cache": current_kv
            }
            
            if not use_cache and self.training:
                layer_out = checkpoint(layer, h, use_reentrant=False, **kwargs)
            else:
                layer_out = layer(h, **kwargs)
            
            if use_cache:
                h, new_kv = layer_out
                next_key_values.append(new_kv)
            else:
                h = layer_out

        h = self.out_norm(h)
        
        # If we injected the 4 Control tokens at the start, drop them from the output
        # to ensure the logits array perfectly aligns with the target labels
        if is_first_step:
            h = h[:, 4:, :]

        logits = self.lm_head(h)

        if use_cache: return logits, next_key_values
        return logits