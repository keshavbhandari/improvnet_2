import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func
from fla.layers import DeltaNet
from improvnet.model.hybrid_ar_config import *

NUM_INSTRUMENTS = 41 # Matches the length of processor.INSTRUMENT_CLASSES

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

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

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

class CausalGroupedQueryAttention(nn.Module):
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

    def forward(self, x, seq_pos, use_cache=False, kv_cache=None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply pure 1D RoPE based on the absolute sequence position
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
        
        # Flash Attention Causal Router Quirk
        is_causal = True if q.size(1) > 1 else False

        attn_out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=is_causal)
        out = self.o_proj(attn_out.reshape(B, T, C))
        return out, new_kv_cache

class ARTransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, ffn_mult, dropout, is_flash=False):
        super().__init__()
        self.is_flash = is_flash
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        
        if is_flash:
            # The Anchor Point
            self.attn = CausalGroupedQueryAttention(embed_dim, n_heads, n_kv_heads, dropout)
        else:
            # The Sequence Compressor
            self.attn = DeltaNet(
                hidden_size=embed_dim, 
                num_heads=n_heads,
                use_gate=True
            )
            
        self.ffn = SwiGLU(embed_dim, mult=ffn_mult, dropout=dropout)

    def forward(self, x, seq_pos, use_cache=False, kv_cache=None):
        norm_x = self.norm1(x)
        
        if self.is_flash:
            attn_out, new_kv_cache = self.attn(
                norm_x, seq_pos=seq_pos, use_cache=use_cache, kv_cache=kv_cache
            )
        else:
            if use_cache:
                attn_out, new_kv_cache = self.attn(norm_x, state=kv_cache)
            else:
                attn_out = self.attn(norm_x)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                new_kv_cache = None

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        
        if use_cache: 
            return x, new_kv_cache
        return x

class ARContextModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.token_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.genre_emb = nn.Embedding(NUM_GENRES, EMBED_DIM)
        self.multihot_proj = nn.Linear(NUM_INSTRUMENTS, EMBED_DIM, bias=False)
        
        # 3 DeltaNet : 1 Flash Attention (Every 4th layer is an Anchor Point)
        self.layers = nn.ModuleList([
            ARTransformerBlock(
                EMBED_DIM, N_HEADS, N_KV_HEADS, ffn_mult=8/3, dropout=0.1,
                is_flash=((i + 1) % 4 == 0)
            ) for i in range(N_LAYERS)
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
        self, target, genre=None, multi_hot=None, 
        seq_offset=0, use_cache=False, past_key_values=None,
        return_hidden=False # Flag to support Liger Kernel!
    ):
        B, T = target.shape
        device = target.device

        x = self.token_emb(target) 
        
        is_first_step = past_key_values is None
        
        if is_first_step:
            if genre is None: genre = torch.zeros(B, dtype=torch.long, device=device)
            if multi_hot is None: multi_hot = torch.zeros((B, NUM_INSTRUMENTS), dtype=torch.float32, device=device)
            
            e_genre = self.genre_emb(genre).unsqueeze(1)
            e_mh = self.multihot_proj(multi_hot).unsqueeze(1)
            
            x = torch.cat([e_genre, e_mh, x], dim=1) 
            seq_pos = torch.arange(seq_offset, seq_offset + x.size(1), device=device).unsqueeze(0).expand(B, -1)
        else:
            seq_pos = torch.arange(seq_offset, seq_offset + T, device=device).unsqueeze(0).expand(B, -1)
            
        h = x
        next_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            current_kv = past_key_values[i] if not is_first_step else None
            
            if not use_cache and self.training:
                layer_out = checkpoint(layer, h, seq_pos, use_cache, current_kv, use_reentrant=False)
            else:
                layer_out = layer(h, seq_pos=seq_pos, use_cache=use_cache, kv_cache=current_kv)
            
            if use_cache:
                h, new_kv = layer_out
                next_key_values.append(new_kv)
            else:
                h = layer_out

        h = self.out_norm(h)
        
        if is_first_step:
            h = h[:, 2:, :]

        # Short-circuit for Liger Kernel training to avoid materializing [B, T, V] logits
        if return_hidden:
            return h

        logits = self.lm_head(h)

        if use_cache: 
            return logits, next_key_values
        return logits