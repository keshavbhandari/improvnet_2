import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func
from improvnet.model.caddi_config import *

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

class FME(nn.Module):
    def __init__(self, out_dim, freq_dim=64, base=10000.0):
        super().__init__()
        self.out_dim = out_dim
        self.freq_dim = freq_dim
        half_dim = freq_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.proj = nn.Linear(freq_dim, out_dim)

    def forward(self, x):
        freqs = x.unsqueeze(-1).float() * self.inv_freq 
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1) 
        return self.proj(emb)

class CompoundInputEmbed(nn.Module):
    def __init__(self, embed_dim, vocab_sizes):
        super().__init__()
        self.attr_emb_dim = embed_dim // len(vocab_sizes)
        self.embeds = nn.ModuleList()
        self.embeds.append(nn.Embedding(vocab_sizes[0], self.attr_emb_dim))
        
        bases = [1000.0, 500.0, 10000.0, 10000.0]
        for i in range(1, len(vocab_sizes)):
            self.embeds.append(FME(self.attr_emb_dim, freq_dim=64, base=bases[i-1]))

    def forward(self, x):
        embs = [self.embeds[i](x[..., i]) for i in range(len(self.embeds))]
        return torch.cat(embs, dim=-1)

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

class MRARotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        
        self.dims = [20, 12, 12, 12, 12, 12]
        assert sum(self.dims) == head_dim, "MRA dims must perfectly sum to head_dim!"
        
        bases = [10000.0, 1000.0, 1000.0, 500.0, 10000.0, 10000.0]
        
        for i, d in enumerate(self.dims):
            base = bases[i]
            inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
            self.register_buffer(f"inv_freq_{i}", inv_freq)

    def forward(self, qk, coords):
        out_slices = []
        start_idx = 0
        
        for i, d in enumerate(self.dims):
            qk_slice = qk[..., start_idx : start_idx + d]
            
            c = coords[:, :, i].float() 
            inv_freq = getattr(self, f"inv_freq_{i}")
            freqs = torch.einsum("bt,f->btf", c, inv_freq) 
            emb = torch.cat((freqs, freqs), dim=-1) 
            
            cos = emb.cos().unsqueeze(2).to(dtype=qk.dtype) 
            sin = emb.sin().unsqueeze(2).to(dtype=qk.dtype)
            
            rotated_slice = (qk_slice * cos) + (rotate_half(qk_slice) * sin)
            out_slices.append(rotated_slice)
            
            start_idx += d
            
        return torch.cat(out_slices, dim=-1)

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
        
        self.rope = MRARotaryEmbedding(self.head_dim)

    def forward(self, x, coords, use_cache=False, kv_cache=None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope(q, coords)
        k = self.rope(k, coords)

        if use_cache:
            if kv_cache is not None:
                past_k, past_v = kv_cache
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        dropout_p = self.dropout.p if self.training else 0.0

        # PURE CAUSALITY: Because previous drafts are in the KV cache, the active block 
        # seamlessly gains "bidirectional" past context while remaining strictly causal!
        is_causal = (T > 1)
        attn_out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=is_causal)

        out = self.o_proj(attn_out.reshape(B, T, C))
        return out, new_kv_cache

class CausalTransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, ffn_mult, dropout):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.attn  = CausalGroupedQueryAttention(embed_dim, n_heads, n_kv_heads, dropout)
        self.ffn = SwiGLU(embed_dim, mult=ffn_mult, dropout=dropout)

    def forward(self, x, coords, use_cache=False, kv_cache=None):
        attn_out, new_kv_cache = self.attn(
            self.norm1(x), coords=coords, use_cache=use_cache, kv_cache=kv_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        if use_cache: return x, new_kv_cache
        return x

class SequentialGRUDecoder(nn.Module):
    def __init__(self, embed_dim, vocab_sizes):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.num_attrs = len(vocab_sizes)
        self.attr_emb_dim = embed_dim // self.num_attrs
        
        self.embs = nn.ModuleList()
        self.embs.append(nn.Embedding(vocab_sizes[0], self.attr_emb_dim))
        bases = [1000.0, 500.0, 10000.0, 10000.0]
        for i in range(1, len(vocab_sizes)):
            self.embs.append(FME(self.attr_emb_dim, freq_dim=64, base=bases[i-1]))

        self.start_emb = nn.Parameter(torch.randn(1, self.attr_emb_dim) * 0.02)
        self.gru = nn.GRUCell(self.attr_emb_dim, embed_dim)
        
        self.out_heads = nn.ModuleList([
            nn.Linear(embed_dim, vs, bias=False) for vs in vocab_sizes
        ])

    def forward(self, h_transformer, target_attrs):
        B, T, D = h_transformer.shape
        h_gru = h_transformer.reshape(B * T, D)
        logits_list = []

        gru_in = self.start_emb.expand(B * T, -1)
        h_gru = self.gru(gru_in, h_gru)
        logits_0 = self.out_heads[0](h_gru)
        logits_list.append(logits_0.view(B, T, -1))

        for i in range(1, self.num_attrs):
            prev_attr = target_attrs[:, :, i-1].contiguous().view(B * T)
            gru_in = self.embs[i-1](prev_attr)
            h_gru = self.gru(gru_in, h_gru)
            logits_i = self.out_heads[i](h_gru)
            logits_list.append(logits_i.view(B, T, -1))

        return logits_list

class CaDDiModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.genre_emb = nn.Embedding(NUM_GENRES, EMBED_DIM)
        self.time_emb = TimestepEmbedder(EMBED_DIM)
        self.target_emb = CompoundInputEmbed(EMBED_DIM, VOCAB_SIZES)
        
        self.layers = nn.ModuleList([
            CausalTransformerBlock(
                EMBED_DIM, N_HEADS, N_KV_HEADS, ffn_mult=8/3, dropout=0.1
            ) for _ in range(N_LAYERS)
        ])
        self.out_norm = RMSNorm(EMBED_DIM)
        self.gru_decoder = SequentialGRUDecoder(EMBED_DIM, VOCAB_SIZES)
        self._init_weights()

    def _init_weights(self):
        scale_factor = 1.0 / math.sqrt(2.0 * len(self.layers))
        for name, p in self.named_parameters():
            if p.dim() <= 1:
                continue
            if "o_proj.weight" in name or "w3.weight" in name:
                nn.init.normal_(p, mean=0.0, std=0.02 * scale_factor)
            elif any(k in name for k in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "w1.weight", "w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 * scale_factor)
            elif "inv_freq" in name:
                pass
            else:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self, target, coords, timestep, genre, 
        use_cache=False, past_key_values=None
    ):
        e_target = self.target_emb(target) 
        
        # Inject dynamic Timestep embedding
        e_time = self.time_emb(timestep)
        e_target = e_target + e_time
        
        is_first_step = past_key_values is None
        if is_first_step:
            e_genre = self.genre_emb(genre)
            e_target = e_target.clone()
            e_target[:, 0, :] = e_target[:, 0, :] + e_genre
            
        h = e_target
        next_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            current_kv = past_key_values[i] if not is_first_step else None
            
            if not use_cache and self.training:
                layer_out = checkpoint(layer, h, coords, use_cache, current_kv, use_reentrant=False)
            else:
                layer_out = layer(h, coords=coords, use_cache=use_cache, kv_cache=current_kv)
            
            if use_cache:
                h, new_kv = layer_out
                next_key_values.append(new_kv)
            else:
                h = layer_out

        h = self.out_norm(h)

        if use_cache:
            return h, next_key_values
            
        return self.gru_decoder(h, target)