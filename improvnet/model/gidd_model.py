import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func
from improvnet.model.gidd_config import *


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

# ---------------------------------------------------------------------------
# FUNDAMENTAL MUSIC EMBEDDING (FME)
# ---------------------------------------------------------------------------
class FME(nn.Module):
    def __init__(self, out_dim, freq_dim=64, base=10000.0):
        """
        Projects continuous music tokens into a continuous sinusoidal space 
        to preserve relative magnitude and translation-equivariance.
        """
        super().__init__()
        self.out_dim = out_dim
        self.freq_dim = freq_dim
        half_dim = freq_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.proj = nn.Linear(freq_dim, out_dim)

    def forward(self, x):
        # x shape: [B, T] or [B * T] depending on caller
        freqs = x.unsqueeze(-1).float() * self.inv_freq 
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1) 
        return self.proj(emb)

class CompoundInputEmbed(nn.Module):
    def __init__(self, embed_dim, vocab_sizes):
        super().__init__()
        self.attr_emb_dim = embed_dim // len(vocab_sizes)
        self.embeds = nn.ModuleList()
        
        # Index 0: Instrument (Categorical) -> Standard Embedding
        self.embeds.append(nn.Embedding(vocab_sizes[0], self.attr_emb_dim))
        
        # Indices 1..4: Pitch, Velocity, Onset, Duration (Continuous) -> FME
        bases = [1000.0, 500.0, 10000.0, 10000.0]
        for i in range(1, len(vocab_sizes)):
            self.embeds.append(FME(self.attr_emb_dim, freq_dim=64, base=bases[i-1]))

    def forward(self, x):
        embs = [self.embeds[i](x[..., i]) for i in range(len(self.embeds))]
        return torch.cat(embs, dim=-1)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# ---------------------------------------------------------------------------
# GIDD TIMESTEP EMBEDDER (Sinusoidal)
# ---------------------------------------------------------------------------
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
        """
        Expects a float tensor t in range [0, 1].
        Scales it to [0, 1000] for standard sinusoidal frequency variation.
        """
        t_scaled = t * 1000.0
        
        half_dim = self.freq_dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t_scaled[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# MOONBEAM MRA ROPE (Grouped Heads)
# ---------------------------------------------------------------------------
class MRARotaryEmbedding(nn.Module):
    def __init__(self, head_dim, n_heads):
        """
        Partitions the n_heads across the 6 coordinate axes.
        Each head rotates its ENTIRE head_dim based on a single musical attribute.
        """
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_coords = 6
        
        # Sequence Position, Instrument, Pitch, Velocity, Onset, Duration
        bases = [10000.0, 1000.0, 1000.0, 500.0, 10000.0, 10000.0]
        
        self.inv_freqs = nn.ParameterList()
        for i in range(self.n_coords):
            base = bases[i]
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            self.inv_freqs.append(nn.Parameter(inv_freq, requires_grad=False))
            
        # Distribute heads evenly across the 6 coordinate dimensions
        self.head_to_coord = [i % self.n_coords for i in range(n_heads)]

    def forward(self, qk, coords):
        B, T, H, D = qk.shape
        cos_list, sin_list = [], []

        for h in range(H):
            coord_idx = self.head_to_coord[h]
            c = coords[:, :, coord_idx].float() 
            inv_freq = self.inv_freqs[coord_idx] 
            
            freqs = torch.einsum("bt,f->btf", c, inv_freq) 
            emb = torch.cat((freqs, freqs), dim=-1) 
            
            cos_list.append(emb.cos().unsqueeze(2)) # [B, T, 1, head_dim]
            sin_list.append(emb.sin().unsqueeze(2))

        cos = torch.cat(cos_list, dim=2).to(dtype=qk.dtype) # [B, T, n_heads, head_dim]
        sin = torch.cat(sin_list, dim=2).to(dtype=qk.dtype)

        return (qk * cos) + (rotate_half(qk) * sin)


class StandardGroupedQueryAttention(nn.Module):
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
        
        self.rope = MRARotaryEmbedding(self.head_dim, self.n_heads)

    def forward(self, x, coords, prefix_len=0, use_cache=False, kv_cache=None):
        B, T, C = x.shape
        
        # Flash Attention expects [B, T, n_heads, head_dim]. No transposing!
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # MRA Grouping Requirement: To apply mathematically independent RoPE coordinates 
        # to each Q head, K and V must be expanded to match n_heads locally. 
        # Flash Attention will seamlessly compute this as standard MHA.
        if self.n_kv_heads < self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)

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

        # --- SPLIT FLASH ATTENTION ROUTER ---
        if prefix_len > 0 and prefix_len < T:
            # Chunk 1: Prefix sees Prefix Causally
            q_prefix = q[:, :prefix_len]
            k_prefix = k[:, :prefix_len]
            v_prefix = v[:, :prefix_len]
            out_prefix = flash_attn_func(q_prefix, k_prefix, v_prefix, dropout_p=dropout_p, causal=True)
            
            # Chunk 2: Target Block sees Prefix + Block Bidirectionally
            q_block = q[:, prefix_len:]
            out_block = flash_attn_func(q_block, k, v, dropout_p=dropout_p, causal=False)
            
            attn_out = torch.cat([out_prefix, out_block], dim=1)
            
        elif prefix_len >= T:
            attn_out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=True)
            
        else:
            attn_out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=False)

        out = self.o_proj(attn_out.reshape(B, T, C))
        return out, new_kv_cache


class StandardTransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, n_kv_heads, ffn_mult, dropout):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.attn  = StandardGroupedQueryAttention(embed_dim, n_heads, n_kv_heads, dropout)
        self.ffn = SwiGLU(embed_dim, mult=ffn_mult, dropout=dropout)

    def forward(self, x, coords, prefix_len=0, use_cache=False, kv_cache=None):
        attn_out, new_kv_cache = self.attn(
            self.norm1(x), coords=coords, prefix_len=prefix_len,
            use_cache=use_cache, kv_cache=kv_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        
        if use_cache:
            return x, new_kv_cache
        return x


# ---------------------------------------------------------------------------
# SEQUENTIAL GRU DECODER
# ---------------------------------------------------------------------------
class SequentialGRUDecoder(nn.Module):
    def __init__(self, embed_dim, vocab_sizes):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.num_attrs = len(vocab_sizes)

        self.attr_emb_dim = embed_dim // self.num_attrs
        
        # Mirror the FME Embedder for Decoder Conditionings
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

    @torch.no_grad()
    def decode_sample(self, h_transformer, temperature=1.0, top_k=50):
        B, T, D = h_transformer.shape
        h_gru = h_transformer.reshape(B * T, D)

        sampled_attrs = []
        gru_in = self.start_emb.expand(B * T, -1)
        h_gru = self.gru(gru_in, h_gru)
        logits = self.out_heads[0](h_gru) / max(temperature, 1e-5)

        if top_k > 0:
            actual_top_k = min(top_k, logits.shape[-1])
            values, _ = torch.topk(logits, actual_top_k)
            logits = torch.where(logits < values[:, -1:].expand_as(logits), float('-inf'), logits)
        
        probs = F.softmax(logits, dim=-1)
        sampled_attrs.append(torch.multinomial(probs, num_samples=1))

        for i in range(1, self.num_attrs):
            prev_attr = sampled_attrs[-1].view(B * T)
            gru_in = self.embs[i-1](prev_attr)
            
            h_gru = self.gru(gru_in, h_gru)
            logits = self.out_heads[i](h_gru) / max(temperature, 1e-5)

            if top_k > 0:
                actual_top_k = min(top_k, logits.shape[-1])
                values, _ = torch.topk(logits, actual_top_k)
                logits = torch.where(logits < values[:, -1:].expand_as(logits), float('-inf'), logits)
                
            probs = F.softmax(logits, dim=-1)
            sampled_attrs.append(torch.multinomial(probs, num_samples=1))

        return torch.stack(sampled_attrs, dim=-1).view(B, T, 5)


# ---------------------------------------------------------------------------
# MAIN BLOCK-WISE GIDD DIFFUSION MODEL
# ---------------------------------------------------------------------------
class PrefixARModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.genre_emb = nn.Embedding(NUM_GENRES, EMBED_DIM)
        
        self.time_emb = TimestepEmbedder(EMBED_DIM)
        self.target_emb = CompoundInputEmbed(EMBED_DIM, VOCAB_SIZES)
        
        self.layers = nn.ModuleList([
            StandardTransformerBlock(
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
        self, target, genre, prefix_len=0, timestep=None, 
        use_cache=False, past_key_values=None, seq_offset=0
    ):
        device = target.device
        B, T = target.shape[0], target.shape[1]

        # 1. Timestep Conditioning
        if timestep is None:
            timestep = torch.zeros(B, device=device)
            
        t_emb = self.time_emb(timestep).unsqueeze(1) # [B, 1, D]
        
        # Inject the timestep dynamically into the active tokens via addition!
        e_target = self.target_emb(target) + t_emb 
        
        is_first_step = past_key_values is None
        
        if is_first_step:
            e_genre = self.genre_emb(genre).unsqueeze(1) 
            h = torch.cat([e_genre, e_target], dim=1) 
            
            coords_genre = torch.zeros((B, 1, 6), device=device, dtype=torch.long)
            seq_pos = torch.arange(seq_offset, seq_offset + T, device=device).unsqueeze(0).expand(B, -1).unsqueeze(-1)
            coords_target = torch.cat([seq_pos, target], dim=-1)
            coords = torch.cat([coords_genre, coords_target], dim=1)
            
            # The <Genre> token counts as part of the causal prefix
            actual_prefix_len = prefix_len + 1
        else:
            h = e_target
            seq_pos = torch.arange(seq_offset, seq_offset + T, device=device).unsqueeze(0).expand(B, -1).unsqueeze(-1)
            coords = torch.cat([seq_pos, target], dim=-1)
            
            # When caching Generation, the queries are strictly the block itself.
            actual_prefix_len = 0

        next_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            current_kv = past_key_values[i] if not is_first_step else None
            
            kwargs = {
                "coords": coords,
                "prefix_len": actual_prefix_len,
                "use_cache": use_cache,
                "kv_cache": current_kv
            }
            
            if USE_CKPT and self.training:
                layer_out = checkpoint(layer, h, use_reentrant=False, **kwargs)
            else:
                layer_out = layer(h, **kwargs)
            
            if use_cache:
                h, new_kv = layer_out
                next_key_values.append(new_kv)
            else:
                h = layer_out

        h = self.out_norm(h)

        if is_first_step:
            h_target = h[:, 1:, :]
        else:
            h_target = h

        if use_cache:
            return h_target, next_key_values
            
        return self.gru_decoder(h_target, target)


    # =========================================================================
    # INFERENCE HOOKS 
    # =========================================================================
    @torch.no_grad()
    def encode_prefix(self, prefix_target, genre):
        B, T = prefix_target.shape[0], prefix_target.shape[1]
        
        # Provide a static dummy timestep for the prefix tokens
        dummy_t = torch.zeros(B, device=prefix_target.device)
        
        _, past_key_values = self.forward(
            target=prefix_target, genre=genre, 
            prefix_len=T, timestep=dummy_t,
            use_cache=True, past_key_values=None
        )
        return past_key_values

    @torch.no_grad()
    def denoise_step(self, masked_block, genre, past_key_values, prefix_length, timestep_val):
        B = masked_block.shape[0]
        
        # Explicitly broadcast the current diffusion timestep to drive generation!
        t_tensor = torch.full((B,), timestep_val, device=masked_block.device)
        
        h_block, _ = self.forward(
            target=masked_block, genre=genre,
            prefix_len=0, timestep=t_tensor, 
            use_cache=True, past_key_values=past_key_values, seq_offset=prefix_length 
        )
        sampled_tokens = self.gru_decoder.decode_sample(h_block, temperature=1.0, top_k=50)
        return sampled_tokens