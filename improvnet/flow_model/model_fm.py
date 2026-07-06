import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from improvnet.flow_model.config_fm import MAX_LATENT_SEQ_LEN

# ---------------------------------------------------------------------------
# Flow Matching Utilities
# ---------------------------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AbsoluteSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, dim):
        super().__init__()
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]
    

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precomputes the complex frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(x, freqs_cis):
    """Applies the RoPE rotation to queries and keys."""
    # Cast to float32 for complex math (required by PyTorch AMP)
    x_float = x.float()
    x_complex = torch.view_as_complex(x_float.reshape(*x_float.shape[:-1], -1, 2))
    
    # Broadcast frequencies to match [Batch, Seq, Heads, Half_Head_Dim]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) 
    
    # Rotate and cast back to original shape and dtype (fp16/bf16)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).reshape_as(x)
    return x_out.type_as(x)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# class SDPA(nn.Module):
#     def __init__(self, dim, n_heads):
#         super().__init__()
#         self.n_heads = n_heads
#         self.head_dim = dim // n_heads
#         self.q_proj = nn.Linear(dim, dim, bias=False)
#         self.k_proj = nn.Linear(dim, dim, bias=False)
#         self.v_proj = nn.Linear(dim, dim, bias=False)
#         self.out_proj = nn.Linear(dim, dim, bias=False)

#     def forward(self, q_x, kv_x, mask=None):
#         B, T_q, _ = q_x.shape
#         _, T_k, _ = kv_x.shape

#         q = self.q_proj(q_x).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
#         k = self.k_proj(kv_x).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
#         v = self.v_proj(kv_x).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

#         if mask is not None:
#             # Your dataset mask uses True = Padding (Ignore).
#             # PyTorch SDPA boolean masks require True = Attend, False = Ignore.
#             # We invert the mask (~) and reshape to [B, 1, 1, T_k] to broadcast across heads and queries.
#             sdpa_mask = (~mask).unsqueeze(1).unsqueeze(2)
#         else:
#             sdpa_mask = None

#         # This single line forces FlashAttention or Memory-Efficient Attention!
#         y = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        
#         y = y.transpose(1, 2).contiguous().view(B, T_q, -1)
#         return self.out_proj(y)

class SDPA(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, q_x, kv_x, mask=None, freqs_cis=None):
        B, T_q, _ = q_x.shape
        _, T_k, _ = kv_x.shape

        # Keep them as [B, T, Heads, Head_Dim] for rotation
        q = self.q_proj(q_x).view(B, T_q, self.n_heads, self.head_dim)
        k = self.k_proj(kv_x).view(B, T_k, self.n_heads, self.head_dim)
        v = self.v_proj(kv_x).view(B, T_k, self.n_heads, self.head_dim)

        # --- THE SOTA UPGRADE: Apply RoPE if frequencies are provided ---
        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis[:T_q])
            k = apply_rotary_emb(k, freqs_cis[:T_k])

        # Now transpose for standard attention math
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            sdpa_mask = (~mask).unsqueeze(1).unsqueeze(2)
        else:
            sdpa_mask = None

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        
        y = y.transpose(1, 2).contiguous().view(B, T_q, -1)
        return self.out_proj(y)

# ---------------------------------------------------------------------------
# DiT Block (Self-Attention + Cross-Attention + AdaLN)
# ---------------------------------------------------------------------------
class FlowTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, ffn_mult=4.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = SDPA(hidden_dim, n_heads) # Replaced MHA
        
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cross_attn = SDPA(hidden_dim, n_heads) # Replaced MHA
        
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * ffn_mult)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * ffn_mult), hidden_dim)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, x, context, context_mask, c_emb, freqs_cis):
        shift_msa, scale_msa, shift_ca, scale_ca, shift_mlp, scale_mlp = self.adaLN_modulation(c_emb).chunk(6, dim=1)
        
        # 1. Self-Attention (Pass freqs_cis here to rotate the timeline)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + self.attn(x_norm, x_norm, mask=None, freqs_cis=freqs_cis)
        
        # 2. Cross-Attention (freqs_cis=None! Do NOT rotate the external context)
        x_norm2 = modulate(self.norm2(x), shift_ca, scale_ca)
        x = x + self.cross_attn(x_norm2, context, mask=context_mask, freqs_cis=None)
        
        # 3. FFN
        x_norm3 = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + self.ffn(x_norm3)
        
        return x

# ---------------------------------------------------------------------------
# Master Flow Matching Model
# ---------------------------------------------------------------------------
class FlowMatchingModel(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=1024, num_layers=16, num_heads=16, num_inst_classes=40):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Input Projection
        self.x_proj = nn.Linear(latent_dim, hidden_dim)
        self.context_proj = nn.Linear(latent_dim, hidden_dim)

        # Positional Embeddings (Absolute Sinusoidal)
        self.abs_pos_embed = AbsoluteSinusoidalPositionalEmbedding(MAX_LATENT_SEQ_LEN, hidden_dim)

        # Precompute RoPE frequencies
        head_dim = hidden_dim // num_heads
        freqs_cis = precompute_freqs_cis(head_dim, MAX_LATENT_SEQ_LEN)
        self.register_buffer("freqs_cis", freqs_cis)
        
        # Time & Instrument Embeddings
        self.time_embed = SinusoidalPositionEmbeddings(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.inst_proj = nn.Linear(num_inst_classes, hidden_dim)
        
        # --- LEARNED PREFIXES & NULL EMBEDDINGS (For CFG) ---
        # Prefixes to separate the dynamic segments
        self.melody_prefix = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.harmony_prefix = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.rhythm_prefix = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Null embeddings used when a condition is "dropped" via CFG
        self.null_melody = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.null_harmony = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.null_rhythm = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.null_inst = nn.Parameter(torch.randn(1, hidden_dim))

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            FlowTransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Output Projection (Predicting the Vector Field)
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize output to zero for stable ODE start
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_t, time, z_mel, mel_mask, z_har, har_mask, z_rhy, rhy_mask, inst_multihot, cfg_drops):
        """
        z_t: Noisy target latents [B, Seq, 128]
        z_mel, z_har, z_rhy: Condition latents [B, Cond_Seq, 128]
        cfg_drops: Dictionary of booleans dictating which conditions to drop for this batch
        """
        B = z_t.shape[0]
        
        # 1. Embed Time and Instruments
        t_emb = self.time_mlp(self.time_embed(time))
        
        # CFG Drop for Instruments
        if cfg_drops.get("inst", False):
            inst_emb = self.null_inst.expand(B, -1)
        else:
            inst_emb = self.inst_proj(inst_multihot)
            
        c_emb = t_emb + inst_emb # Combined conditioning for AdaLN
        
        # 2. Prepare Context Sequence (Melody, Harmony, Rhythm)
        contexts = []
        masks = [] # True means IGNORE in PyTorch MultiheadAttention
        
        # Helper to process each segment
        def process_segment(z_cond, cond_mask, prefix, null_emb, is_dropped):
            if is_dropped:
                # If dropped, context is just the single Null token
                ctx = null_emb.expand(B, -1, -1)
                msk = torch.zeros((B, 1), dtype=torch.bool, device=z_t.device) # Don't ignore null token
            else:
                # Project latents, prepend Prefix
                ctx_proj = self.context_proj(z_cond)
                pref = prefix.expand(B, -1, -1)
                ctx = torch.cat([pref, ctx_proj], dim=1)
                # Prefix mask is False (pay attention to it), concat with rest
                pref_msk = torch.zeros((B, 1), dtype=torch.bool, device=z_t.device)
                msk = torch.cat([pref_msk, cond_mask], dim=1)
            return ctx, msk

        # Process Melody
        m_ctx, m_msk = process_segment(z_mel, mel_mask, self.melody_prefix, self.null_melody, cfg_drops.get("melody", False))
        contexts.append(m_ctx); masks.append(m_msk)
        
        # Process Harmony
        h_ctx, h_msk = process_segment(z_har, har_mask, self.harmony_prefix, self.null_harmony, cfg_drops.get("harmony", False))
        contexts.append(h_ctx); masks.append(h_msk)
        
        # Process Rhythm
        r_ctx, r_msk = process_segment(z_rhy, rhy_mask, self.rhythm_prefix, self.null_rhythm, cfg_drops.get("rhythm", False))
        contexts.append(r_ctx); masks.append(r_msk)
        
        # Concatenate everything into one super-sequence
        full_context = torch.cat(contexts, dim=1)
        full_mask = torch.cat(masks, dim=1)
        
        # 3. Main Transformer Pass
        x = self.x_proj(z_t)

        x = self.abs_pos_embed(x)

        for block in self.blocks:
            if self.training:
                x = checkpoint(block, x, full_context, full_mask, c_emb, self.freqs_cis, use_reentrant=False)
            else:
                x = block(x, full_context, full_mask, c_emb, self.freqs_cis)
            
        # 4. Output Vector Field
        # We need shift/scale for the final norm too
        shift_final, scale_final, _, _, _, _ = block.adaLN_modulation(c_emb).chunk(6, dim=1) 
        x = modulate(self.final_norm(x), shift_final, scale_final)
        
        v_pred = self.out_proj(x)
        return v_pred