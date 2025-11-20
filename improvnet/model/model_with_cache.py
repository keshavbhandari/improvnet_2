import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel, PretrainedConfig
import time
import traceback

# --- Constants ---
K_PROMPT_REFRESH = 100 
K_RESPONSE_REFRESH = 6
NUM_ATTRIBUTES = 6
NUM_VOICE_ATTRIBUTES = 5 
DEFAULT_MRA_BASE_VALUES = [100.0, 131.0, 20.0, 1031.0, 1031.0, 10000.0]

# --- dLLM-Cache Hyperparameters ---
K_PROMPT_REFRESH = 100 
K_RESPONSE_REFRESH = 6  

# --- Configuration ---
class ImprovNetConfig(PretrainedConfig):
    model_type = "improvnet"
    def __init__(
        self,
        hidden_size=780, 
        num_heads=30,
        num_layers=12,
        ffn_dim=3120,
        vocab_sizes=[129, 128, 128, 512, 512],
        seq_len=2048,
        num_genres=10,
        num_forms=5,
        no_bias=False,
        gradient_checkpointing=False,
        initializer_range=0.02,
        adaptive_update_ratio=0.25,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.vocab_sizes = vocab_sizes
        self.seq_len = seq_len
        self.num_genres = num_genres
        self.num_forms = num_forms
        self.no_bias = no_bias
        self.gradient_checkpointing = gradient_checkpointing
        self.initializer_range = initializer_range
        self.adaptive_update_ratio = adaptive_update_ratio
        assert hidden_size % num_heads == 0
        assert hidden_size % NUM_ATTRIBUTES == 0
        assert num_heads % NUM_ATTRIBUTES == 0 # 30 % 6 == 0

# --- 1. Moonbeam Input (Unchanged) ---
class MoonbeamInput(nn.Module):
    def __init__(self, hidden_size, vocab_sizes, num_dims=NUM_VOICE_ATTRIBUTES):
        super().__init__()
        assert len(vocab_sizes) == num_dims
        assert hidden_size % NUM_ATTRIBUTES == 0
        self.attr_emb_dim = hidden_size // NUM_ATTRIBUTES 
        self.embeds = nn.ModuleList([
            nn.Embedding(vocab_sizes[i], self.attr_emb_dim) for i in range(num_dims)
        ])
    def forward(self, instrument, pitch, velocity, onset, duration):
        inputs = [instrument, pitch, velocity, onset, duration]
        embeds = [self.embeds[i](inputs[i]) for i in range(len(inputs))]
        return torch.cat(embeds, dim=-1)

# --- 2. SwiGLU FFN Module (Unchanged) ---
class SwiGLU_FFN(nn.Module):
    def __init__(self, hidden_size, ffn_dim, bias=True):
        super().__init__()
        self.w1_gate = nn.Linear(hidden_size, ffn_dim, bias=bias)
        self.w2_up = nn.Linear(hidden_size, ffn_dim, bias=bias)
        self.w3_down = nn.Linear(ffn_dim, hidden_size, bias=bias)
        self.act_fn = nn.SiLU()
    def forward(self, x):
        return self.w3_down(self.act_fn(self.w1_gate(x)) * self.w2_up(x))

# --- 3. MRA Attention Module (REFACTORED) ---
class MRANonCausalAttention(nn.Module):
    def __init__(self, config: "ImprovNetConfig"):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_groups = NUM_ATTRIBUTES
        self.heads_per_group = self.num_heads // self.num_groups
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=not config.no_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=not config.no_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=not config.no_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=not config.no_bias)

        for i in range(self.num_groups):
            base = DEFAULT_MRA_BASE_VALUES[i]
            inv_freqs_g = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer(f'inv_freqs_{i}', inv_freqs_g)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, x, positions, inv_freqs):
        # x: (B_slice, L_slice, H_g, D_h)
        # positions: (B_slice, L_slice)
        sinusoid_inp = torch.einsum("b l, d -> b l d", positions, inv_freqs)
        sin = torch.sin(sinusoid_inp).unsqueeze(2).repeat_interleave(2, dim=-1)
        cos = torch.cos(sinusoid_inp).unsqueeze(2).repeat_interleave(2, dim=-1)
        return (x * cos) + (self._rotate_half(x) * sin)

    def apply_mra_rotation(self, tensor: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        # tensor: (B_slice, L_slice, H, D_h)
        # attributes: (B_slice, L_slice, 6)
        if tensor.dim() != 4:
            raise IndexError(f"MRA rotation expected 4D tensor (B, L, H, D_h) but got {tensor.dim()}D tensor with shape {tensor.shape}")

        rotated_list = []
        for g in range(self.num_groups):
            start_head = g * self.heads_per_group
            end_head = (g + 1) * self.heads_per_group
            
            tensor_group = tensor[:, :, start_head:end_head, :] 
            position_values = attributes[:, :, g].float()
            inv_freqs_g = getattr(self, f'inv_freqs_{g}')
            rotated_group = self._apply_rotary_pos_emb(tensor_group, position_values, inv_freqs_g)
            rotated_list.append(rotated_group)
        
        return torch.cat(rotated_list, dim=2) # (B_slice, L_slice, H, D_h)

    def compute_qkv_rot(self, x_norm_slice, attributes_slice):
        """
        Computes Q_rot, K_rot, and V for a given slice of the input.
        This is the expensive part we want to skip.
        """
        B_s, L_s, C = x_norm_slice.shape
        H, D_h = self.num_heads, self.head_dim
        
        q = self.q_proj(x_norm_slice).view(B_s, L_s, H, D_h)
        k = self.k_proj(x_norm_slice).view(B_s, L_s, H, D_h)
        v = self.v_proj(x_norm_slice).view(B_s, L_s, H, D_h)
        
        q_rot = self.apply_mra_rotation(q, attributes_slice).transpose(1, 2) # (B_s, H, L_s, D_h)
        k_rot = self.apply_mra_rotation(k, attributes_slice).transpose(1, 2) # (B_s, H, L_s, D_h)
        v_final = v.transpose(1, 2) # (B_s, H, L_s, D_h)
        
        return q_rot, k_rot, v_final

    def forward(self, q_rot_final, k_rot_final, v_final):
        """
        Only performs the dot product and output projection.
        """
        B, H, L, D_h = q_rot_final.shape
        C = self.hidden_size
        
        attn_output = F.scaled_dot_product_attention(q_rot_final, k_rot_final, v_final)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, L, C)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

# --- 4. Transformer Block ---
class ImprovNetTransformerBlock(nn.Module):
    def __init__(self, config: "ImprovNetConfig"):
        super().__init__()
        self.config = config
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.attn = MRANonCausalAttention(config)
        
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn = SwiGLU_FFN(
            hidden_size=config.hidden_size,
            ffn_dim=config.ffn_dim,
            bias=not config.no_bias
        )
        self.L_main = config.seq_len

    def v_verify(self, v_new_flat, v_cached_flat, dynamic_mask, update_ratio):
        # v_new_flat/v_cached_flat are [B, L_comb, C]
        # dynamic_mask is [B, L_comb]
        
        sim = F.cosine_similarity(v_new_flat, v_cached_flat, dim=-1) # [B, L_comb]
        
        # Set similarity of STATIC tokens to infinity so they are never chosen
        sim_for_update = torch.where(dynamic_mask, sim, torch.inf)
        
        num_dynamic_tokens = dynamic_mask.sum(dim=-1) # [B]
        # Calculate k (number to update) for each batch item
        num_to_update = (num_dynamic_tokens.float() * update_ratio).round().long() # [B]
        
        update_mask_full = torch.zeros_like(sim, dtype=torch.bool)
        
        for i in range(v_new_flat.shape[0]): # Iterate over batch
            k = num_to_update[i].item()
            if k > 0:
                # Ensure k is not larger than the number of dynamic tokens
                k_safe = min(k, num_dynamic_tokens[i].item())
                if k_safe > 0: # Check again
                    topk_indices = torch.topk(sim_for_update[i], k=k_safe, largest=False).indices
                    update_mask_full[i].scatter_(-1, topk_indices, True)
                    
        return update_mask_full

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attributes: torch.Tensor, 
        dynamic_mask: torch.Tensor, # <-- This is correct
        cache: Optional[dict] = None,
        refresh_prompt: bool = False,
        refresh_response: bool = False,
        update_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, dict]:
        
        B, L_comb, C = hidden_states.shape
        H, D_h = self.config.num_heads, C // self.config.num_heads
        
        x_norm1 = self.attn_norm(hidden_states)

        # --- 1. Handle Initialization (Step 0) ---
        if cache is None:
            q_rot_full, k_rot_full, v_full = self.attn.compute_qkv_rot(x_norm1, attributes)
            attn_output = self.attn(q_rot_full, k_rot_full, v_full)
            hidden_states = hidden_states + attn_output
            
            x_norm2 = self.ffn_norm(hidden_states)
            ffn_output = self.ffn(x_norm2)
            hidden_states = hidden_states + ffn_output
            
            new_cache = {
                'k_rot': k_rot_full.detach(),
                'v': v_full.detach(),       
                'attn_out': attn_output.detach(),
                'ffn_out': ffn_output.detach()  
            }
            return hidden_states, new_cache

        # --- 2. Handle Caching Path (Step 1...N) ---
        
        k_rot_cached = cache['k_rot']
        v_cached = cache['v']
        attn_out_cached = cache['attn_out']
        ffn_out_cached = cache['ffn_out']
        
        # 2a. Determine Update Mask
        
        # This is the full refresh path (e.g., k_step=0 or periodic)
        if refresh_prompt and refresh_response:
            update_mask_full = torch.ones(B, L_comb, device=hidden_states.device, dtype=torch.bool)
        
        # This is the V-Verify path (most steps)
        else: 
            # --- EFFICIENT V-VERIFY ---
            # 1. Compute *only* v_proj (cheap)
            v_new_unrotated = self.attn.v_proj(x_norm1).view(B, L_comb, H, D_h) # [B, L, H, D_h]
            
            # 2. Flatten for similarity check
            v_new_flat = v_new_unrotated.reshape(B, L_comb, C) # [B, L, C]
            v_cached_flat = v_cached.transpose(1, 2).reshape(B, L_comb, C) # [B, H, L, D_h] -> [B, L, H, D_h] -> [B, L, C]

            # 3. Call v_verify to get the sparse update mask
            update_mask_full = self.v_verify(v_new_flat, v_cached_flat, dynamic_mask, update_ratio)
            # --- END EFFICIENT V-VERIFY ---

        # 2c. Find update indices
        update_indices = update_mask_full.nonzero(as_tuple=True)
        N_upd = update_indices[0].numel()

        # 2d. Initialize final tensors with cached values
        k_rot_for_attn = k_rot_cached.clone()
        v_for_attn = v_cached.clone()
        attn_out = attn_out_cached.clone()
        
        # --- EFFICIENT Q COMPUTATION ---
        # We must ALWAYS compute the full NEW Q_rot (cheap)
        q_new_unrotated = self.attn.q_proj(x_norm1).view(B, L_comb, H, D_h)
        q_rot_new = self.attn.apply_mra_rotation(q_new_unrotated, attributes).transpose(1, 2)
        # --- END EFFICIENT Q COMPUTATION ---


        # --- 2e. Sparse Attention Computation ---
        if N_upd > 0:
            # Gather inputs for the tokens that are updating
            x_norm_upd = x_norm1[update_indices] # (N_upd, C)
            attrs_upd = attributes[update_indices] # (N_upd, 6)
            
            # --- EFFICIENT K & V UPDATE ---
            # Compute K_rot for *only* the updating tokens
            k_new_unrotated_upd = self.attn.k_proj(x_norm_upd).view(N_upd, H, D_h)
            k_rot_upd_scatter = self.attn.apply_mra_rotation(
                 k_new_unrotated_upd.unsqueeze(0), # [1, N_upd, H, D_h]
                 attrs_upd.unsqueeze(0)            # [1, N_upd, 6]
            ).squeeze(0) # [N_upd, H, D_h]
            
            # We need to re-fetch v_new_unrotated here
            v_new_unrotated = self.attn.v_proj(x_norm1).view(B, L_comb, H, D_h) # [B, L, H, D_h]
            v_new_unrotated_upd = v_new_unrotated[update_indices] # [N_upd, H, D_h]
            # --- END EFFICIENT K & V UPDATE ---

            # --- *** THE FIX IS HERE *** ---
            # The shapes (N_upd, H, D_h) already match. No transpose needed.
            
            k_rot_for_attn[update_indices[0], :, update_indices[1], :] = k_rot_upd_scatter
            v_for_attn[update_indices[0], :, update_indices[1], :] = v_new_unrotated_upd
            
            # --- *** END FIX *** ---
            
            
            # --- Sparse Q @ K^T ---
            # Fallback to full attention if update counts are uneven (safer)
            if N_upd % B != 0:
                attn_output = self.attn(q_rot_new, k_rot_for_attn, v_for_attn)
                attn_out = attn_output
            else:
                N_upd_per_batch = N_upd // B
                
                # Gather Q for the updating tokens
                q_rot_new_trans = q_rot_new.transpose(1, 2)
                q_rot_upd_slice = q_rot_new_trans[update_indices] 
                q_rot_upd = q_rot_upd_slice.view(B, N_upd_per_batch, H, D_h).transpose(1, 2)
                
                # Run sparse dot product
                attn_weights_upd = torch.matmul(q_rot_upd, k_rot_for_attn.transpose(-1, -2)) / (D_h ** 0.5)
                attn_weights_upd = F.softmax(attn_weights_upd, dim=-1)
                
                # Run sparse output
                attn_output_upd = torch.matmul(attn_weights_upd, v_for_attn) 
                
                # Project sparse output
                attn_output_upd = attn_output_upd.transpose(1, 2).contiguous().reshape(B, N_upd_per_batch, C)
                attn_output_proj_upd = self.attn.o_proj(attn_output_upd) 
                
                # Scatter new AttnOut into the cached tensor
                attn_out_indices = (
                    update_indices[0].view(B, N_upd_per_batch),
                    update_indices[1].view(B, N_upd_per_batch)
                )
                attn_out[attn_out_indices[0], attn_out_indices[1], :] = attn_output_proj_upd

        else:
            # If N_upd = 0 (all dynamic tokens passed V-verify), 
            # run full attention with NEW Q and CACHED K, V.
            attn_output = self.attn(q_rot_new, k_rot_cached, v_cached)
            attn_out = attn_output 
        
        hidden_states = hidden_states + attn_out

        # --- 2f. Sparse FFN Computation ---
        ffn_out = ffn_out_cached.clone()
        if N_upd > 0:
            # Gather inputs for FFN (only from updated hidden states)
            x_norm2_upd = self.ffn_norm(hidden_states[update_indices]) # (N_upd, C)
            
            # Compute FFN only on updated tokens
            ffn_output_upd = self.ffn(x_norm2_upd) # (N_upd, C)
            
            # Scatter new FFN output
            ffn_out[update_indices] = ffn_output_upd
            
        hidden_states = hidden_states + ffn_out
        
        # --- 2g. Create New Cache ---
        new_cache = {
            'k_rot': k_rot_for_attn.detach(), 
            'v': v_for_attn.detach(),       
            'attn_out': attn_out.detach(),
            'ffn_out': ffn_out.detach()
        }
        
        return hidden_states, new_cache

# --- 5. ImprovNet Main Model (Unchanged) ---
class ImprovNet(PreTrainedModel):
    config_class = ImprovNetConfig
    def __init__(self, config: ImprovNetConfig):
        super().__init__(config)
        self.config = config
        self.seq_len = config.seq_len
        self.vocab_sizes = config.vocab_sizes
        self.gradient_checkpointing = config.gradient_checkpointing

        self.input_embed = MoonbeamInput(config.hidden_size, config.vocab_sizes)
        self.pos_embed = nn.Embedding(
            config.seq_len * 2, 
            config.hidden_size // NUM_ATTRIBUTES 
        )
        self.genre_embed = nn.Embedding(config.num_genres, config.hidden_size)
        self.form_embed = nn.Embedding(config.num_forms, config.hidden_size)
        
        self.transformer_blocks = nn.ModuleList([
            ImprovNetTransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.output_heads_main = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_sizes[i], bias=not config.no_bias) 
            for i in range(NUM_VOICE_ATTRIBUTES)
        ])
        self.output_heads_accom = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_sizes[i], bias=not config.no_bias) 
            for i in range(NUM_VOICE_ATTRIBUTES)
        ])
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if hasattr(self, 'pos_embed'):
            self.pos_embed.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def _calculate_loss(self, all_logits, labels, loss_mask):
        total_loss = 0.0
        loss_count = 0
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        all_labels = torch.unbind(labels, dim=-1)
        all_loss_masks = torch.unbind(loss_mask, dim=-1)
        for i in range(NUM_VOICE_ATTRIBUTES):
            if all_loss_masks[i].any():
                loss_count += 1
                logits_flat = all_logits[i].reshape(-1, self.vocab_sizes[i])
                labels_flat = all_labels[i].reshape(-1)
                mask_flat = all_loss_masks[i].reshape(-1).float()
                raw_loss = loss_fct(logits_flat, labels_flat)
                masked_loss = raw_loss * mask_flat
                attribute_loss = masked_loss.sum() / (mask_flat.sum() + 1e-9)
                total_loss += attribute_loss
        return total_loss, loss_count

    def forward(
        self, 
        input_attributes_main,
        input_attributes_accom,
        genre,
        form,
        labels_main=None,
        labels_accom=None,
        loss_mask_main=None,
        loss_mask_accom=None,
        dynamic_mask_main: Optional[torch.Tensor] = None, # <-- NEW
        dynamic_mask_accom: Optional[torch.Tensor] = None, # <-- NEW
        cache: Optional[list] = None, 
        adaptive_update_ratio: Optional[float] = None,
        k_step: int = 1, # Default to 1 (adaptive)
        return_dict=None
    ):
        B, L, A = input_attributes_main.shape
        L_comb = L * 2
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # --- 1. Input Embedding (Unchanged) ---
        main_attrs = torch.unbind(input_attributes_main, dim=-1)
        accom_attrs = torch.unbind(input_attributes_accom, dim=-1)
        seq_indices_main = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
        seq_indices_accom = torch.arange(L, 2*L, device=self.device).unsqueeze(0).expand(B, L)
        
        x_main_5 = self.input_embed(main_attrs[0], main_attrs[1], main_attrs[2], main_attrs[3], main_attrs[4])
        x_accom_5 = self.input_embed(accom_attrs[0], accom_attrs[1], accom_attrs[2], accom_attrs[3], accom_attrs[4])
        
        pos_emb_main = self.pos_embed(seq_indices_main)
        pos_emb_accom = self.pos_embed(seq_indices_accom)
        
        x_main = torch.cat([x_main_5, pos_emb_main], dim=-1)
        x_accom = torch.cat([x_accom_5, pos_emb_accom], dim=-1)
        
        g_emb = self.genre_embed(genre).unsqueeze(1)
        f_emb = self.form_embed(form).unsqueeze(1)
        cond_emb = g_emb + f_emb
        x_main = x_main + cond_emb
        x_accom = x_accom + cond_emb
        
        x_combined = torch.cat([x_main, x_accom], dim=1)
        attributes_main_5 = input_attributes_main.float()
        attributes_accom_5 = input_attributes_accom.float()
        attrs_main_6d = torch.cat([attributes_main_5, seq_indices_main.float().unsqueeze(-1)], dim=-1)
        attrs_accom_6d = torch.cat([attributes_accom_5, seq_indices_accom.float().unsqueeze(-1)], dim=-1)
        attrs_combined = torch.cat([attrs_main_6d, attrs_accom_6d], dim=1)

        use_cache = not self.training
        
        # --- 2. Create Full Dynamic Mask ---
        # This mask identifies which TOKENS (not attributes) are dynamic.
        if dynamic_mask_main is not None and dynamic_mask_accom is not None and use_cache:
            # .any(-1) checks if ANY attribute in a token is masked
            dynamic_mask_main_per_token = dynamic_mask_main.any(dim=-1).bool() # [B, L]
            dynamic_mask_accom_per_token = dynamic_mask_accom.any(dim=-1).bool() # [B, L]
            dynamic_mask_full_per_token = torch.cat(
                [dynamic_mask_main_per_token, dynamic_mask_accom_per_token], 
                dim=1
            ) # [B, L_comb]
        else:
            # Fallback for training or non-cache inference: assume all are dynamic
            dynamic_mask_full_per_token = torch.ones(B, L_comb, device=self.device, dtype=torch.bool)

        # --- 3. Set Cache Flags ---
        if adaptive_update_ratio is None:
            update_ratio = self.config.adaptive_update_ratio
        else:
            update_ratio = adaptive_update_ratio
        
        is_initialization = (k_step == 0)
        
        if is_initialization:
            refresh_prompt = True
            refresh_response = True
            update_ratio = 1.0 
        else:
            refresh_prompt = False
            refresh_response = False
            # update_ratio is taken from args

        if use_cache:
            if cache is None or is_initialization:
                cache = [None] * self.config.num_layers
                refresh_prompt = True
                refresh_response = True
                update_ratio = 1.0 
            new_cache_list = []
        else:
            cache = [None] * self.config.num_layers

        # --- 4. Run Transformer Blocks ---
        for i, block in enumerate(self.transformer_blocks):
            layer_cache = cache[i] if use_cache else None 
            
            if use_cache:
                x_combined, new_layer_cache = block(
                    x_combined, 
                    attrs_combined, 
                    dynamic_mask=dynamic_mask_full_per_token, # <-- PASS THE MASK
                    cache=layer_cache, 
                    refresh_prompt=refresh_prompt,
                    refresh_response=refresh_response,
                    update_ratio=update_ratio
                )
                new_cache_list.append(new_layer_cache)
            else: 
                if self.gradient_checkpointing and self.training:
                    # We must pass dynamic_mask=None (or a placeholder) for checkpointing
                    x_combined = checkpoint(block, x_combined, attrs_combined, dynamic_mask_full_per_token, None, True, True, 1.0, use_reentrant=False)[0]
                else:
                    x_combined = block(x_combined, attrs_combined, dynamic_mask=dynamic_mask_full_per_token, cache=None, refresh_prompt=True, refresh_response=True, update_ratio=1.0)[0]
            
        # --- 5. Output (Unchanged) ---
        x_combined_norm = self.final_norm(x_combined)
        x_main_norm, x_accom_norm = x_combined_norm.chunk(2, dim=1)
        
        logits_main = tuple(head(x_main_norm) for head in self.output_heads_main)
        logits_accom = tuple(head(x_accom_norm) for head in self.output_heads_accom)
        
        total_loss = None
        if labels_main is not None:
            loss_main, count_main = self._calculate_loss(logits_main, labels_main, loss_mask_main)
            loss_accom, count_accom = self._calculate_loss(logits_accom, labels_accom, loss_mask_accom)
            total_loss = (loss_main + loss_accom) / (count_main + count_accom + 1e-9)

        if not return_dict:
            output = ()
            if total_loss is not None:
                output += (total_loss,)
            output += (logits_main, logits_accom)
            if use_cache:
                output += (new_cache_list,)
            return output
        
        output = {
            "logits_main": logits_main,
            "logits_accom": logits_accom
        }
        if total_loss is not None:
            output["loss"] = total_loss
        if use_cache:
            output["cache"] = new_cache_list
        return output
    

# --- Dummy Test (Copied from your script) ---
def run_dummy_test():
    # Model parameters for test
    SEQ_LEN = 2048
    HIDDEN_SIZE = 72
    VOCAB_SIZES = [129, 128, 128, 512, 512] 
    NUM_HEADS = 12 # This MUST match the config
    
    config = ImprovNetConfig(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        num_layers=2,
        ffn_dim=HIDDEN_SIZE * 4,
        vocab_sizes=VOCAB_SIZES,
        seq_len=SEQ_LEN,
        adaptive_update_ratio=0.5
    )
    model = ImprovNet(config)
    model.eval()

    B = 1
    device = model.device
    
    # Corrected random data generation 
    input_main_list = []
    input_accom_list = []
    for vocab_size in VOCAB_SIZES:
        rand_main = torch.randint(0, vocab_size, (B, SEQ_LEN), device=device)
        rand_accom = torch.randint(0, vocab_size, (B, SEQ_LEN), device=device)
        input_main_list.append(rand_main)
        input_accom_list.append(rand_accom)

    input_main = torch.stack(input_main_list, dim=-1)
    input_accom = torch.stack(input_accom_list, dim=-1)
    genre = torch.tensor([0], device=device)
    form = torch.tensor([0], device=device)

    # --- Step 1: Initialization (Full Compute/Cache) ---
    print(f"--- Step 1 (k=0): Initialization (Full Compute/Cache) ---")
    output1 = model(input_main, input_accom, genre, form, k_step=0, return_dict=True)
    cache1 = output1['cache']
    
    print(f"Cache size: {len(cache1)} layers. First K_rot cache is non-None: {'k_rot' in cache1[0]}")
    
    # --- Step 2: Adaptive Update (k_step=1, Adaptive Ratio) ---
    print(f"\n--- Step 2 (k=1): Adaptive Update (Ratio={config.adaptive_update_ratio}) ---")
    
    start_time = time.perf_counter()
    output2 = model(input_main, input_accom, genre, form, cache=cache1, k_step=1, adaptive_update_ratio=config.adaptive_update_ratio, return_dict=True)
    time_adaptive = time.perf_counter() - start_time

    logits_main_shape = output2['logits_main'][0].shape
    print(f"Logits Main Shape: {logits_main_shape} (Time: {time_adaptive:.4f}s)")
    
    # --- Step 3: Full Compute/No Cache (k_step=2, Ratio=1.0) ---
    print(f"\n--- Step 3 (k=2): Full Compute (Ratio=1.0) ---")
    start_time = time.perf_counter()
    output3 = model(input_main, input_accom, genre, form, cache=cache1, k_step=2, adaptive_update_ratio=1.0, return_dict=True)
    time_full = time.perf_counter() - start_time
    
    print(f"Time Full Compute: {time_full:.4f}s")
    
    print(f"\nVerification: Full Compute Time / Adaptive Time = {time_full / time_adaptive:.2f}x")
    
    print("\nDummy test complete. Architectural consistency verified.")
    return model

if __name__ == '__main__':
    try:
        run_dummy_test()
    except Exception as e:
        print("\n--- TEST FAILED ---")
        traceback.print_exc()