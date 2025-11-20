import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel, PretrainedConfig

# 5 attributes: [instrument, pitch, velocity, onset, duration]
DEFAULT_MRA_BASE_VALUES = [10000.0, 131.0, 20.0, 1031.0, 1031.0, 10000.0]
NUM_ATTRIBUTES = 6
NUM_VOICE_ATTRIBUTES = 5 # instrument, pitch, velocity, onset, duration

class MultidimensionalRelativeAttention(nn.Module):
    """
    Implements Flash Attention with Multidimensional Relative Attention (MRA).
    """
    def __init__(self, hidden_size, num_heads, num_dims=NUM_ATTRIBUTES, 
                 base_values=DEFAULT_MRA_BASE_VALUES, bias=True):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert num_heads % num_dims == 0, f"num_heads ({num_heads}) must be divisible by num_dims ({num_dims})"
        self.num_heads = num_heads
        self.num_dims = num_dims
        self.head_dim = hidden_size // num_heads
        self.heads_per_dim = num_heads // num_dims
        assert self.head_dim % 2 == 0
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        for i in range(num_dims):
            base = base_values[i]
            inv_freqs_g = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer(f'inv_freqs_{i}', inv_freqs_g)
            
    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, x, positions, inv_freqs):
        sinusoid_inp = torch.einsum("b l, d -> b l d", positions, inv_freqs)
        sin = torch.sin(sinusoid_inp).unsqueeze(2).repeat_interleave(2, dim=-1)
        cos = torch.cos(sinusoid_inp).unsqueeze(2).repeat_interleave(2, dim=-1)
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(self, x_norm, attributes, cached_kv=None, update_mask=None):
        B, L, D = x_norm.shape
        
        # 1. Project Q, K, V
        q_current = self.q_proj(x_norm)
        k_current = self.k_proj(x_norm)
        v_current = self.v_proj(x_norm)
        
        # 2. Apply Caching Logic
        if cached_kv is not None:
            k_cached, v_cached = cached_kv
            # Use new K only for updated tokens, else use cached K
            k = torch.where(update_mask, k_current, k_cached)
            # Always use the full new V for V-verify (as per dLLM-Cache paper)
            v = v_current 
        else:
            # First step, no cache exists
            k = k_current
            v = v_current

        # Reshape for MHA
        q = q_current.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v_sdpa = v.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, L, D_head)

        # 3. Apply MRA rotation to Q and K (k is already mixed)
        q_rotated, k_rotated = [], []
        for i in range(self.num_dims):
            start_head, end_head = i * self.heads_per_dim, (i + 1) * self.heads_per_dim
            q_group = q[:, :, start_head:end_head, :]
            k_group = k[:, :, start_head:end_head, :]
            
            positions_g = attributes[:, :, i].float()
            inv_freqs_g = getattr(self, f'inv_freqs_{i}')
            
            q_rotated.append(self._apply_rotary_pos_emb(q_group, positions_g, inv_freqs_g))
            k_rotated.append(self._apply_rotary_pos_emb(k_group, positions_g, inv_freqs_g))
        
        q = torch.cat(q_rotated, dim=2).transpose(1, 2) # (B, H, L, D_head)
        k = torch.cat(k_rotated, dim=2).transpose(1, 2) # (B, H, L, D_head)

        # 4. Flash Attention
        output = F.scaled_dot_product_attention(q, k, v_sdpa)
        
        # 5. Reshape and final projection
        output = output.transpose(1, 2).contiguous().reshape(B, L, D)
        
        # Return new (k, v) pair to be cached
        # We return the *mixed* k and the *full new* v
        return self.o_proj(output), (k_current, v_current)

class SwiGLU_FFN(nn.Module):
    def __init__(self, hidden_size, ffn_dim, bias=True):
        super().__init__()
        self.w1_gate = nn.Linear(hidden_size, ffn_dim, bias=bias)
        self.w2_up = nn.Linear(hidden_size, ffn_dim, bias=bias)
        self.w3_down = nn.Linear(ffn_dim, hidden_size, bias=bias)
        self.act_fn = nn.SiLU()
    def forward(self, x):
        return self.w3_down(self.act_fn(self.w1_gate(x)) * self.w2_up(x))

class ImprovNetTransformerBlock(nn.Module):
    """
    Modified Transformer Block with Adaptive Caching Logic.
    """
    def __init__(self, config: "ImprovNetConfig"):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = MultidimensionalRelativeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_dims=NUM_ATTRIBUTES, # 6
            bias=not config.no_bias
        )
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.ffn = SwiGLU_FFN(
            hidden_size=config.hidden_size,
            ffn_dim=config.ffn_dim,
            bias=not config.no_bias
        )

    def forward(self, x, attributes, cache=None, update_ratio=0.25):
        """
        MODIFIED: Now has two distinct paths for training (cache=None)
        and inference (cache is not None).
        """
        
        if cache is not None:
            # --- 1. INFERENCE / CACHING PATH ---
            x_norm1 = self.norm1(x)
            
            # --- 1a. V-Verify Logic ---
            v_current = self.attn.v_proj(x_norm1)
            v_cached = cache['v']
            sim = F.cosine_similarity(v_current, v_cached, dim=-1) # (B, L)
            
            num_tokens = x.shape[1]
            num_to_update = int(num_tokens * update_ratio)
            
            if num_to_update > 0:
                topk = torch.topk(sim, k=num_to_update, dim=-1, largest=False)
                update_mask = torch.zeros_like(sim, dtype=torch.bool).scatter(
                    -1, topk.indices, True
                ).unsqueeze(-1) # (B, L, 1)
            else:
                update_mask = torch.zeros_like(sim, dtype=torch.bool).unsqueeze(-1)
            
            cached_kv = (cache.get('k'), v_cached)

            # --- 1b. Adaptive Attention ---
            attn_out_current, (k_new, v_new) = self.attn(
                x_norm1, attributes, cached_kv, update_mask
            )
            attn_out_cached = cache.get('attn_out')
            attn_out = torch.where(update_mask, attn_out_current, attn_out_cached)
            x = x + attn_out

            # --- 1c. Adaptive FFN ---
            x_norm2 = self.norm2(x)
            ffn_out_current = self.ffn(x_norm2)
            ffn_out_cached = cache.get('ffn_out')
            ffn_out = torch.where(update_mask, ffn_out_current, ffn_out_cached)
            x = x + ffn_out
            
            # --- 1d. Prepare new cache for next step ---
            new_cache = {'k': k_new, 'v': v_new, 'attn_out': attn_out, 'ffn_out': ffn_out}
            
            return x, new_cache

        else:
            # --- 2. TRAINING / NO-CACHE PATH ---
            
            # 2a. Standard Attention
            x_norm1 = self.norm1(x)
            # We don't pass cache or update_mask, and we don't need the new K, V
            attn_out, _ = self.attn(
                x_norm1, attributes, cached_kv=None, update_mask=None
            )
            x = x + attn_out

            # 2b. Standard FFN
            x_norm2 = self.norm2(x)
            ffn_out = self.ffn(x_norm2)
            x = x + ffn_out
            
            # 2c. Return x and an empty cache (which will be ignored)
            return x, None

class MoonbeamInput(nn.Module):
    """
    Input Embedding for Moonbeam / ImprovNet. Splits input attributes
    and embeds each separately, then concatenates.
    """
    def __init__(self, hidden_size, vocab_sizes, num_dims=NUM_VOICE_ATTRIBUTES):
        super().__init__()
        assert len(vocab_sizes) == num_dims
        # Must be divisible by 6 (5 attrs + 1 pos_index)
        assert hidden_size % NUM_ATTRIBUTES == 0
        self.emb_dim = hidden_size // NUM_ATTRIBUTES # e.g., 780 / 6 = 130
        
        self.embeds = nn.ModuleList([
            nn.Embedding(vocab_sizes[i], self.emb_dim) for i in range(num_dims)
        ])
        
    def forward(self, instrument, pitch, velocity, onset, duration):
        inputs = [instrument, pitch, velocity, onset, duration]
        embeds = [self.embeds[i](inputs[i]) for i in range(len(inputs))]
        # Returns shape (B, L, C * 5/6)
        return torch.cat(embeds, dim=-1)

class ImprovNetConfig(PretrainedConfig):
    model_type = "improvnet"

    def __init__(
        self,
        hidden_size=780,
        num_heads=12,
        num_layers=16,
        ffn_dim=3120, # 780 * 4
        vocab_sizes=[129, 128, 128, 1024, 1024], # 5 vocabs
        seq_len=2048, # Length of ONE stream
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
        
        # MODIFICATION: New 6-D constraints
        assert hidden_size % num_heads == 0
        assert num_heads % NUM_ATTRIBUTES == 0, f"num_heads ({num_heads}) must be divisible by {NUM_ATTRIBUTES}"
        assert hidden_size % NUM_ATTRIBUTES == 0, f"hidden_size ({hidden_size}) must be divisible by {NUM_ATTRIBUTES}"

# ---
# --- ImprovNet (MODIFIED) ---
# ---
class ImprovNet(PreTrainedModel):
    config_class = ImprovNetConfig
    
    def __init__(self, config: ImprovNetConfig):
        super().__init__(config)
        self.config = config
        self.seq_len = config.seq_len # This is L (length of one stream)
        self.vocab_sizes = config.vocab_sizes
        self.gradient_checkpointing = config.gradient_checkpointing

        # --- MODIFICATION: Handle 5 attribute vocabs + 1 pos embedding ---
        self.input_embed = MoonbeamInput(config.hidden_size, config.vocab_sizes)
        
        # Pos embedding for 2 * L total tokens
        self.pos_embed = nn.Embedding(
            config.seq_len * 2, 
            config.hidden_size // NUM_ATTRIBUTES # emb_dim (C/6)
        )
        
        self.genre_embed = nn.Embedding(config.num_genres, config.hidden_size)
        self.form_embed = nn.Embedding(config.num_forms, config.hidden_size)

        self.transformer_blocks = nn.ModuleList([
            ImprovNetTransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.hidden_size)

        # Output heads are still for 5 attributes
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
        # Manually init pos_embed
        if hasattr(self, 'pos_embed'):
            self.pos_embed.weight.data.normal_(mean=0.0, std=self.config.initializer_range)


    def _calculate_loss(self, all_logits, labels, loss_mask):
        total_loss = 0.0
        loss_count = 0
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        all_labels = torch.unbind(labels, dim=-1)
        all_loss_masks = torch.unbind(loss_mask, dim=-1)
        
        # We only have 5 attributes in labels
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
        cache: Optional[list] = None, 
        return_dict=None
    ):
        B, L, A = input_attributes_main.shape
        assert A == NUM_VOICE_ATTRIBUTES, f"Input tensors must have {NUM_VOICE_ATTRIBUTES} attributes"
        assert L == self.seq_len, f"Input sequence length ({L}) doesn't match config.seq_len ({self.seq_len})"
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # --- 1. Embedding and Conditioning ---
        main_attrs = torch.unbind(input_attributes_main, dim=-1)
        accom_attrs = torch.unbind(input_attributes_accom, dim=-1)
        
        # --- MODIFICATION: Create 6th attribute (seq_index) ---
        # (B, L) tensor with values [0, 1, ..., L-1]
        seq_indices_main = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
        # (B, L) tensor with values [L, L+1, ..., 2L-1]
        seq_indices_accom = torch.arange(L, 2*L, device=self.device).unsqueeze(0).expand(B, L)

        # Get 5-attr embedding: (B, L, C * 5/6)
        x_main_5 = self.input_embed(main_attrs[0], main_attrs[1], main_attrs[2], main_attrs[3], main_attrs[4])
        x_accom_5 = self.input_embed(accom_attrs[0], accom_attrs[1], accom_attrs[2], accom_attrs[3], accom_attrs[4])

        # Get 6th attr embedding: (B, L, C * 1/6)
        pos_emb_main = self.pos_embed(seq_indices_main)
        pos_emb_accom = self.pos_embed(seq_indices_accom)
        
        # Concatenate to full C dimension: (B, L, C)
        x_main = torch.cat([x_main_5, pos_emb_main], dim=-1)
        x_accom = torch.cat([x_accom_5, pos_emb_accom], dim=-1)
        
        g_emb = self.genre_embed(genre).unsqueeze(1)
        f_emb = self.form_embed(form).unsqueeze(1)
        cond_emb = g_emb + f_emb
        x_main = x_main + cond_emb
        x_accom = x_accom + cond_emb
        
        # --- 2. Concatenate Streams ---
        x_combined = torch.cat([x_main, x_accom], dim=1) # (B, 2*L, C)
        
        # --- MODIFICATION: Build 6-attribute tensor for MRA ---
        attributes_main_5 = input_attributes_main.float()
        attributes_accom_5 = input_attributes_accom.float()
        
        # Add 6th attribute (seq_index)
        attrs_main_6d = torch.cat([attributes_main_5, seq_indices_main.float().unsqueeze(-1)], dim=-1)
        attrs_accom_6d = torch.cat([attributes_accom_5, seq_indices_accom.float().unsqueeze(-1)], dim=-1)
        
        attrs_combined = torch.cat([attrs_main_6d, attrs_accom_6d], dim=1) # (B, 2*L, 6)

        # --- 3. Transformer Stack with Cache Logic ---
        use_cache = not self.training
        
        if use_cache:
            if cache is None:
                cache = [None] * self.config.num_layers
            new_cache_list = []
        else:
            cache = [None] * self.config.num_layers

        for i, block in enumerate(self.transformer_blocks):
            if use_cache:
                x_combined, new_cache = block(
                    x_combined, 
                    attrs_combined, 
                    cache=cache[i], 
                    update_ratio=self.config.adaptive_update_ratio
                )
                new_cache_list.append(new_cache)
            else: 
                if self.gradient_checkpointing and self.training:
                    output_tuple = checkpoint(block, x_combined, attrs_combined, None, self.config.adaptive_update_ratio, use_reentrant=False)
                    x_combined = output_tuple[0]
                else:
                    output_tuple = block(x_combined, attrs_combined, cache=None, update_ratio=self.config.adaptive_update_ratio)
                    x_combined = output_tuple[0]
            
        # --- 4. Final Norm and Split ---
        x_combined_norm = self.final_norm(x_combined)
        x_main_norm, x_accom_norm = x_combined_norm.chunk(2, dim=1)
        
        # --- 5. Get Logits ---
        logits_main = tuple(head(x_main_norm) for head in self.output_heads_main)
        logits_accom = tuple(head(x_accom_norm) for head in self.output_heads_accom)
        
        # --- 6. Calculate Loss ---
        total_loss = None
        if labels_main is not None:
            loss_main, count_main = self._calculate_loss(logits_main, labels_main, loss_mask_main)
            loss_accom, count_accom = self._calculate_loss(logits_accom, labels_accom, loss_mask_accom)
            total_loss_value = loss_main + loss_accom
            total_count = count_main + count_accom
            total_loss = total_loss_value / total_count if total_count > 0 else torch.tensor(0.0, device=x_combined_norm.device)
        
        # --- 7. Return Output ---
        output = {
            "logits_main": logits_main,
            "logits_accom": logits_accom
        }
        if total_loss is not None:
            output["loss"] = total_loss
        if use_cache:
            output["cache"] = new_cache_list
        if not return_dict:
            list_output = []
            if total_loss is not None: list_output.append(output["loss"])
            list_output.extend([output["logits_main"], output["logits_accom"]])
            if use_cache: list_output.append(output["cache"])
            return tuple(list_output)
        
        return output

# ---
# --- TEST SCRIPT (MODIFIED for Non-FIT ImprovNet) ---
# ---
if __name__ == "__main__":
    
    # --- 1. Model Configuration ---
    NUM_MRA_HEADS = 5 
    HIDDEN_SIZE = 80
    NUM_LAYERS = 4
    FFN_DIM = 320 # 80 * 4
    BATCH_SIZE = 2
    SEQ_LEN = 128
    
    # [instrument, pitch, velocity, onset, duration]
    VOCAB_SIZES = [129, 128, 128, 1024, 1024]
    NUM_GENRES = 10
    NUM_FORMS = 5 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 2. Dual Stream Dummy Inputs ---
    def create_stream_data(B, L, vocabs):
        return torch.stack([
            torch.randint(0, vocabs[0], (B, L)),
            torch.randint(0, vocabs[1], (B, L)),
            torch.randint(0, vocabs[2], (B, L)),
            torch.randint(0, vocabs[3], (B, L)),
            torch.randint(0, vocabs[4], (B, L))
        ], dim=-1).to(device)

    input_attributes_main = create_stream_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZES) # Shape: (B, L, 5)
    input_attributes_accom = create_stream_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZES) # Shape: (B, L, 5)
    genre = torch.randint(0, NUM_GENRES, (BATCH_SIZE,)).to(device)
    form = torch.randint(0, NUM_FORMS, (BATCH_SIZE,)).to(device)
    
    labels_main = input_attributes_main.clone()
    labels_accom = input_attributes_accom.clone()
    loss_mask_main = (torch.rand(BATCH_SIZE, SEQ_LEN, NUM_ATTRIBUTES) > 0.8).to(device)
    loss_mask_accom = (torch.rand(BATCH_SIZE, SEQ_LEN, NUM_ATTRIBUTES) > 0.8).to(device)
    loss_mask_main[0, 0, 0] = True 
    loss_mask_accom[0, 0, 0] = True

    print(f"--- Testing NON-FIT ImprovNet (Flash MRA, SwiGLU, HF) ---")
    print(f"Device: {device}")
    
    # --- 3. Instantiate via Config ---
    config = ImprovNetConfig(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_MRA_HEADS,
        num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM,
        vocab_sizes=VOCAB_SIZES,
        seq_len=SEQ_LEN,
        num_genres=NUM_GENRES,
        num_forms=NUM_FORMS,
        gradient_checkpointing=True,
        no_bias=True
    )
    
    model = ImprovNet(config=config).to(device)
    model.train() 

    try:
        # --- 4. Run Forward Pass ---
        output = model(
            input_attributes_main=input_attributes_main,
            input_attributes_accom=input_attributes_accom,
            genre=genre,
            form=form,
            labels_main=labels_main,
            labels_accom=labels_accom,
            loss_mask_main=loss_mask_main,
            loss_mask_accom=loss_mask_accom,
            return_dict=True
        )
        
        print("\nForward pass SUCCESSFUL!")
        print(f"Computed Loss: {output['loss'].item():.4f}")
        
        # --- 5. Test Saving and Loading ---
        print("\n--- Testing .save_pretrained() & .from_pretrained() ---")
        save_directory = "./improvnet_model"
        
        model.save_pretrained(save_directory)
        print(f"Model saved to {save_directory}")
        
        loaded_model = ImprovNet.from_pretrained(save_directory).to(device)
        loaded_model.eval()
        print("Model successfully loaded with .from_pretrained()")

        with torch.no_grad():
            output_loaded = loaded_model(
                input_attributes_main=input_attributes_main,
                input_attributes_accom=input_attributes_accom,
                genre=genre,
                form=form
            )
        
        assert "loss" not in output_loaded
        assert "logits_main" in output_loaded
        print("Loaded model forward pass successful.")
        
        # Check logits shape
        logits_main = output_loaded["logits_main"]
        assert isinstance(logits_main, tuple) and len(logits_main) == NUM_ATTRIBUTES
        assert list(logits_main[0].shape) == [BATCH_SIZE, SEQ_LEN, VOCAB_SIZES[0]]
        print(f"Logits shape check PASSED.")

            
    except Exception as e:
        print(f"\nForward pass FAILED: {e}")
        import traceback
        traceback.print_exc()