import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from transformers import PreTrainedModel, PretrainedConfig

# --- Constants ---
NUM_ATTRIBUTES = 6
NUM_VOICE_ATTRIBUTES = 5 
DEFAULT_MRA_BASE_VALUES = [100.0, 131.0, 20.0, 1031.0, 1031.0, 10000.0]

class ImprovNetConfig(PretrainedConfig):
    model_type = "improvnet"
    def __init__(
        self,
        hidden_size=780, 
        num_heads=30,
        num_decoder_layers=12, 
        num_encoder_layers=4,
        ffn_dim=3120,
        vocab_sizes=[129, 128, 128, 512, 512],
        seq_len=2048,
        num_genres=10,
        num_forms=5,
        no_bias=False,
        gradient_checkpointing=False,
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.ffn_dim = ffn_dim
        self.vocab_sizes = vocab_sizes
        self.seq_len = seq_len
        self.num_genres = num_genres
        self.num_forms = num_forms
        self.no_bias = no_bias
        self.gradient_checkpointing = gradient_checkpointing
        self.initializer_range = initializer_range
        self.head_dim = hidden_size // num_heads

# --- 1. Embeddings ---
class MoonbeamInput(nn.Module):
    def __init__(self, hidden_size, vocab_sizes, num_dims=NUM_VOICE_ATTRIBUTES):
        super().__init__()
        self.attr_emb_dim = hidden_size // NUM_ATTRIBUTES 
        self.embeds = nn.ModuleList([
            nn.Embedding(vocab_sizes[i], self.attr_emb_dim) for i in range(num_dims)
        ])
    def forward(self, instrument, pitch, velocity, onset, duration):
        inputs = [instrument, pitch, velocity, onset, duration]
        embeds = [self.embeds[i](inputs[i]) for i in range(len(inputs))]
        return torch.cat(embeds, dim=-1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

# --- 2. Attention with KV Cache Support ---
class MRAAttention(nn.Module):
    def __init__(self, config: ImprovNetConfig, is_cross_attention=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.is_cross_attention = is_cross_attention
        
        # MRA Setup (Only for Self-Attention)
        if not is_cross_attention:
            self.num_groups = NUM_ATTRIBUTES
            self.heads_per_group = self.num_heads // self.num_groups
            for i in range(self.num_groups):
                base = DEFAULT_MRA_BASE_VALUES[i]
                inv_freqs_g = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
                self.register_buffer(f'inv_freqs_{i}', inv_freqs_g)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=not config.no_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=not config.no_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=not config.no_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=not config.no_bias)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, x, positions, inv_freqs):
        sinusoid_inp = torch.einsum("b l, d -> b l d", positions, inv_freqs)
        sin = torch.sin(sinusoid_inp).unsqueeze(2).repeat_interleave(2, dim=-1)
        cos = torch.cos(sinusoid_inp).unsqueeze(2).repeat_interleave(2, dim=-1)
        return (x * cos) + (self._rotate_half(x) * sin)

    def apply_mra_rotation(self, tensor, attributes):
        rotated_list = []
        for g in range(self.num_groups):
            start_head = g * self.heads_per_group
            end_head = (g + 1) * self.heads_per_group
            tensor_group = tensor[:, :, start_head:end_head, :]
            position_values = attributes[:, :, g].float()
            inv_freqs_g = getattr(self, f'inv_freqs_{g}')
            rotated_group = self._apply_rotary_pos_emb(tensor_group, position_values, inv_freqs_g)
            rotated_list.append(rotated_group)
        return torch.cat(rotated_list, dim=2)

    def forward(
        self, 
        hidden_states, 
        encoder_hidden_states=None, 
        attributes=None,            
        attention_mask=None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None
    ):
        B, L, C = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, D)
        
        if self.is_cross_attention:
            if past_key_value is not None:
                k, v = past_key_value
            elif encoder_hidden_states is not None:
                B_enc, L_enc, _ = encoder_hidden_states.shape
                k = self.k_proj(encoder_hidden_states).view(B_enc, L_enc, H, D)
                v = self.v_proj(encoder_hidden_states).view(B_enc, L_enc, H, D)
            else:
                raise ValueError("Cross attention requires encoder_hidden_states or past_key_value")
            
            q_rot, k_rot = q.transpose(1, 2), k.transpose(1, 2)
            v_final = v.transpose(1, 2)
            
            current_key_value = (k, v)

        else:
            # Self-Attention
            k = self.k_proj(hidden_states).view(B, L, H, D)
            v = self.v_proj(hidden_states).view(B, L, H, D)
            
            q_rot = self.apply_mra_rotation(q, attributes)
            k_rot = self.apply_mra_rotation(k, attributes)
            
            if past_key_value is not None:
                past_k, past_v = past_key_value
                k_rot = torch.cat([past_k, k_rot], dim=1) 
                v = torch.cat([past_v, v], dim=1)
            
            current_key_value = (k_rot, v)
            
            q_rot = q_rot.transpose(1, 2)
            k_rot = k_rot.transpose(1, 2)
            v_final = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q_rot, k_rot, v_final, attn_mask=attention_mask, is_causal=False 
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, L, C)
        return self.o_proj(attn_output), current_key_value

# --- 3. Encoder Block ---
class EncoderBlock(nn.Module):
    def __init__(self, config: ImprovNetConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.attn = MRAAttention(config, is_cross_attention=False)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.w1 = nn.Linear(config.hidden_size, config.ffn_dim)
        self.w2 = nn.Linear(config.hidden_size, config.ffn_dim)
        self.w3 = nn.Linear(config.ffn_dim, config.hidden_size)
        self.act = nn.SiLU()

    def forward(self, x, attributes, attention_mask=None):
        res = x
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, attributes=attributes, attention_mask=attention_mask)
        x = res + attn_out
        
        res = x
        x_norm = self.ffn_norm(x)
        ffn_out = self.w3(self.act(self.w1(x_norm)) * self.w2(x_norm))
        x = res + ffn_out
        return x

# --- 4. Decoder Block ---
class DecoderBlock(nn.Module):
    def __init__(self, config: ImprovNetConfig):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(config.hidden_size)
        self.self_attn = MRAAttention(config, is_cross_attention=False)
        
        self.cross_attn_norm = nn.LayerNorm(config.hidden_size)
        self.cross_attn = MRAAttention(config, is_cross_attention=True)
        
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.w1 = nn.Linear(config.hidden_size, config.ffn_dim)
        self.w2 = nn.Linear(config.hidden_size, config.ffn_dim)
        self.w3 = nn.Linear(config.ffn_dim, config.hidden_size)
        self.act = nn.SiLU()

    def forward(
        self, 
        x, 
        attributes, 
        encoder_hidden_states, 
        self_attention_mask=None,
        cross_attention_mask=None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None
    ):
        self_cache = past_key_value[0] if past_key_value is not None else None
        cross_cache = past_key_value[1] if past_key_value is not None else None

        # 1. Self Attention
        res = x
        x_norm = self.self_attn_norm(x)
        attn_out, new_self_cache = self.self_attn(
            x_norm, 
            attributes=attributes, 
            attention_mask=self_attention_mask,
            past_key_value=self_cache
        )
        x = res + attn_out
        
        # 2. Cross Attention
        res = x
        x_norm = self.cross_attn_norm(x)
        attn_out, new_cross_cache = self.cross_attn(
            x_norm,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=cross_attention_mask,
            past_key_value=cross_cache
        )
        x = res + attn_out
        
        # 3. FFN
        res = x
        x_norm = self.ffn_norm(x)
        ffn_out = self.w3(self.act(self.w1(x_norm)) * self.w2(x_norm))
        x = res + ffn_out
        
        return x, (new_self_cache, new_cross_cache)

# --- 5. Amortized Model ---
class AmortizedImprovNet(PreTrainedModel):
    config_class = ImprovNetConfig
    def __init__(self, config: ImprovNetConfig):
        super().__init__(config)
        self.config = config
        
        self.input_embed = MoonbeamInput(config.hidden_size, config.vocab_sizes)
        self.pos_embed = nn.Embedding(config.seq_len * 2, config.hidden_size // NUM_ATTRIBUTES)
        self.genre_embed = nn.Embedding(config.num_genres, config.hidden_size)
        self.form_embed = nn.Embedding(config.num_forms, config.hidden_size)
        self.time_embed = TimestepEmbedder(config.hidden_size)
        
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_encoder_layers)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_decoder_layers)])
        
        self.final_norm = nn.LayerNorm(config.hidden_size)
        
        self.output_heads_main = nn.ModuleList([nn.Linear(config.hidden_size, config.vocab_sizes[i]) for i in range(NUM_VOICE_ATTRIBUTES)])
        self.output_heads_accom = nn.ModuleList([nn.Linear(config.hidden_size, config.vocab_sizes[i]) for i in range(NUM_VOICE_ATTRIBUTES)])
        
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _prepare_embeddings(self, attrs, start_idx, device, genre, form, timestep):
        # Corrected Logic: Supports arbitrary slices (e.g., 1 token for inference)
        B, L_chunk, _ = attrs.shape
        
        # 1. Unbind attributes (Compound token)
        unbound = torch.unbind(attrs, dim=-1)
        # Embed compound tokens (Instrument, Pitch, Vel, Onset, Dur)
        x = self.input_embed(unbound[0], unbound[1], unbound[2], unbound[3], unbound[4])
        
        # 2. Positional Embeddings
        # Generate global position indices for this chunk
        pos_ids = torch.arange(start_idx, start_idx + L_chunk, device=device).unsqueeze(0).expand(B, L_chunk)
        x = torch.cat([x, self.pos_embed(pos_ids)], dim=-1)
        
        # 3. Conditions
        cond_emb = (self.genre_embed(genre) + self.form_embed(form)).unsqueeze(1)
        time_emb = self.time_embed(timestep).unsqueeze(1)
        
        x = x + cond_emb + time_emb
        return x

    def forward(
        self, 
        input_attributes_encoder, 
        input_attributes_decoder, 
        genre,
        form,
        timestep, 
        labels_main=None,
        labels_accom=None,
        past_key_values=None, 
        return_dict=None
    ):
        device = input_attributes_encoder.device
        B = input_attributes_encoder.shape[0]
        
        # --- 1. Encoder Pass (Bidirectional) ---
        x_enc = self._prepare_embeddings(input_attributes_encoder, start_idx=0, device=device, genre=genre, form=form, timestep=timestep)
        
        # MRA Attributes for Encoder
        L_enc = input_attributes_encoder.shape[1]
        seq_idx_enc = torch.arange(0, L_enc, device=device).unsqueeze(0).expand(B, L_enc).unsqueeze(-1)
        
        mra_attrs_enc = torch.cat([
            input_attributes_encoder[..., :5].float(),
            seq_idx_enc.float()
        ], dim=-1)

        # --- Cleaned Encoder Loop ---
        for block in self.encoder_blocks:
            if self.config.gradient_checkpointing and self.training:
                # Directly pass 'block' as the function
                x_enc = torch.utils.checkpoint.checkpoint(
                    block,
                    x_enc,
                    mra_attrs_enc,
                    None, # attention_mask
                    use_reentrant=False
                )
            else:
                x_enc = block(x_enc, attributes=mra_attrs_enc)
            
        encoder_hidden_states = x_enc

        # --- 2. Decoder Pass (Causal) ---
        if past_key_values is not None:
            start_idx = past_key_values[0][0][0].shape[1]
        else:
            start_idx = 0
            
        x_dec = self._prepare_embeddings(input_attributes_decoder, start_idx=start_idx, device=device, genre=genre, form=form, timestep=timestep)
        
        L_dec_slice = input_attributes_decoder.shape[1]
        seq_idx_dec = torch.arange(start_idx, start_idx + L_dec_slice, device=device).unsqueeze(0).expand(B, L_dec_slice).unsqueeze(-1)
        
        mra_attrs_dec = torch.cat([
            input_attributes_decoder[..., :5].float(),
            seq_idx_dec.float()
        ], dim=-1)

        if past_key_values is None:
            causal_mask = torch.triu(torch.ones(L_dec_slice, L_dec_slice, device=device) * float('-inf'), diagonal=1)
        else:
            causal_mask = None 

        new_past_key_values = []
        
        # --- Cleaned Decoder Loop ---
        for i, block in enumerate(self.decoder_blocks):
            layer_cache = past_key_values[i] if past_key_values is not None else None
            
            if self.config.gradient_checkpointing and self.training:
                # Note: Checkpointing is usually incompatible with caching because
                # caching breaks the computation graph needed for re-computation.
                if layer_cache is not None:
                     x_dec, new_cache = block(
                        x_dec,
                        attributes=mra_attrs_dec,
                        encoder_hidden_states=encoder_hidden_states,
                        self_attention_mask=causal_mask,
                        past_key_value=layer_cache
                    )
                else:
                    # Directly pass 'block'
                    # Block returns (x, cache), checkpoint handles tuple returns automatically
                    x_dec, new_cache = torch.utils.checkpoint.checkpoint(
                        block,
                        x_dec,
                        mra_attrs_dec,
                        encoder_hidden_states,
                        causal_mask, # self_attention_mask
                        None,        # cross_attention_mask
                        None,        # past_key_value
                        use_reentrant=False
                    )
            else:
                x_dec, new_cache = block(
                    x_dec,
                    attributes=mra_attrs_dec,
                    encoder_hidden_states=encoder_hidden_states,
                    self_attention_mask=causal_mask,
                    past_key_value=layer_cache
                )
            
            new_past_key_values.append(new_cache)

        # --- 3. Heads ---
        x_final = self.final_norm(x_dec)
        
        L_total = self.config.seq_len * 2
        
        if x_final.shape[1] == L_total:
            x_main_out, x_accom_out = x_final.chunk(2, dim=1)
            logits_main = tuple(head(x_main_out) for head in self.output_heads_main)
            logits_accom = tuple(head(x_accom_out) for head in self.output_heads_accom)
        else:
            logits_main = tuple(head(x_final) for head in self.output_heads_main)
            logits_accom = tuple(head(x_final) for head in self.output_heads_accom)

        # --- 4. Loss Calculation with SHIFTING ---
        total_loss = None
        if labels_main is not None and x_final.shape[1] == L_total:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            shift_labels_main = labels_main[:, 1:, :] # [B, L-1, 5]
            shift_labels_accom = labels_accom[:, 1:, :] # [B, L-1, 5]
            
            loss_main = 0.0
            for i in range(NUM_VOICE_ATTRIBUTES):
                shift_logits = logits_main[i][:, :-1, :].contiguous()
                shift_labels = shift_labels_main[:, :, i].contiguous()
                
                # SAFETY CHECK: Only calculate loss if there are valid tokens
                if (shift_labels != -100).any():
                    loss_main += loss_fct(
                        shift_logits.view(-1, self.config.vocab_sizes[i]), 
                        shift_labels.view(-1)
                    )

            loss_accom = 0.0
            for i in range(NUM_VOICE_ATTRIBUTES):
                shift_logits = logits_accom[i][:, :-1, :].contiguous()
                shift_labels = shift_labels_accom[:, :, i].contiguous()
                
                # SAFETY CHECK: Only calculate loss if there are valid tokens
                # This prevents NaN when the entire batch is Solo (Empty Accompaniment)
                if (shift_labels != -100).any():
                    loss_accom += loss_fct(
                        shift_logits.view(-1, self.config.vocab_sizes[i]), 
                        shift_labels.view(-1)
                    )

            total_loss = loss_main + loss_accom
            
            # Final safety: if total_loss is 0.0 (e.g. complete silence), make it a tensor
            if isinstance(total_loss, float):
                total_loss = torch.tensor(total_loss, device=device, requires_grad=True)

        if not return_dict:
            return (logits_main, logits_accom, new_past_key_values)
            
        return {
            "loss": total_loss,
            "logits_main": logits_main,
            "logits_accom": logits_accom,
            "past_key_values": new_past_key_values
        }