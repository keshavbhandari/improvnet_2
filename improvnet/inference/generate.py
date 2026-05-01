import torch
import torch.nn.functional as F
import tqdm
import sys
import math
import os

from improvnet.model.model import AmortizedImprovNet, ImprovNetConfig
from improvnet.utils.utils import ProcessData, MAX_DIFFUSION_STEPS
from improvnet.train.training_config import *

# --- Constants ---
ATTR_ORDER = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
NUM_ATTRIBUTES = 5
PAD_TOKEN_ID = 2 
MASK_TOKEN_ID = 6 # Ensure this matches your tokenizer!
EOS_TOKEN_ID = 1   

def tokens_to_tensor(processor, tokens, device):
    """Converts list of tokens to [1, L, 5] tensor."""
    if not tokens:
        # Return empty valid tensor for logic consistency
        return torch.zeros((1, 0, NUM_ATTRIBUTES), dtype=torch.long, device=device)
    
    t_dict = processor.tokens_to_tensor(tokens)
    t_stack = torch.stack([t_dict[attr] for attr in ATTR_ORDER], dim=1)
    return t_stack.unsqueeze(0).to(device)

def tensor_to_tokens(processor, tensor):
    """Converts [1, L, 5] tensor back to list of tokens."""
    if tensor.shape[1] == 0:
        return []
    t = tensor.squeeze(0)
    d = {ATTR_ORDER[i]: t[:, i].cpu() for i in range(NUM_ATTRIBUTES)}
    return processor.tensor_to_tokens(d)

def pad_tensor_to_len(tensor, target_len, pad_val=PAD_TOKEN_ID):
    """Pads [1, L, 5] to [1, target_len, 5]."""
    curr_len = tensor.shape[1]
    if curr_len >= target_len:
        return tensor[:, :target_len, :]
    
    pad_len = target_len - curr_len
    pad_t = torch.full((1, pad_len, NUM_ATTRIBUTES), pad_val, dtype=torch.long, device=tensor.device)
    return torch.cat([tensor, pad_t], dim=1)

@torch.no_grad()
def decode_stream(
    model: AmortizedImprovNet,
    heads: torch.nn.ModuleList, 
    encoder_hidden_states: torch.Tensor, 
    context_seq: torch.Tensor, 
    target_len: int,           
    start_idx_offset: int,     
    genre, form, timestep,
    past_key_values,           
    temperature=1.0, 
    top_k=50
):
    """
    Autoregressive Decoder Loop with KV Cache and EOS Stopping.
    """
    device = context_seq.device
    curr_seq = context_seq.clone()
    
    current_cache = past_key_values
    
    # Progress Bar configuration
    steps_to_gen = target_len - curr_seq.shape[1]
    
    if steps_to_gen > 0:
        with tqdm.tqdm(total=steps_to_gen, desc=f"Decoding (Offset {start_idx_offset})", leave=False) as pbar:
            
            while curr_seq.shape[1] < target_len:
                # 1. Prepare Input (Last token only if caching)
                if current_cache is not None:
                    dec_input = curr_seq[:, -1:, :]
                    curr_global_idx = start_idx_offset + curr_seq.shape[1] - 1
                else:
                    dec_input = curr_seq
                    curr_global_idx = start_idx_offset

                # 2. Embeddings
                x_dec = model._prepare_embeddings(
                    dec_input, start_idx=curr_global_idx, device=device, 
                    genre=genre, form=form, timestep=timestep
                )
                
                # 3. MRA Attributes
                L_slice = dec_input.shape[1]
                seq_idx = torch.arange(curr_global_idx, curr_global_idx + L_slice, device=device).unsqueeze(0).expand(1, L_slice).unsqueeze(-1)
                mra_attrs = torch.cat([dec_input[..., :5].float(), seq_idx.float()], dim=-1)
                
                # 4. Mask
                if current_cache is None:
                    causal_mask = torch.triu(torch.ones(L_slice, L_slice, device=device) * float('-inf'), diagonal=1)
                else:
                    causal_mask = None

                # 5. Forward through Blocks
                new_kv_list = []
                for i, block in enumerate(model.decoder_blocks):
                    layer_cache = current_cache[i] if current_cache is not None else None
                    
                    x_dec, new_kv = block(
                        x_dec,
                        attributes=mra_attrs,
                        encoder_hidden_states=encoder_hidden_states,
                        self_attention_mask=causal_mask,
                        past_key_value=layer_cache
                    )
                    new_kv_list.append(new_kv)
                
                current_cache = new_kv_list
                
                # 6. Sampling
                x_final = model.final_norm(x_dec) 
                
                next_tok_attrs = []
                for i in range(NUM_ATTRIBUTES):
                    logits = heads[i](x_final)[:, -1, :] 
                    logits = logits / temperature
                    
                    if top_k > 0:
                        # Safety: Clamp top_k to vocab size
                        vocab_size = logits.shape[-1]
                        effective_k = min(top_k, vocab_size)
                        v, _ = torch.topk(logits, effective_k)
                        logits[logits < v[:, [-1]]] = -float('Inf')
                    
                    probs = F.softmax(logits, dim=-1)
                    next_val = torch.multinomial(probs, 1)
                    next_tok_attrs.append(next_val)
                    
                next_token_t = torch.cat(next_tok_attrs, dim=-1).unsqueeze(1) # [1, 1, 5]
                
                # 7. Check for EOS
                if (next_token_t[0, 0, 0] == EOS_TOKEN_ID):
                    curr_seq = torch.cat([curr_seq, next_token_t], dim=1)
                    pbar.update(1)
                    break
                    
                curr_seq = torch.cat([curr_seq, next_token_t], dim=1)
                pbar.update(1)
        
    return curr_seq, current_cache

def generate_segment(
    model, processor, 
    main_fixed_t, accom_fixed_t, 
    main_new_init_t, accom_new_init_t, 
    genre, form, 
    inference_steps, 
    segment_len, 
    start_ratio, 
    temperature,
    device
):
    # 1. Setup Schedule
    start_t = int(MAX_DIFFUSION_STEPS * start_ratio)
    timesteps = torch.linspace(start_t, 0, inference_steps + 1).long().to(device)
    
    len_main_new = main_new_init_t.shape[1]
    len_accom_new = accom_new_init_t.shape[1]
    
    # Initialize "Clean Estimates" with Original Data (Style Transfer)
    # or Empty if lengths are 0
    curr_main_new = main_new_init_t.clone()
    curr_accom_new = accom_new_init_t.clone()
    
    gen_main = len_main_new > 0
    gen_accom = len_accom_new > 0

    # --- DIFFUSION LOOP ---
    for i in range(inference_steps):
        curr_t = timesteps[i]
        print(f"    Step {i+1}/{inference_steps} (t={curr_t.item()})")
        
        # Use BF16 for speed
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            
            # --- A. Noising / Re-composition ---
            mask_prob = processor.get_mask_prob_for_timestep(curr_t.item())
            
            def apply_mask(clean_est, prob):
                if clean_est.shape[1] == 0: return clean_est
                # Independent attribute masking on the CURRENT ESTIMATE
                mask = torch.rand(clean_est.shape, device=device) < prob
                noisy = clean_est.clone()
                noisy[mask] = MASK_TOKEN_ID
                return noisy
                
            noisy_main_new = apply_mask(curr_main_new, mask_prob)
            noisy_accom_new = apply_mask(curr_accom_new, mask_prob)
            
            # Concatenate [Fixed; Noisy_New]
            enc_input_main = torch.cat([main_fixed_t, noisy_main_new], dim=1)
            enc_input_accom = torch.cat([accom_fixed_t, noisy_accom_new], dim=1)
            
            # PAD to segment_len for the Encoder (Required for Pos Emb Match)
            enc_input_main_pad = pad_tensor_to_len(enc_input_main, segment_len)
            enc_input_accom_pad = pad_tensor_to_len(enc_input_accom, segment_len)
            
            # Combine [Main; Accom]
            enc_input_full = torch.cat([enc_input_main_pad, enc_input_accom_pad], dim=1)
            
            # --- B. Encoder Pass ---
            x_enc = model._prepare_embeddings(enc_input_full, 0, device, genre, form, curr_t.unsqueeze(0))
            
            L_enc = enc_input_full.shape[1]
            seq_idx_enc = torch.arange(0, L_enc, device=device).unsqueeze(0).expand(1, L_enc).unsqueeze(-1)
            mra_attrs_enc = torch.cat([enc_input_full[..., :5].float(), seq_idx_enc.float()], dim=-1)
            
            for blk in model.encoder_blocks:
                x_enc = blk(x_enc, attributes=mra_attrs_enc)
            encoder_context = x_enc
            
            # --- C. Decoder Phase ---
            
            # 1. Generate Main
            if gen_main:
                target_len = main_fixed_t.shape[1] + len_main_new
                full_main, past_kv = decode_stream(
                    model, model.output_heads_main, encoder_context,
                    context_seq=main_fixed_t, 
                    target_len=target_len,
                    start_idx_offset=0,
                    genre=genre, form=form, timestep=curr_t.unsqueeze(0),
                    past_key_values=None, temperature=temperature
                )
                # Update Estimate for next step
                curr_main_new = full_main[:, main_fixed_t.shape[1]:, :]
            else:
                # If not generating Main, we MUST still prefill the cache with Fixed Main
                # so Accom can attend to it.
                if gen_accom:
                     _, past_kv = decode_stream(
                        model, model.output_heads_main, encoder_context,
                        context_seq=main_fixed_t,
                        target_len=main_fixed_t.shape[1], # Prefill only
                        start_idx_offset=0,
                        genre=genre, form=form, timestep=curr_t.unsqueeze(0),
                        past_key_values=None, temperature=temperature
                    )
                else:
                    past_kv = None
                    full_main = main_fixed_t 

            # 2. Generate Accom
            if gen_accom:
                target_len = accom_fixed_t.shape[1] + len_accom_new
                full_accom, _ = decode_stream(
                    model, model.output_heads_accom, encoder_context,
                    context_seq=accom_fixed_t,
                    target_len=target_len,
                    start_idx_offset=MAX_LEN, # Important: Accom Offset
                    genre=genre, form=form, timestep=curr_t.unsqueeze(0),
                    past_key_values=past_kv, # Attend to Main
                    temperature=temperature
                )
                curr_accom_new = full_accom[:, accom_fixed_t.shape[1]:, :]
    
    return curr_main_new, curr_accom_new

def run_cascaded_generation(
    model_path, input_midi, output_path,
    genre_str="classical", form_str="unknown",
    segment_len=MAX_LEN, overlap_ratio=0.20,
    inference_steps=10, start_ratio=0.8,
    temperature=1.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Load Model (BF16) ---
    print(f"Loading model from {model_path} in BF16...")
    model = AmortizedImprovNet.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    model.eval()
    processor = ProcessData()
    
    # 2. Read Full MIDI
    main_toks_full, accom_toks_full, g_tens, f_tens = processor.inference_pipeline(input_midi, genre_str, form_str)
    g_tens, f_tens = g_tens.to(device), f_tens.to(device)
    
    # 3. Check Streams
    has_main = len(main_toks_full) > 0
    has_accom = len(accom_toks_full) > 0
    
    # 4. Setup Windows
    overlap_len = int(segment_len * overlap_ratio)
    stride = segment_len - overlap_len 
    total_len = max(len(main_toks_full), len(accom_toks_full))
    num_windows = math.ceil(total_len / stride)
    
    final_main_seq = []
    final_accom_seq = []
    
    # Initial Context (Fixed Prompt from File)
    final_main_seq.extend(main_toks_full[:overlap_len])
    final_accom_seq.extend(accom_toks_full[:overlap_len])
    
    print(f"Starting Style Transfer ({num_windows} windows, Start Ratio={start_ratio})...")
    
    for w in range(num_windows):
        print(f"--- Window {w+1}/{num_windows} ---")
        
        ctx_main = final_main_seq[-overlap_len:]
        ctx_accom = final_accom_seq[-overlap_len:]
        
        start_idx = overlap_len + (w * stride)
        end_idx = start_idx + stride
        
        # Get Original Data for "New" part (to be transformed)
        raw_main_new = main_toks_full[start_idx:end_idx] if has_main else []
        raw_accom_new = accom_toks_full[start_idx:end_idx] if has_accom else []
        
        t_ctx_main = tokens_to_tensor(processor, ctx_main, device)
        t_ctx_accom = tokens_to_tensor(processor, ctx_accom, device)
        t_raw_main = tokens_to_tensor(processor, raw_main_new, device)
        t_raw_accom = tokens_to_tensor(processor, raw_accom_new, device)
        
        # Run Diffusion on this Segment
        gen_main_t, gen_accom_t = generate_segment(
            model, processor,
            t_ctx_main, t_ctx_accom,
            t_raw_main, t_raw_accom,
            g_tens, f_tens,
            inference_steps, segment_len, start_ratio, temperature, device
        )
        
        if has_main:
            new_main_toks = tensor_to_tokens(processor, gen_main_t)
            final_main_seq.extend(new_main_toks)
            
        if has_accom:
            new_accom_toks = tensor_to_tokens(processor, gen_accom_t)
            final_accom_seq.extend(new_accom_toks)

    # 5. Save
    print("Decoding and Saving...")
    if has_main:
        midi_main = processor.tokens_to_midi(final_main_seq)
        processor.save_midi(midi_main, output_path.replace(".mid", "_main.mid"))
        print(f"Saved Main track.")
        
    if has_accom:
        midi_accom = processor.tokens_to_midi(final_accom_seq)
        processor.save_midi(midi_accom, output_path.replace(".mid", "_accom.mid"))
        print(f"Saved Accom track.")

if __name__ == "__main__":
    model_path = "/gpfs/scratch/acw769/improvnet_new/pretraining/checkpoint/"
    input_midi = "/data/home/acw769/improvnet_2/improvnet/inference/debussy-clair-de-lune_original.mid"
    output_path = "/data/home/acw769/improvnet_2/improvnet/inference/output.mid"
    run_cascaded_generation(model_path, input_midi, output_path)