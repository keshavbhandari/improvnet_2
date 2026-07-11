import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from improvnet.model.caddi_config import *
from improvnet.model.caddi_model import CaDDiModel
from improvnet.utils.utils import ProcessData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _amp_dtype():
    if not torch.cuda.is_available(): return None
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16

AMP_DTYPE = _amp_dtype()


class CaDDiInference:
    def __init__(self, model_path: str):
        self.processor = ProcessData()
        self.tokenizer = self.processor.tokenizer
        
        print("Loading 1D CaDDi AR Diffusion Model...")
        self.model = CaDDiModel().to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Tokens the model should NEVER generate as outputs
        self.forbidden_ids = [
            PAD_ID, MASK_ID, BLANK_ID, SEP_ID, 
            self.tokenizer.tok_to_id.get('<U>', -1)
        ]

    @torch.no_grad()
    def _encode_clean_block(self, block, start_pos, genre_id, past_kv):
        """Passes a fully finalized clean block into the model in parallel to permanently update the KV cache."""
        B, L = block.shape
        timestep = torch.zeros(B, L, device=DEVICE) 
        
        with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
            _, next_kv = self.model(
                target=block, timestep=timestep, genre=genre_id, 
                use_cache=True, past_key_values=past_kv, seq_offset=start_pos
            )
        return next_kv

    @torch.no_grad()
    def _denoise_block_ar(self, canvas_block, allowed_mask, past_kv_clean, start_pos, genre_id, mask_ratios, temperature):
        B, L = canvas_block.shape
        # Timesteps associated with the masking ratios
        time_vals = [1.0, 0.75, 0.50, 0.25][:len(mask_ratios)]
        
        current_block = canvas_block.clone()
        kv_accum = past_kv_clean
        prev_confs = None
        draft_pos = start_pos
        
        # -----------------------------------------------------------
        # 0. INJECT PARALLEL BASE DRAFT (DRAFT 0) 
        # This acts as "future context" for the token-by-token loop,
        # ensuring the model can see ALL unmasked pitches across the block!
        # -----------------------------------------------------------
        has_unmasked_context = (~allowed_mask).any()
        
        if has_unmasked_context:
            base_draft = canvas_block.clone()
            valid_indices = allowed_mask[0].nonzero(as_tuple=True)[0]
            base_draft[0, valid_indices] = MASK_ID
            
            if start_pos > 0:
                sep_tensor = torch.tensor([[SEP_ID]], dtype=torch.long, device=DEVICE)
                timestep_sep = torch.zeros(1, 1, device=DEVICE)
                with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                    _, kv_accum = self.model(
                        target=sep_tensor, timestep=timestep_sep, genre=genre_id, 
                        use_cache=True, past_key_values=kv_accum, seq_offset=draft_pos
                    )
                draft_pos += 1
                
            timestep_base = torch.full((B, L), 1.0, device=DEVICE)
            with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                _, kv_accum = self.model(
                    target=base_draft, timestep=timestep_base, genre=genre_id, 
                    use_cache=True, past_key_values=kv_accum, seq_offset=draft_pos
                )
            draft_pos += L
        # -----------------------------------------------------------
        
        for step in range(len(mask_ratios)):
            ratio = mask_ratios[step]
            target_draft = current_block.clone()
            
            # Apply Masks strictly to the allowed tokens (e.g. only Rhythm tokens)
            num_to_mask = int(ratio * allowed_mask.sum().item())
            if num_to_mask > 0:
                valid_indices = allowed_mask[0].nonzero(as_tuple=True)[0]
                
                if prev_confs is None or ratio == 1.0:
                    perm = torch.randperm(len(valid_indices))
                    chosen = valid_indices[perm[:num_to_mask]]
                else:
                    confs = prev_confs[0, valid_indices]
                    _, worst_idx = torch.topk(confs, num_to_mask, largest=False)
                    chosen = valid_indices[worst_idx]
                    
                target_draft[0, chosen] = MASK_ID
                
            draft_out = target_draft.clone()
            draft_confs = torch.zeros(B, L, device=DEVICE) if prev_confs is None else prev_confs.clone()
            
            # -----------------------------------------------------------
            # 1. INJECT THE <SEP> TOKEN INTO THE TIMELINE
            # -----------------------------------------------------------
            if step > 0 or start_pos > 0 or has_unmasked_context:
                sep_tensor = torch.tensor([[SEP_ID]], dtype=torch.long, device=DEVICE)
                timestep_sep = torch.zeros(1, 1, device=DEVICE)
                with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                    _, kv_accum = self.model(
                        target=sep_tensor, timestep=timestep_sep, genre=genre_id, 
                        use_cache=True, past_key_values=kv_accum, seq_offset=draft_pos
                    )
                draft_pos += 1
            
            # -----------------------------------------------------------
            # 2. 1D TOKEN-BY-TOKEN CAUSAL GENERATION LOOP
            # -----------------------------------------------------------
            for i in tqdm(range(L), desc=f"  Refinement Draft {step+1}/{len(mask_ratios)}", leave=False):
                curr_token = target_draft[:, i:i+1]
                timestep = torch.full((1, 1), time_vals[step], device=DEVICE)
                
                if curr_token[0, 0] == MASK_ID:
                    # 1. Forward pass with MASK_ID to query the model (temporary cache discarded)
                    with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                        logits, _ = self.model(
                            target=curr_token, timestep=timestep, genre=genre_id, 
                            use_cache=True, past_key_values=kv_accum, seq_offset=draft_pos
                        )
                        
                    step_logits = logits[:, 0, :] / max(temperature, 1e-5)
                    
                    for f_id in self.forbidden_ids:
                        if f_id >= 0:
                            step_logits[:, f_id] = float('-inf')
                            
                    probs = F.softmax(step_logits, dim=-1)
                    samp = torch.multinomial(probs, num_samples=1) # Shape: [B, 1]
                    conf = probs.gather(-1, samp).squeeze(-1)      # Shape: [B]
                    
                    draft_out[0, i] = samp[0, 0]
                    draft_confs[0, i] = conf[0]
                    
                    # 2. Update permanent KV Cache with the NEW prediction!
                    # This ensures future tokens within THIS draft see the generated context,
                    # eliminating the "dur followed by dur" KV cache blindness.
                    with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                        _, kv_accum = self.model(
                            target=samp, timestep=timestep, genre=genre_id, 
                            use_cache=True, past_key_values=kv_accum, seq_offset=draft_pos
                        )
                else:
                    # Token is clean (ground truth or locked-in prediction from previous draft).
                    # Feed it directly to update KV cache naturally.
                    with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                        _, kv_accum = self.model(
                            target=curr_token, timestep=timestep, genre=genre_id, 
                            use_cache=True, past_key_values=kv_accum, seq_offset=draft_pos
                        )
                    draft_out[0, i] = curr_token[0, 0]
                    draft_confs[0, i] = float('inf') # Protect from being re-masked in later steps
                        
                draft_pos += 1
            
            # The fully completed AR draft becomes the baseline for the next diffusion step
            current_block = draft_out
            prev_confs = draft_confs
            
        return current_block

    @torch.no_grad()
    def generate(
        self, 
        prompt_tokens: list, 
        strategy: str = "rhythm_then_pitch", 
        genre_str: str = "jazz", 
        keep_prompt_len: int = 256,
        temperature: float = 1.0
    ) -> list:
        
        genre_id = torch.tensor([self.processor.get_genre_id(genre_str)], dtype=torch.long, device=DEVICE)
        
        if strategy == "completion":
            prompt_len = min(len(prompt_tokens), keep_prompt_len)
        else:
            prompt_len = min(len(prompt_tokens), SEQ_LEN)
            
        canvas = torch.full((1, SEQ_LEN), PAD_ID, dtype=torch.long, device=DEVICE)
        if prompt_len > 0:
            prompt_tensor = self.processor.format_variable_sequence(prompt_tokens[:prompt_len], prompt_len, pad_id=PAD_ID).unsqueeze(0).to(DEVICE)
            canvas[:, :prompt_len] = prompt_tensor

        final_seq_len = SEQ_LEN
        t_id = self.tokenizer.tok_to_id.get('<T>', -1)

        if strategy == "completion":
            tqdm.write(f"Encoding {prompt_len} Prompt Tokens...")
            past_kv_clean = self._encode_clean_block(canvas[:, :prompt_len], 0, genre_id, None) if prompt_len > 0 else None
            curr_pos = prompt_len
            
            pbar = tqdm(total=SEQ_LEN, initial=prompt_len, desc="Generating Completion")
            while curr_pos < SEQ_LEN:
                L = min(BLOCK_SIZE, SEQ_LEN - curr_pos)
                if L == 0: break
                
                tqdm.write(f"\nProcessing Completion Block at Pos {curr_pos} (Size: {L})...")
                canvas_block = torch.full((1, L), PAD_ID, device=DEVICE, dtype=torch.long)
                
                # In completion, ALL tokens are allowed to be masked and generated
                allowed_mask = torch.ones((1, L), dtype=torch.bool, device=DEVICE)
                ratios = [1.0, 0.75, 0.50, 0.25]
                
                clean_block = self._denoise_block_ar(
                    canvas_block, allowed_mask, past_kv_clean, curr_pos, genre_id, ratios, temperature
                )
                canvas[:, curr_pos:curr_pos+L] = clean_block
                
                # Early Stopping Check
                e_id = self.tokenizer.tok_to_id.get('<E>', -1)
                if e_id >= 0 and (clean_block[0] == e_id).any():
                    e_pos = (clean_block[0] == e_id).nonzero(as_tuple=True)[0]
                    if len(e_pos) > 0:
                        final_seq_len = curr_pos + e_pos[0].item()
                        tqdm.write(f"End token <E> encountered at {final_seq_len}. Stopping generation.")
                        pbar.update(final_seq_len - curr_pos)
                        pbar.close()
                        break
                
                past_kv_clean = self._encode_clean_block(clean_block, curr_pos, genre_id, past_kv_clean)
                curr_pos += L
                pbar.update(L)
                
            if pbar.n < SEQ_LEN: pbar.close()

        else:
            curr_pos = 0
            past_kv_clean = None
            actual_seq_len = (canvas[0] != PAD_ID).sum().item()
            final_seq_len = actual_seq_len
            
            pbar = tqdm(total=actual_seq_len, desc=f"Style Transfer ({strategy})")
            while curr_pos < actual_seq_len:
                L = min(BLOCK_SIZE, actual_seq_len - curr_pos)
                if L == 0: break
                
                tqdm.write(f"\nApplying Style Transfer [{strategy}] at Pos {curr_pos}...")
                canvas_block = canvas[:, curr_pos:curr_pos+L].clone()
                valid_len = (canvas_block[0] != PAD_ID).sum().item()
                
                # 1D targeted masking requires decoding to understand structure
                tokens_text = self.processor.tensor_to_tokens(canvas_block[0, :valid_len])
                is_rhythm = torch.zeros((1, L), dtype=torch.bool, device=DEVICE)
                is_pitch = torch.zeros((1, L), dtype=torch.bool, device=DEVICE)
                
                for i, tok in enumerate(tokens_text):
                    if isinstance(tok, tuple):
                        if tok[0] in ['onset', 'dur']:
                            is_rhythm[0, i] = True
                        elif tok[0] not in ['prefix']: # Exclude purely structural tokens
                            is_pitch[0, i] = True
                            
                # Protect <T> tokens
                is_t = (canvas_block == t_id)
                is_rhythm &= ~is_t
                is_pitch &= ~is_t
                
                if strategy == "rhythm_transfer":
                    ratios = [1.0, 0.75, 0.50, 0.25]
                    clean_block = self._denoise_block_ar(
                        canvas_block, is_rhythm, past_kv_clean, curr_pos, genre_id, ratios, temperature
                    )
                elif strategy == "rhythm_then_pitch":
                    tqdm.write("  Stage 1: Rhythm Generation")
                    rhythm_ratios = [1.0, 0.50]
                    rhythm_fixed = self._denoise_block_ar(
                        canvas_block, is_rhythm, past_kv_clean, curr_pos, genre_id, rhythm_ratios, temperature
                    )
                    tqdm.write("  Stage 2: Pitch Generation")
                    pitch_ratios = [1.0, 0.50]
                    clean_block = self._denoise_block_ar(
                        rhythm_fixed, is_pitch, past_kv_clean, curr_pos, genre_id, pitch_ratios, temperature
                    )
                
                canvas[:, curr_pos:curr_pos+L] = clean_block
                past_kv_clean = self._encode_clean_block(clean_block, curr_pos, genre_id, past_kv_clean)
                
                curr_pos += L
                pbar.update(L)
            
            pbar.close()

        # Final filtering of PAD and <S> tokens
        final_tensor = canvas[0, :final_seq_len]
        valid_indices = (final_tensor != PAD_ID)
        s_id = self.tokenizer.tok_to_id.get('<S>', -1)
        if s_id >= 0:
            valid_indices &= (final_tensor != s_id)

        return self.processor.tensor_to_tokens(final_tensor[valid_indices])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/gpfs/scratch/acw769/improvnet/artifacts/caddi_diffusion/small/best_model.pt")
    parser.add_argument("--input", type=str, default="/data/home/acw769/improvnet_2/improvnet/inference/debussy-clair-de-lune_original.mid")
    parser.add_argument("--output", type=str, default="/data/home/acw769/improvnet_2/improvnet/inference/generated.mid")
    parser.add_argument("--genre", type=str, default="jazz")
    parser.add_argument("--strategy", type=str, default="rhythm_transfer", choices=["completion", "rhythm_transfer", "rhythm_then_pitch"])
    parser.add_argument("--keep_prompt_len", type=int, default=128, help="Number of prompt tokens to keep for completion strategy")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    inference = CaDDiInference(model_path=args.model)
    processor = ProcessData()
    
    print(f"Reading {args.input}...")
    source_midi = processor.read_midi(args.input)
    source_tokens = processor.midi_to_tokens(source_midi)
    
    print(f"Generating {args.genre} in Pure AR Refinement mode...")
    new_tokens = inference.generate(
        prompt_tokens=source_tokens,
        strategy=args.strategy,
        genre_str=args.genre,
        keep_prompt_len=args.keep_prompt_len,
        temperature=args.temperature
    )
    
    print(f"Saving to {args.output}...")
    new_midi = processor.tokens_to_midi(new_tokens)
    processor.save_midi(new_midi, args.output)
    print("Done!")