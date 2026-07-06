import math
import torch
import torch.nn.functional as F
from improvnet.model.config import *
from improvnet.model.gidd_model import PrefixARModel
from improvnet.utils.gidd_utils import ProcessData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _amp_dtype():
    if not torch.cuda.is_available(): return None
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16

AMP_DTYPE = _amp_dtype()

def cosine_schedule(step: int, total_steps: int) -> float:
    """
    Returns the fraction of tokens that should REMAIN masked at the current step.
    step 1 = ~0.99 masked, step 12 = 0.0 masked.
    """
    return math.cos((math.pi / 2) * (step / total_steps))

@torch.no_grad()
def sample_and_score(gru_decoder, h_block, temperature=1.0, top_k=50, forbidden_ids=None):
    """
    Custom GRU sampler that returns both the sampled tokens AND their softmax confidence scores.
    forbidden_ids: A list of 5 token IDs (one for each attribute) that the model is NOT allowed to generate.
    """
    B, T, D = h_block.shape
    h_gru = h_block.reshape(B * T, D)
    
    sampled_attrs = []
    confidences = []
    
    gru_in = gru_decoder.start_emb.expand(B * T, -1)
    h_gru = gru_decoder.gru(gru_in, h_gru)
    
    for i in range(gru_decoder.num_attrs):
        if i > 0:
            prev_attr = sampled_attrs[-1].view(B * T)
            gru_in = gru_decoder.embs[i-1](prev_attr)
            h_gru = gru_decoder.gru(gru_in, h_gru)
            
        logits = gru_decoder.out_heads[i](h_gru) / max(temperature, 1e-5)
        logits = logits.view(B, T, -1)
            
        # Static Forbidden IDs (protects <T> tokens)
        if forbidden_ids is not None and forbidden_ids[i] is not None:
            logits[:, :, forbidden_ids[i]] = float('-inf')
            
        logits = logits.view(B * T, -1)
        
        if top_k > 0:
            val, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits = torch.where(logits < val[:, -1:], float('-inf'), logits)
            
        probs = F.softmax(logits, dim=-1)
        samp = torch.multinomial(probs, num_samples=1)
        conf = probs.gather(-1, samp).squeeze(-1) 
        
        sampled_attrs.append(samp)
        confidences.append(conf)
        
    sampled_tokens = torch.stack(sampled_attrs, dim=-1).view(B, T, 5)
    confidence_scores = torch.stack(confidences, dim=-1).view(B, T, 5)
    
    return sampled_tokens, confidence_scores


class GIDDInference:
    def __init__(self, model_path: str):
        self.processor = ProcessData()
        self.tokenizer = self.processor.tokenizer
        
        print("Loading GIDD Diffusion Model...")
        self.model = PrefixARModel().to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.t_ids = [
            self.tokenizer.tok_to_id_instrument.get('<T>'),
            self.tokenizer.tok_to_id_pitch.get('<T>'),
            self.tokenizer.tok_to_id_velocity.get('<T>'),
            self.tokenizer.tok_to_id_onset.get('<T>'),
            self.tokenizer.tok_to_id_duration.get('<T>')
        ]

    def _denoise_block(
        self, block, initial_mask, genre_id, past_key_values, start_offset, 
        diffusion_steps, temperature, forbidden, unmasked_edit_threshold, allowed_edit_attrs,
        valid_len 
    ):
        """Standard 12-step discrete diffusion loop with GIDD unmasked-token refinement."""
        
        cumulative_unmasked_edits = torch.zeros_like(initial_mask)
        attr_names = {0: 'Instrument', 1: 'Pitch', 2: 'Velocity', 3: 'Onset', 4: 'Duration'}
        
        # Boundary constraint: Never allow edits on padding tokens
        is_valid_pos = torch.arange(block.shape[1], device=DEVICE) < valid_len
        
        for step in range(1, diffusion_steps + 1):
            timestep_val = 1.0 - (step / diffusion_steps) 
            t_tensor = torch.full((1,), timestep_val, device=DEVICE)
            
            with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                h_block, _ = self.model(
                    target=block, genre=genre_id, prefix_len=0, timestep=t_tensor,
                    use_cache=True, past_key_values=past_key_values, seq_offset=start_offset
                )
                
                sampled_tokens, confidences = sample_and_score(
                    self.model.gru_decoder, h_block, temperature, top_k=50, forbidden_ids=forbidden
                )
            
            raw_sampled_tokens = sampled_tokens.clone()
            raw_confidences = confidences.clone()
            
            # 1. Ground Truth Enforcement (Initially clean tokens remain locked by default)
            sampled_tokens[~initial_mask] = block[~initial_mask]
            confidences[~initial_mask] = float('inf') 
            
            # 2. GIDD Unmasked Token Editing (Refine BOTH classical ground truth AND previously generated tokens)
            if unmasked_edit_threshold < 1.0 and allowed_edit_attrs:
                for attr in allowed_edit_attrs:
                    # Target ANY token that is currently visible to the model (not masked)
                    is_currently_unmasked = block[0, :, attr] != MASK_ID
                    
                    is_different = raw_sampled_tokens[0, :, attr] != block[0, :, attr]
                    is_confident = raw_confidences[0, :, attr] >= unmasked_edit_threshold
                    is_not_t = block[0, :, attr] != self.t_ids[attr]
                    
                    # Apply boundary mask to completely ignore PAD_ID slots
                    valid_edits = is_currently_unmasked & is_different & is_confident & is_not_t & is_valid_pos
                    
                    if valid_edits.any():
                        # Override the locked value with the model's new confident prediction
                        sampled_tokens[0, valid_edits, attr] = raw_sampled_tokens[0, valid_edits, attr]
                        cumulative_unmasked_edits[0, valid_edits, attr] = True
                        
                        # Immediately update the block so this becomes the new context for future steps
                        block[0, valid_edits, attr] = raw_sampled_tokens[0, valid_edits, attr]
            
            # 3. COSINE UNMASKING (For tokens still holding MASK_ID)
            ratio_to_mask = cosine_schedule(step, diffusion_steps)
            for attr_idx in range(5):
                current_masked = (block[0, :, attr_idx] == MASK_ID)
                num_currently_masked = current_masked.sum().item()
                
                total_editable = initial_mask[0, :, attr_idx].sum().item()
                target_num_masked = int(ratio_to_mask * total_editable)
                
                num_to_reveal = num_currently_masked - target_num_masked
                
                if num_to_reveal > 0:
                    attr_confs = confidences[0, :, attr_idx].clone()
                    attr_confs[~current_masked] = float('-inf')
                    
                    actual_reveal = min(num_to_reveal, num_currently_masked)
                    _, top_indices = torch.topk(attr_confs, actual_reveal)
                    
                    block[0, top_indices, attr_idx] = sampled_tokens[0, top_indices, attr_idx]
                    
        # 4. Summary Print
        if unmasked_edit_threshold < 1.0:
            summary_strs = []
            for attr in allowed_edit_attrs:
                num_edits = cumulative_unmasked_edits[0, :, attr].sum().item()
                if num_edits > 0:
                    summary_strs.append(f"{attr_names[attr]}: {num_edits}")
            if summary_strs:
                print(f"    -> Unmasked Edits Evaluated (Clean + Refinements): {', '.join(summary_strs)}")
                
        return block

    @torch.no_grad()
    def generate(
        self, 
        prompt_tokens: list, 
        strategy: str = "rhythm_then_pitch", 
        genre_str: str = "jazz", 
        diffusion_steps: int = 12, 
        temperature: float = 1.0,
        mask_percentage: float = 0.30,
        pitch_mask_ratio: float = 0.15,
        unmasked_edit_threshold: float = 0.90
    ) -> list:
        
        genre_id = torch.tensor([self.processor.get_genre_id(genre_str)], dtype=torch.long, device=DEVICE)
        prompt_len = min(len(prompt_tokens), SEQ_LEN)
        
        canvas = torch.full((1, SEQ_LEN, 5), PAD_ID, dtype=torch.long, device=DEVICE)
        prompt_tensor = self.processor.format_variable_sequence(prompt_tokens, prompt_len).unsqueeze(0).to(DEVICE)
        canvas[:, :prompt_len] = prompt_tensor

        num_blocks = math.ceil(prompt_len / BLOCK_SIZE)
        past_key_values = None
        forbidden = self.t_ids

        for b in range(num_blocks):
            start = b * BLOCK_SIZE
            end = start + BLOCK_SIZE
            block = canvas[:, start:end].clone()
            
            valid_len = (block[0, :, 0] != PAD_ID).sum().item()
            if valid_len == 0: continue
            
            initial_mask = torch.zeros_like(block, dtype=torch.bool)
            
            print(f"\nDenoising Block {b} ({strategy})...")
            
            # ====================================================================
            # STRATEGY 1: RHYTHM ONLY
            # ====================================================================
            if strategy == "rhythm_only":
                initial_mask[:, :valid_len, 3] = True # Onset
                initial_mask[:, :valid_len, 4] = True # Duration
                
                # Protect <T> tokens
                for attr in [3, 4]:
                    is_t = (block[0, :, attr] == self.t_ids[attr])
                    initial_mask[0, is_t, attr] = False
                    
                block[initial_mask] = MASK_ID
                block = self._denoise_block(
                    block, initial_mask, genre_id, past_key_values, start, 
                    diffusion_steps, temperature, forbidden, 
                    unmasked_edit_threshold, allowed_edit_attrs=[3, 4],
                    valid_len=valid_len
                )

            # ====================================================================
            # STRATEGY 2: RHYTHM THEN PITCH (Two-Stage Diffusion)
            # ====================================================================
            elif strategy == "rhythm_then_pitch":
                # Stage 1: Lock in the Rhythm
                print(f"  Stage 1: Rhythm Generation")
                rhythm_mask = torch.zeros_like(block, dtype=torch.bool)
                rhythm_mask[:, :valid_len, 3] = True
                rhythm_mask[:, :valid_len, 4] = True
                for attr in [3, 4]:
                    is_t = (block[0, :, attr] == self.t_ids[attr])
                    rhythm_mask[0, is_t, attr] = False
                    
                block[rhythm_mask] = MASK_ID
                block = self._denoise_block(
                    block, rhythm_mask, genre_id, past_key_values, start, 
                    diffusion_steps, temperature, forbidden, 
                    unmasked_edit_threshold, allowed_edit_attrs=[3, 4],
                    valid_len=valid_len
                )
                
                # Stage 2: Critique the harmony against the new rhythm, then denoise Pitch/Velocity
                print(f"  Stage 2: Pitch & Velocity Generation")
                pitch_vel_mask = torch.zeros_like(block, dtype=torch.bool)
                
                dummy_t = torch.zeros(1, device=DEVICE)
                with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                    h_block, _ = self.model(
                        target=block, genre=genre_id, prefix_len=block.shape[1], 
                        timestep=dummy_t, use_cache=True, past_key_values=past_key_values, seq_offset=start
                    )
                    _, confidences = sample_and_score(self.model.gru_decoder, h_block)
                
                for attr in [1, 2]: # Pitch and Velocity
                    confs = confidences[0, :valid_len, attr].clone()
                    is_t = (block[0, :valid_len, attr] == self.t_ids[attr])
                    confs[is_t] = float('inf') # Don't mask <T> tokens
                    
                    num_to_mask = int(pitch_mask_ratio * valid_len)
                    if num_to_mask > 0:
                        _, worst_idx = torch.topk(confs, num_to_mask, largest=False)
                        pitch_vel_mask[0, worst_idx, attr] = True
                        
                block[pitch_vel_mask] = MASK_ID
                block = self._denoise_block(
                    block, pitch_vel_mask, genre_id, past_key_values, start, 
                    diffusion_steps, temperature, forbidden, 
                    unmasked_edit_threshold, allowed_edit_attrs=[1, 2],
                    valid_len=valid_len
                )

            # ====================================================================
            # STRATEGY 3: STANDARD RANDOM MASKING
            # ====================================================================
            elif strategy == "standard_masking":
                num_to_mask = int(mask_percentage * valid_len)
                for attr in [1, 2, 3, 4]:
                    valid_indices = torch.arange(valid_len, device=DEVICE)
                    is_t = (block[0, :valid_len, attr] == self.t_ids[attr])
                    valid_indices = valid_indices[~is_t]
                    
                    if len(valid_indices) > 0:
                        actual_mask = min(num_to_mask, len(valid_indices))
                        perm = torch.randperm(len(valid_indices))[:actual_mask]
                        initial_mask[0, valid_indices[perm], attr] = True
                        
                block[initial_mask] = MASK_ID
                block = self._denoise_block(
                    block, initial_mask, genre_id, past_key_values, start, 
                    diffusion_steps, temperature, forbidden, 
                    unmasked_edit_threshold, allowed_edit_attrs=[0, 1, 2, 3, 4],
                    valid_len=valid_len
                )
                
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            canvas[:, start:end] = block
            
            # Causal KV Cache Update for the next block
            dummy_t = torch.zeros(1, device=DEVICE)
            with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                _, past_key_values = self.model(
                    target=block, genre=genre_id, prefix_len=block.shape[1], timestep=dummy_t,
                    use_cache=True, past_key_values=past_key_values, seq_offset=start
                )
            
        denoised_tensor_dict = {
            'instrument': canvas[0, :, 0],
            'pitch': canvas[0, :, 1],
            'velocity': canvas[0, :, 2],
            'onset': canvas[0, :, 3],
            'duration': canvas[0, :, 4]
        }
        
        valid_indices = (canvas[0, :, 0] != PAD_ID)
        for k in denoised_tensor_dict:
            denoised_tensor_dict[k] = denoised_tensor_dict[k][valid_indices]

        return self.processor.tensor_to_tokens(denoised_tensor_dict)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/gpfs/scratch/acw769/improvnet/artifacts/block_diffusion_gidd/latest_checkpoint.pt")
    parser.add_argument("--input", type=str, required=True, default="/data/home/acw769/improvnet_2/improvnet/inference/debussy-clair-de-lune_original.mid")
    parser.add_argument("--output", type=str, default="/data/home/acw769/improvnet_2/improvnet/inference/generated.mid")
    parser.add_argument("--genre", type=str, default="jazz")
    parser.add_argument("--strategy", type=str, default="rhythm_then_pitch", choices=["rhythm_only", "rhythm_then_pitch", "standard_masking"])
    parser.add_argument("--mask_percentage", type=float, default=0.30, help="For standard_masking strategy")
    parser.add_argument("--pitch_mask_ratio", type=float, default=0.15, help="For rhythm_then_pitch strategy")
    parser.add_argument("--unmasked_edit_threshold", type=float, default=0.9, help="Confidence threshold to allow editing non-masked tokens (1.0 disables)")
    args = parser.parse_args()

    inference = GIDDInference(model_path=args.model)
    processor = ProcessData()
    
    print(f"Reading {args.input}...")
    source_midi = processor.read_midi(args.input)
    source_tokens = processor.midi_to_tokens(source_midi)
    
    print(f"Generating {args.genre} in {args.strategy} mode...")
    new_tokens = inference.generate(
        prompt_tokens=source_tokens,
        strategy=args.strategy,
        genre_str=args.genre,
        diffusion_steps=128,
        temperature=1.0,
        mask_percentage=args.mask_percentage,
        pitch_mask_ratio=args.pitch_mask_ratio,
        unmasked_edit_threshold=args.unmasked_edit_threshold
    )
    
    print(f"Saving to {args.output}...")
    new_midi = processor.tokens_to_midi(new_tokens)
    processor.save_midi(new_midi, args.output)
    print("Done!")