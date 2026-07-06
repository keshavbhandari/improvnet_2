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

class GIDDCompletionInference:
    def __init__(self, model_path: str):
        self.processor = ProcessData()
        self.tokenizer = self.processor.tokenizer
        
        print("Loading GIDD Diffusion Model for Completion...")
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
        """Standard 12-step discrete diffusion loop with GIDD clean-token editing."""
        
        cumulative_clean_edits = torch.zeros_like(initial_mask)
        attr_names = {0: 'Instrument', 1: 'Pitch', 2: 'Velocity', 3: 'Onset', 4: 'Duration'}
        
        # Boundary constraint: Never allow edits on padding tokens
        is_valid_pos = torch.arange(block.shape[1], device=DEVICE) < valid_len
        
        # Pre-calculate how many column positions to reveal per step (e.g., 256 / 128 = 2 tokens per step)
        positions_per_step = math.ceil(valid_len / diffusion_steps)
        
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
            
            # 1. Ground Truth Enforcement (Clean tokens remain locked by default)
            sampled_tokens[~initial_mask] = block[~initial_mask]
            confidences[~initial_mask] = float('inf') 
            
            # 2. GIDD Clean Token Editing (Override enforcement for highly confident corrections)
            if unmasked_edit_threshold < 1.0 and allowed_edit_attrs:
                for attr in allowed_edit_attrs:
                    is_currently_unmasked = block[0, :, attr] != MASK_ID
                    
                    is_different = raw_sampled_tokens[0, :, attr] != block[0, :, attr]
                    is_confident = raw_confidences[0, :, attr] >= unmasked_edit_threshold
                    is_not_t = block[0, :, attr] != self.t_ids[attr]
                    
                    # Apply boundary mask to completely ignore PAD_ID slots
                    valid_edits = is_currently_unmasked & is_different & is_confident & is_not_t & is_valid_pos
                    
                    if valid_edits.any():
                        # Override the locked value with the model's new confident prediction
                        sampled_tokens[0, valid_edits, attr] = raw_sampled_tokens[0, valid_edits, attr]
                        cumulative_clean_edits[0, valid_edits, attr] = True
                        
                        # Immediately update the block so this becomes the new context for future steps
                        block[0, valid_edits, attr] = raw_sampled_tokens[0, valid_edits, attr]
            
            # 3. LEFT-TO-RIGHT UNMASKING (Autoregressive-style within block)
            # Find all column positions that currently have at least one MASK_ID
            current_masked_positions = (block[0, :valid_len, :] == MASK_ID).any(dim=-1).nonzero(as_tuple=True)[0]
            
            if len(current_masked_positions) > 0:
                # Take the leftmost N positions
                positions_to_reveal = current_masked_positions[:positions_per_step]
                
                # Unmask them unconditionally
                for pos in positions_to_reveal:
                    for attr_idx in range(5):
                        if block[0, pos, attr_idx] == MASK_ID:
                            block[0, pos, attr_idx] = sampled_tokens[0, pos, attr_idx]
                    
        # 4. Summary Print
        if unmasked_edit_threshold < 1.0:
            summary_strs = []
            for attr in allowed_edit_attrs:
                num_edits = cumulative_clean_edits[0, :, attr].sum().item()
                if num_edits > 0:
                    summary_strs.append(f"{attr_names[attr]}: {num_edits}")
            if summary_strs:
                print(f"    -> Non-Masked Edits Evaluated (Clean + Refinements): {', '.join(summary_strs)}")
                
        return block

    @torch.no_grad()
    def generate(
        self, 
        prompt_tokens: list, 
        genre_str: str = "jazz", 
        keep_prompt_len: int = 128,
        max_tokens: int = 2048,
        diffusion_steps: int = 12, 
        temperature: float = 1.0,
        unmasked_edit_threshold: float = 0.90
    ) -> list:
        
        genre_id = torch.tensor([self.processor.get_genre_id(genre_str)], dtype=torch.long, device=DEVICE)
        
        # Determine actual prompt length to retain (locked context)
        prompt_len = min(len(prompt_tokens), keep_prompt_len, max_tokens)
        
        # Load the ENTIRE sequence so we can extract its <T> tokens later
        actual_seq_len = min(len(prompt_tokens), max_tokens)
        
        canvas = torch.full((1, max_tokens, 5), PAD_ID, dtype=torch.long, device=DEVICE)
        full_tensor = self.processor.format_variable_sequence(prompt_tokens, actual_seq_len).unsqueeze(0).to(DEVICE)
        canvas[:, :actual_seq_len] = full_tensor

        num_blocks = math.ceil(actual_seq_len / BLOCK_SIZE)
        past_key_values = None
        forbidden = self.t_ids
        
        # Token ID for early stopping
        e_id = self.tokenizer.tok_to_id_instrument.get('<E>')
        final_seq_len = actual_seq_len

        for b in range(num_blocks):
            start = b * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, max_tokens)
            block_len = end - start
            
            block = canvas[:, start:end].clone()
            
            # How many valid (non-padding) tokens are actually in this block?
            valid_len = min(block_len, max(0, actual_seq_len - start))
            if valid_len == 0: break
            
            # Create mask for the unknown generation region
            initial_mask = torch.zeros_like(block, dtype=torch.bool)
            gen_start_in_block = max(0, prompt_len - start)
            
            if gen_start_in_block < valid_len:
                initial_mask[:, gen_start_in_block:valid_len, :] = True
                
            # Protect ALL <T> tokens in the valid region from being masked
            # This preserves the structural grid of the entire original piece!
            for attr in range(5):
                is_t = (block[0, :valid_len, attr] == self.t_ids[attr])
                initial_mask[0, :valid_len, attr][is_t] = False

            block[initial_mask] = MASK_ID
            
            if initial_mask.any():
                print(f"\nGenerating Completion Block {b}...")
                block = self._denoise_block(
                    block, initial_mask, genre_id, past_key_values, start, 
                    diffusion_steps, temperature, forbidden, 
                    unmasked_edit_threshold, allowed_edit_attrs=[0, 1, 2, 3, 4],
                    valid_len=valid_len
                )
            else:
                print(f"\nEncoding Prompt Block {b}...")

            canvas[:, start:end] = block
            
            # Check for <E> token in the newly completed block
            if e_id is not None:
                e_positions = (block[0, :, 0] == e_id).nonzero(as_tuple=True)[0]
                if len(e_positions) > 0:
                    first_e = e_positions[0].item()
                    final_seq_len = start + first_e
                    print(f"End token <E> encountered at position {final_seq_len}. Stopping generation.")
                    break
            
            # Causal KV Cache Update for the next block
            dummy_t = torch.zeros(1, device=DEVICE)
            with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                _, past_key_values = self.model(
                    target=block, genre=genre_id, prefix_len=block_len, timestep=dummy_t,
                    use_cache=True, past_key_values=past_key_values, seq_offset=start
                )
            
        denoised_tensor_dict = {
            'instrument': canvas[0, :final_seq_len, 0],
            'pitch': canvas[0, :final_seq_len, 1],
            'velocity': canvas[0, :final_seq_len, 2],
            'onset': canvas[0, :final_seq_len, 3],
            'duration': canvas[0, :final_seq_len, 4]
        }
        
        # Filter out padding and <S> start tokens before detokenization
        valid_indices = (denoised_tensor_dict['instrument'] != PAD_ID)
        s_id = self.tokenizer.tok_to_id_instrument.get('<S>')
        if s_id is not None:
            valid_indices &= (denoised_tensor_dict['instrument'] != s_id)
            
        for k in denoised_tensor_dict:
            denoised_tensor_dict[k] = denoised_tensor_dict[k][valid_indices]

        return self.processor.tensor_to_tokens(denoised_tensor_dict)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/gpfs/scratch/acw769/improvnet/artifacts/block_diffusion_gidd/latest_checkpoint.pt")
    parser.add_argument("--input", type=str, required=True, default="/data/home/acw769/improvnet_2/improvnet/inference/debussy-clair-de-lune_original.mid")
    parser.add_argument("--output", type=str, default="/data/home/acw769/improvnet_2/improvnet/inference/completion.mid")
    parser.add_argument("--genre", type=str, default="classical")
    parser.add_argument("--keep_prompt_len", type=int, default=128, help="Number of tokens from input MIDI to use as context prompt.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum length of the generated sequence.")
    parser.add_argument("--unmasked_edit_threshold", type=float, default=0.90, help="Confidence threshold to allow editing the prompt tokens (1.0 disables)")
    args = parser.parse_args()

    inference = GIDDCompletionInference(model_path=args.model)
    processor = ProcessData()
    
    print(f"Reading {args.input}...")
    source_midi = processor.read_midi(args.input)
    source_tokens = processor.midi_to_tokens(source_midi)
    
    print(f"Generating {args.genre} completion...")
    new_tokens = inference.generate(
        prompt_tokens=source_tokens,
        genre_str=args.genre,
        keep_prompt_len=args.keep_prompt_len,
        max_tokens=args.max_tokens,
        diffusion_steps=128,
        temperature=1.0,
        unmasked_edit_threshold=args.unmasked_edit_threshold
    )
    
    print(f"Saving to {args.output}...")
    new_midi = processor.tokens_to_midi(new_tokens)
    processor.save_midi(new_midi, args.output)
    print("Done!")