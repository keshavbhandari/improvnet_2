import math
import torch
import torch.nn.functional as F
from improvnet.model.caddi_config import *
from improvnet.model.caddi_model import CaDDiModel
from improvnet.utils.utils import ProcessData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _amp_dtype():
    if not torch.cuda.is_available(): return None
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16

AMP_DTYPE = _amp_dtype()

@torch.no_grad()
def sample_and_score(gru_decoder, h_transformer, temperature=1.0, top_k=50):
    """
    Autoregressive GRU sampler that returns both the sampled tokens AND their softmax confidence scores.
    """
    B, T, D = h_transformer.shape
    h_gru = h_transformer.reshape(B * T, D)
    
    sampled_attrs = []
    confidences = []
    
    # --- Step 0: Instrument ---
    gru_in = gru_decoder.start_emb.expand(B * T, -1)
    h_gru = gru_decoder.gru(gru_in, h_gru)
    logits = gru_decoder.out_heads[0](h_gru) / max(temperature, 1e-5)
        
    if top_k > 0:
        val, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        logits = torch.where(logits < val[:, -1:], float('-inf'), logits)
        
    probs = F.softmax(logits, dim=-1)
    samp = torch.multinomial(probs, num_samples=1)
    conf = probs.gather(-1, samp).squeeze(-1)
    
    sampled_attrs.append(samp)
    confidences.append(conf)

    # --- Steps 1 to 4: Pitch, Velocity, Onset, Duration ---
    for i in range(1, gru_decoder.num_attrs):
        prev_attr = sampled_attrs[-1].view(B * T)
        gru_in = gru_decoder.embs[i-1](prev_attr)
        h_gru = gru_decoder.gru(gru_in, h_gru)
        logits = gru_decoder.out_heads[i](h_gru) / max(temperature, 1e-5)
            
        if top_k > 0:
            val, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits = torch.where(logits < val[:, -1:], float('-inf'), logits)
            
        probs = F.softmax(logits, dim=-1)
        samp = torch.multinomial(probs, num_samples=1)
        conf = probs.gather(-1, samp).squeeze(-1)
        
        sampled_attrs.append(samp)
        confidences.append(conf)
        
    # Reshape back to [B, T, 5]
    sampled_tokens = torch.stack(sampled_attrs, dim=-1).view(B, T, 5)
    confidence_scores = torch.stack(confidences, dim=-1).view(B, T, 5)
    
    return sampled_tokens, confidence_scores

class CaDDiInference:
    def __init__(self, model_path: str):
        self.processor = ProcessData()
        self.tokenizer = self.processor.tokenizer
        
        print("Loading CaDDi AR Diffusion Model...")
        self.model = CaDDiModel().to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _encode_clean_block(self, block, start_pos, genre_id, past_kv):
        """Passes a fully finalized clean block into the model simply to update the causal KV cache for future blocks."""
        B, L, _ = block.shape
        coords_pos = torch.arange(start_pos, start_pos + L, device=DEVICE).unsqueeze(0).unsqueeze(-1)
        coords = torch.cat([coords_pos, block], dim=-1)
        timestep = torch.zeros(B, L, device=DEVICE) # Clean blocks always have timestep 0.0
        
        with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
            _, next_kv = self.model(
                target=block, coords=coords, timestep=timestep, genre=genre_id, 
                use_cache=True, past_key_values=past_kv
            )
        return next_kv

    def _denoise_block(self, canvas_block, past_kv_clean, start_pos, genre_id, strategy):
        B, L, _ = canvas_block.shape
        
        # 4-Step Masking Schedules defined dynamically based on strategy!
        if strategy == "completion":
            mask_sched = [([0,1,2,3,4], 1.0), ([0,1,2,3,4], 0.75), ([0,1,2,3,4], 0.50), ([0,1,2,3,4], 0.25)]
        elif strategy == "rhythm_transfer":
            mask_sched = [([3,4], 1.0), ([3,4], 0.75), ([3,4], 0.50), ([3,4], 0.25)]
        elif strategy == "rhythm_then_pitch":
            # Stage 1: 2 Steps of Rhythm. Stage 2: 2 Steps of Pitch.
            mask_sched = [([3,4], 1.0), ([3,4], 0.50), ([1,2], 1.0), ([1,2], 0.50)]
        else:
            raise ValueError(f"Unknown Strategy: {strategy}")
            
        time_vals = [1.0, 0.75, 0.50, 0.25]
        
        current_block = canvas_block.clone()
        kv_accum = past_kv_clean # Starts with the clean context of prior blocks
        prev_confs = None
        
        # RoPE coordinates advance linearly across drafts to prevent coordinate collision!
        draft_pos = start_pos
        
        for step in range(4):
            attrs, ratio = mask_sched[step]
            x_input = current_block.clone()
            
            for attr in attrs:
                num_to_mask = int(ratio * L)
                if num_to_mask == 0: continue
                
                valid_indices = torch.arange(L, device=DEVICE)
                actual_mask_count = min(num_to_mask, len(valid_indices))
                
                if prev_confs is None or ratio == 1.0:
                    # Random uniform masking for the very first pass
                    perm = torch.randperm(len(valid_indices))
                    chosen = valid_indices[perm[:actual_mask_count]]
                else:
                    # Lowest confidence masking for refinement passes
                    confs = prev_confs[0, valid_indices, attr]
                    _, worst_idx = torch.topk(confs, actual_mask_count, largest=False)
                    chosen = valid_indices[worst_idx]
                    
                x_input[0, chosen, attr] = MASK_ID
                
            coords_pos = torch.arange(draft_pos, draft_pos + L, device=DEVICE).unsqueeze(0).unsqueeze(-1)
            coords = torch.cat([coords_pos, x_input], dim=-1)
            timestep = torch.full((B, L), time_vals[step], device=DEVICE)
            
            # Massive advantage: KV Cache retains the previous noisy drafts (x4, x3, etc.) allowing self-critique!
            with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                h_target, kv_accum = self.model(
                    target=x_input, coords=coords, timestep=timestep, genre=genre_id, 
                    use_cache=True, past_key_values=kv_accum
                )
                
            sampled_tokens, confs = sample_and_score(self.model.gru_decoder, h_target)
            
            # Update ONLY the masked tokens, preserving the ground truth or previously generated clean tokens!
            mask = (x_input == MASK_ID)
            current_block[mask] = sampled_tokens[mask]
            prev_confs = confs
            
            # Advance RoPE sequence position for the next draft!
            draft_pos += L
            
        return current_block

    def generate(
        self, 
        prompt_tokens: list, 
        strategy: str = "rhythm_then_pitch", 
        genre_str: str = "jazz", 
        keep_prompt_len: int = 256
    ) -> list:
        
        genre_id = torch.tensor([self.processor.get_genre_id(genre_str)], dtype=torch.long, device=DEVICE)
        
        if strategy == "completion":
            prompt_len = min(len(prompt_tokens), keep_prompt_len)
        else:
            prompt_len = min(len(prompt_tokens), SEQ_LEN)
            
        canvas = torch.full((1, SEQ_LEN, 5), PAD_ID, dtype=torch.long, device=DEVICE)
        if prompt_len > 0:
            prompt_tensor = self.processor.format_variable_sequence(prompt_tokens[:prompt_len], prompt_len).unsqueeze(0).to(DEVICE)
            canvas[:, :prompt_len] = prompt_tensor

        final_seq_len = SEQ_LEN

        if strategy == "completion":
            print(f"Encoding {prompt_len} Prompt Tokens...")
            past_kv_clean = self._encode_clean_block(canvas[:, :prompt_len], 0, genre_id, None) if prompt_len > 0 else None
            curr_pos = prompt_len
            
            while curr_pos < SEQ_LEN:
                L = min(BLOCK_SIZE, SEQ_LEN - curr_pos)
                if L == 0: break
                
                print(f"Generating Completion Block at Pos {curr_pos}...")
                canvas_block = torch.full((1, L, 5), PAD_ID, device=DEVICE, dtype=torch.long)
                clean_block = self._denoise_block(canvas_block, past_kv_clean, curr_pos, genre_id, strategy)
                canvas[:, curr_pos:curr_pos+L] = clean_block
                
                # Auto-regressive Early Stopping Check
                e_id = self.tokenizer.tok_to_id_instrument.get('<E>')
                if e_id is not None:
                    e_pos = (clean_block[0, :, 0] == e_id).nonzero(as_tuple=True)[0]
                    if len(e_pos) > 0:
                        final_seq_len = curr_pos + e_pos[0].item()
                        print(f"End token <E> encountered at {final_seq_len}. Stopping generation.")
                        break
                
                # Encode the newly finalized block to act as prompt for the next block
                past_kv_clean = self._encode_clean_block(clean_block, curr_pos, genre_id, past_kv_clean)
                curr_pos += L

        else:
            curr_pos = 0
            past_kv_clean = None
            actual_seq_len = (canvas[0, :, 0] != PAD_ID).sum().item()
            final_seq_len = actual_seq_len
            
            while curr_pos < actual_seq_len:
                L = min(BLOCK_SIZE, actual_seq_len - curr_pos)
                if L == 0: break
                
                print(f"Applying Style Transfer [{strategy}] at Pos {curr_pos}...")
                canvas_block = canvas[:, curr_pos:curr_pos+L].clone()
                clean_block = self._denoise_block(canvas_block, past_kv_clean, curr_pos, genre_id, strategy)
                canvas[:, curr_pos:curr_pos+L] = clean_block
                
                # Update KV cache cleanly for the next block
                past_kv_clean = self._encode_clean_block(clean_block, curr_pos, genre_id, past_kv_clean)
                curr_pos += L

        denoised_tensor_dict = {
            'instrument': canvas[0, :final_seq_len, 0],
            'pitch': canvas[0, :final_seq_len, 1],
            'velocity': canvas[0, :final_seq_len, 2],
            'onset': canvas[0, :final_seq_len, 3],
            'duration': canvas[0, :final_seq_len, 4]
        }
        
        # Filter out padding and start tokens
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
    parser.add_argument("--model", type=str, default="/gpfs/scratch/acw769/improvnet/artifacts/caddi_diffusion/best_model.pt")
    parser.add_argument("--input", type=str, default="/data/home/acw769/improvnet_2/improvnet/inference/debussy-clair-de-lune_original.mid")
    parser.add_argument("--output", type=str, default="/data/home/acw769/improvnet_2/improvnet/inference/generated.mid")
    parser.add_argument("--genre", type=str, default="classical")
    parser.add_argument("--strategy", type=str, default="rhythm_then_pitch", choices=["completion", "rhythm_transfer", "rhythm_then_pitch"])
    parser.add_argument("--keep_prompt_len", type=int, default=128, help="Number of prompt tokens to keep for completion strategy")
    args = parser.parse_args()

    inference = CaDDiInference(model_path=args.model)
    processor = ProcessData()
    
    print(f"Reading {args.input}...")
    source_midi = processor.read_midi(args.input)
    source_tokens = processor.midi_to_tokens(source_midi)
    
    print(f"Generating {args.genre} in {args.strategy} mode...")
    new_tokens = inference.generate(
        prompt_tokens=source_tokens,
        strategy=args.strategy,
        genre_str=args.genre,
        keep_prompt_len=args.keep_prompt_len
    )
    
    print(f"Saving to {args.output}...")
    new_midi = processor.tokens_to_midi(new_tokens)
    processor.save_midi(new_midi, args.output)
    print("Done!")