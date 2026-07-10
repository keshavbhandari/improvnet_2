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

class CaDDiDebugInference:
    def __init__(self, model_path: str):
        self.processor = ProcessData()
        self.tokenizer = self.processor.tokenizer
        
        print("Loading 1D CaDDi Model for Stateless AR Debugging...")
        self.model = CaDDiModel().to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.forbidden_ids = [
            PAD_ID, MASK_ID, BLANK_ID, SEP_ID, 
            self.tokenizer.tok_to_id.get('<U>', -1)
        ]

    @torch.no_grad()
    def generate_stateless_ar_diffusion(
        self, 
        prompt_tokens: list, 
        genre_str: str = "jazz", 
        keep_prompt_len: int = 128,
        num_tokens_to_generate: int = 256,
        temperature: float = 1.0
    ) -> list:
        
        genre_id = torch.tensor([self.processor.get_genre_id(genre_str)], dtype=torch.long, device=DEVICE)
        prompt_len = min(len(prompt_tokens), keep_prompt_len)
        
        print(f"Encoding {prompt_len} Prompt Tokens...")
        prompt_tensor = self.processor.format_variable_sequence(prompt_tokens[:prompt_len], prompt_len, pad_id=PAD_ID).unsqueeze(0).to(DEVICE)
        
        L = num_tokens_to_generate
        current_block = torch.full((1, L), PAD_ID, dtype=torch.long, device=DEVICE)
        prev_confs = None
        
        time_vals = [1.0, 0.75, 0.50, 0.25]
        mask_ratios = [1.0, 0.75, 0.50, 0.25]
        
        # We will maintain the exact historical sequence and timesteps across drafts
        history_seq = prompt_tensor.clone()
        history_ts = torch.zeros(1, prompt_len, device=DEVICE)
        
        print(f"Generating block of {L} tokens using Stateless Diffusion AR...")
        
        for step in range(len(mask_ratios)):
            ratio = mask_ratios[step]
            t_val = time_vals[step]
            
            # 1. Prepare target_draft for this diffusion step
            target_draft = current_block.clone()
            num_to_mask = int(ratio * L)
            if num_to_mask > 0:
                if prev_confs is None or ratio == 1.0:
                    perm = torch.randperm(L)
                    chosen = perm[:num_to_mask]
                else:
                    _, worst_idx = torch.topk(prev_confs[0], num_to_mask, largest=False)
                    chosen = worst_idx
                target_draft[0, chosen] = MASK_ID
                
            draft_out = target_draft.clone()
            draft_confs = torch.zeros(1, L, device=DEVICE) if prev_confs is None else prev_confs.clone()
            
            # 2. Append <SEP> to the historical context
            sep_tensor = torch.tensor([[SEP_ID]], dtype=torch.long, device=DEVICE)
            history_seq = torch.cat([history_seq, sep_tensor], dim=1)
            history_ts = torch.cat([history_ts, torch.zeros(1, 1, device=DEVICE)], dim=1)
            
            # 3. Token-by-token generation (Stateless)
            for i in tqdm(range(L), desc=f"  Draft {step+1}/4 (t={t_val})", leave=False):
                if target_draft[0, i] == MASK_ID:
                    
                    # Construct the EXACT sequence up to token 'i'
                    # [Prompt] -> <SEP> -> [Draft 1] -> <SEP> -> [Current Draft Resolved Tokens] -> [MASK_ID]
                    curr_prefix = draft_out[:, :i]
                    curr_token = target_draft[:, i:i+1] # This is MASK_ID
                    
                    seq_to_feed = torch.cat([history_seq, curr_prefix, curr_token], dim=1)
                    ts_to_feed = torch.cat([history_ts, torch.full((1, i + 1), t_val, device=DEVICE)], dim=1)
                    
                    # Stateless Forward Pass: No KV Cache! The model evaluates the entire history array from scratch.
                    with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                        logits = self.model(
                            target=seq_to_feed, timestep=ts_to_feed, genre=genre_id, 
                            use_cache=False, past_key_values=None, seq_offset=0
                        )
                        
                    # Extract the logit for the very last position we appended (the MASK_ID)
                    step_logits = logits[:, -1, :] / max(temperature, 1e-5)
                    for f_id in self.forbidden_ids:
                        if f_id >= 0:
                            step_logits[:, f_id] = float('-inf')
                            
                    probs = F.softmax(step_logits, dim=-1)
                    samp = torch.multinomial(probs, num_samples=1)
                    conf = probs.gather(-1, samp).squeeze(-1)
                    
                    # Save prediction
                    draft_out[0, i] = samp[0, 0]
                    draft_confs[0, i] = conf[0]
                else:
                    # Protect tokens that were not masked in this draft
                    draft_confs[0, i] = float('inf') 
                    
            # 4. Step complete! Append this fully resolved draft to the history for the NEXT draft's context
            history_seq = torch.cat([history_seq, draft_out], dim=1)
            history_ts = torch.cat([history_ts, torch.full((1, L), t_val, device=DEVICE)], dim=1)
            
            current_block = draft_out
            prev_confs = draft_confs
            
        # Final Generation is Prompt + Final Draft
        final_sequence = torch.cat([prompt_tensor, current_block], dim=1)
        return self.processor.tensor_to_tokens(final_sequence[0])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/gpfs/scratch/acw769/improvnet/artifacts/caddi_diffusion/small/best_model.pt")
    parser.add_argument("--input", type=str, default="/data/home/acw769/improvnet_2/improvnet/inference/debussy-clair-de-lune_original.mid")
    parser.add_argument("--output", type=str, default="/data/home/acw769/improvnet_2/improvnet/inference/generated.mid")
    parser.add_argument("--genre", type=str, default="jazz")
    parser.add_argument("--keep_prompt_len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    debugger = CaDDiDebugInference(model_path=args.model)
    processor = ProcessData()
    
    print(f"Reading {args.input}...")
    source_midi = processor.read_midi(args.input)
    source_tokens = processor.midi_to_tokens(source_midi)
    
    new_tokens = debugger.generate_stateless_ar_diffusion(
        prompt_tokens=source_tokens,
        genre_str=args.genre,
        keep_prompt_len=args.keep_prompt_len,
        num_tokens_to_generate=256,
        temperature=args.temperature
    )
    
    print(f"Saving to {args.output}...")
    print(new_tokens[125:150], "...")
    new_midi = processor.tokens_to_midi(new_tokens)
    processor.save_midi(new_midi, args.output)
    print("Done!")