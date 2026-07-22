import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from improvnet.model.omni_config import *
from improvnet.model.omni_model import CaDDiModel
from improvnet.utils.omni_utils import ProcessData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _amp_dtype():
    if not torch.cuda.is_available(): return None
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16

AMP_DTYPE = _amp_dtype()

def cosine_schedule(step: int, total_steps: int) -> float:
    """
    Returns the fraction of tokens that should REMAIN masked at the current step.
    step 1 = ~0.99 masked, final step = 0.0 masked.
    """
    return math.cos((math.pi / 2) * (step / total_steps))

class OmniInference:
    def __init__(self, model_path: str):
        self.processor = ProcessData()
        self.tokenizer = self.processor.tokenizer
        
        print("Loading Bidirectional Omni-CaDDi Model...")
        self.model = CaDDiModel().to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Tokens the model should NEVER hallucinate during inference
        self.forbidden_ids = [PAD_ID, MASK_ID, SEP_ID]

    @torch.no_grad()
    def _denoise_block(
        self, prefix_tensor, draft_tensor, editable_mask, genre_id, 
        mode_id, length_ctrl_id, multi_hot_tensor, diffusion_steps, temperature
    ):
        """
        Executes parallel bidirectional diffusion on the active draft.
        """
        B = 1
        P = prefix_tensor.shape[1]
        D = draft_tensor.shape[1]
        sep_tensor = torch.tensor([[SEP_ID]], dtype=torch.long, device=DEVICE)
        
        # The router requires precise bounds to trigger Bidirectional attention
        # Prefix + SEP evaluates causally, Draft evaluates bidirectionally!
        causal_prefix_len = P + 1
        draft_size = D
        
        current_draft = draft_tensor.clone()
        num_editable = editable_mask.sum().item()
        
        if num_editable == 0:
            return current_draft
            
        for step in range(1, diffusion_steps + 1):
            t_val = 1.0 - (step / diffusion_steps)
            t_tensor = torch.full((1,), t_val, device=DEVICE)
            
            # Combine dynamically: [Prefix] -> <SEP> -> [Draft]
            target_seq = torch.cat([prefix_tensor, sep_tensor, current_draft], dim=1)
            
            with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                logits = self.model(
                    target=target_seq,
                    timestep=t_tensor,
                    genre=genre_id,
                    mode=mode_id,
                    length_ctrl=length_ctrl_id,
                    multi_hot=multi_hot_tensor,
                    causal_prefix_len=causal_prefix_len,
                    draft_size=draft_size,
                    use_cache=False
                )
            
            # Isolate the logits specifically for the Draft portion
            draft_logits = logits[:, causal_prefix_len:, :] / max(temperature, 1e-5)
            
            for f_id in self.forbidden_ids:
                draft_logits[:, :, f_id] = float('-inf')
                
            probs = F.softmax(draft_logits, dim=-1)
            samp = torch.multinomial(probs.view(-1, VOCAB_SIZE), num_samples=1).view(B, D)
            conf = probs.gather(-1, samp.unsqueeze(-1)).squeeze(-1)
            
            # 1. Enforce Non-Editable Tokens (Lock the prompt / classical pitches)
            samp[~editable_mask] = current_draft[~editable_mask]
            conf[~editable_mask] = float('inf')
            
            # 2. Re-Masking based on Confidence
            ratio_to_mask = cosine_schedule(step, diffusion_steps)
            num_to_mask = int(ratio_to_mask * num_editable)
            
            if num_to_mask > 0:
                editable_confs = conf.clone()
                editable_confs[~editable_mask] = float('inf') # Only mask tokens we are allowed to generate
                
                _, lowest_idx = torch.topk(editable_confs.view(-1), num_to_mask, largest=False)
                samp.view(-1)[lowest_idx] = MASK_ID
                
            current_draft = samp
            
        return current_draft

    @torch.no_grad()
    def generate(
        self, 
        prompt_tokens: list, 
        strategy: str = "completion", 
        genre_str: str = "jazz", 
        mode: int = 1,              # 0 = Strict, 1 = Edit
        length_ctrl: int = 1,       # 0 = Fixed, 1 = Elastic (<BLANK> allowed)
        max_tokens: int = 2048,
        keep_prompt_len: int = 128,
        diffusion_steps: int = 16,
        temperature: float = 1.0
    ) -> list:
        
        genre_id = torch.tensor([self.processor.get_genre_id(genre_str)], dtype=torch.long, device=DEVICE)
        mode_id = torch.tensor([mode], dtype=torch.long, device=DEVICE)
        length_ctrl_id = torch.tensor([length_ctrl], dtype=torch.long, device=DEVICE)
        
        # Extract global multihot condition from the prompt to anchor the style
        multi_hot_tensor = self.processor.get_instrument_multihot(prompt_tokens).unsqueeze(0).to(DEVICE)
        e_id = self.tokenizer.tok_to_id.get('<E>', -1)
        
        print(f"Executing '{strategy}' | Mode: {'EDIT' if mode==1 else 'STRICT'} | Elasticity: {'ON' if length_ctrl==1 else 'OFF'}")

        if strategy == "completion":
            prompt_len = min(len(prompt_tokens), keep_prompt_len, max_tokens)
            current_tokens = prompt_tokens[:prompt_len]
            
            pbar = tqdm(total=max_tokens, initial=prompt_len, desc="Completion")
            while len(current_tokens) < max_tokens:
                L_draft = min(BLOCK_SIZE, max_tokens - len(current_tokens))
                if L_draft <= 0: break
                
                # Sliding Prefix Window
                P_tokens = current_tokens[-PROMPT_MAX:]
                prefix_tensor = self.processor.tokens_to_tensor(P_tokens).unsqueeze(0).to(DEVICE)
                
                draft_tensor = torch.full((1, L_draft), MASK_ID, dtype=torch.long, device=DEVICE)
                editable_mask = torch.ones((1, L_draft), dtype=torch.bool, device=DEVICE)
                
                denoised_tensor = self._denoise_block(
                    prefix_tensor, draft_tensor, editable_mask, genre_id, 
                    mode_id, length_ctrl_id, multi_hot_tensor, diffusion_steps, temperature
                )
                
                denoised_tokens = self.processor.tensor_to_tokens(denoised_tensor.squeeze(0))
                
                # Check for Early Stop
                if '<E>' in denoised_tokens:
                    e_idx = denoised_tokens.index('<E>')
                    current_tokens.extend(denoised_tokens[:e_idx + 1])
                    pbar.update(e_idx + 1)
                    print(f"\nEnd token <E> generated! Stopping early.")
                    break
                else:
                    current_tokens.extend(denoised_tokens)
                    pbar.update(L_draft)
                    
            pbar.close()
            final_tokens = current_tokens

        elif strategy == "rhythm_only":
            actual_seq_len = min(len(prompt_tokens), max_tokens)
            full_tokens = prompt_tokens[:actual_seq_len]
            tensor_seq = self.processor.tokens_to_tensor(full_tokens).to(DEVICE)
            
            # Isolate the Rhythm Tokens
            is_rhythm = torch.zeros_like(tensor_seq, dtype=torch.bool)
            for i, tok in enumerate(full_tokens):
                if isinstance(tok, tuple) and tok[0] in ('onset', 'dur'):
                    is_rhythm[i] = True
                    
            canvas = tensor_seq.clone()
            canvas[is_rhythm] = MASK_ID
            
            num_blocks = math.ceil(actual_seq_len / BLOCK_SIZE)
            
            for b in tqdm(range(num_blocks), desc="Rhythm Transfer (Block by Block)"):
                start = b * BLOCK_SIZE
                end = min(start + BLOCK_SIZE, actual_seq_len)
                
                prefix_start = max(0, start - PROMPT_MAX)
                prefix_tensor = canvas[prefix_start:start].unsqueeze(0)
                
                # First block handles the lack of prefix by providing an empty causal trigger
                if prefix_tensor.shape[1] == 0:
                    prefix_tensor = torch.tensor([[self.tokenizer.tok_to_id.get('<S>', PAD_ID)]], dtype=torch.long, device=DEVICE)
                
                draft_tensor = canvas[start:end].unsqueeze(0)
                editable_mask = is_rhythm[start:end].unsqueeze(0)
                
                if editable_mask.any():
                    denoised_tensor = self._denoise_block(
                        prefix_tensor, draft_tensor, editable_mask, genre_id, 
                        mode_id, length_ctrl_id, multi_hot_tensor, diffusion_steps, temperature
                    )
                    canvas[start:end] = denoised_tensor.squeeze(0)
                    
            final_tokens = self.processor.tensor_to_tokens(canvas)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Post-Processing: Strip out structural silences and padding before MIDI conversion!
        cleaned_tokens = [tok for tok in final_tokens if tok not in ('<BLANK>', '<P>', '<S>', '<E>')]
        return cleaned_tokens

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/gpfs/scratch/acw769/improvnet/artifacts/omni_caddi/best_model.pt")
    parser.add_argument("--input", type=str, required=True, default="/data/home/acw769/improvnet_2/improvnet/inference/debussy-clair-de-lune_original.mid")
    parser.add_argument("--output", type=str, default="/data/home/acw769/improvnet_2/improvnet/inference/omni_output.mid")
    parser.add_argument("--strategy", type=str, default="completion", choices=["completion", "rhythm_only"])
    parser.add_argument("--genre", type=str, default="jazz")
    parser.add_argument("--mode", type=int, default=1, choices=[0, 1], help="0: Strict Edit, 1: Loose Target Edit")
    parser.add_argument("--length_ctrl", type=int, default=1, choices=[0, 1], help="0: Fixed Length, 1: Elastic (<BLANK> Allowed)")
    parser.add_argument("--keep_prompt_len", type=int, default=128, help="Context length for completion.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum generated token limit.")
    parser.add_argument("--diffusion_steps", type=int, default=16, help="Number of parallel diffusion steps per block.")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    inference = OmniInference(model_path=args.model)
    processor = ProcessData()
    
    print(f"Reading {args.input}...")
    source_midi = processor.read_midi(args.input)
    source_tokens = processor.midi_to_tokens(source_midi)
    
    new_tokens = inference.generate(
        prompt_tokens=source_tokens,
        strategy=args.strategy,
        genre_str=args.genre,
        mode=args.mode,
        length_ctrl=args.length_ctrl,
        max_tokens=args.max_tokens,
        keep_prompt_len=args.keep_prompt_len,
        diffusion_steps=args.diffusion_steps,
        temperature=args.temperature
    )
    
    print(f"Saving to {args.output}...")
    new_midi = processor.tokens_to_midi(new_tokens)
    processor.save_midi(new_midi, args.output)
    print("Generation Complete!")