import torch
import torch.nn.functional as F
from tqdm import tqdm
from improvnet.model.ar_config import *
from improvnet.model.ar_model import ARContextModel
from improvnet.utils.ar_utils import ProcessData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _amp_dtype():
    if not torch.cuda.is_available(): return None
    major = torch.cuda.get_device_capability()[0]
    return torch.bfloat16 if major >= 8 else torch.float16

AMP_DTYPE = _amp_dtype()

@torch.no_grad()
def sample_logits(logits, temperature=1.0, top_k=50, forbidden_ids=None):
    """
    Standard autoregressive sampling with Temperature and Top-K filtering.
    """
    logits = logits / max(temperature, 1e-5)
    
    if forbidden_ids:
        for f_id in forbidden_ids:
            if f_id >= 0:
                logits[:, f_id] = float('-inf')
                
    if top_k > 0:
        val, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(logits < val[:, -1:], float('-inf'), logits)
        
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class ARContextInference:
    def __init__(self, model_path: str):
        self.processor = ProcessData()
        self.tokenizer = self.processor.tokenizer
        
        print("Loading AR Context Tower (Tower A)...")
        self.model = ARContextModel().to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Prevent the model from hallucinating structural/diffusion padding
        self.forbidden_ids = [
            PAD_ID, MASK_ID, BLANK_ID, SEP_ID,
            self.tokenizer.tok_to_id.get('<U>', -1)
        ]

    @torch.no_grad()
    def generate(
        self, 
        prompt_tokens: list, 
        genre_str: str = "jazz", 
        max_tokens: int = 2048,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> list:
        
        # 1. Conditioning Setup
        genre_id = torch.tensor([self.processor.get_genre_id(genre_str)], dtype=torch.long, device=DEVICE)
        
        # Extract the global multi-hot vector from the prompt to guide the continuation
        multi_hot_tensor = self.processor.get_instrument_multihot(prompt_tokens).unsqueeze(0).to(DEVICE)
        
        prompt_tensor = self.processor.tokens_to_tensor(prompt_tokens).unsqueeze(0).to(DEVICE)
        P = prompt_tensor.size(1)
        
        generated_tokens = prompt_tokens.copy()
        e_id = self.tokenizer.tok_to_id.get('<E>', -1)
        
        print(f"Prefilling context with {P} prompt tokens (Genre: {genre_str.upper()})...")
        
        # --- PREFILL PHASE ---
        # We pass the entire prompt through in one parallel pass to populate the KV cache
        with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
            logits, past_kv = self.model(
                target=prompt_tensor,
                genre=genre_id,
                multi_hot=multi_hot_tensor,
                use_cache=True,
                past_key_values=None,
                seq_offset=0
            )
            
        # The model's forward pass drops the first 2 control logits, aligning logits perfectly with inputs.
        # We sample the prediction for the VERY LAST token of the prompt.
        next_token_id = sample_logits(logits[:, -1, :], temperature, top_k, self.forbidden_ids)
        generated_tokens.append(self.processor.tokenizer.id_to_tok[next_token_id.item()])
        
        curr_token = next_token_id
        
        # The KV cache now holds the prompt (P) PLUS the 2 control embeddings (Genre + Multi-Hot)
        # So the next token's absolute RoPE position is P + 2
        seq_offset = P + 2
        
        # --- DECODE PHASE ---
        pbar = tqdm(total=max_tokens, initial=P + 1, desc="AR Generation")
        
        while len(generated_tokens) < max_tokens:
            
            # Step-by-Step Causal Unrolling
            with torch.amp.autocast(DEVICE, dtype=AMP_DTYPE):
                logits, past_kv = self.model(
                    target=curr_token,
                    genre=None,         # Skipped, already in KV cache
                    multi_hot=None,     # Skipped, already in KV cache
                    use_cache=True,
                    past_key_values=past_kv,
                    seq_offset=seq_offset
                )
                
            next_token_id = sample_logits(logits[:, -1, :], temperature, top_k, self.forbidden_ids)
            curr_token = next_token_id
            
            next_tok_str = self.processor.tokenizer.id_to_tok[next_token_id.item()]
            generated_tokens.append(next_tok_str)
            pbar.update(1)
            seq_offset += 1
            
            # Early Stopping if the model naturally ends the song
            if next_token_id.item() == e_id:
                print("\nEnd token <E> generated! Stopping early.")
                break
                
        pbar.close()
        
        # Clean up any stray PAD or S tokens
        cleaned_tokens = [tok for tok in generated_tokens if tok not in ('<PAD>', '<S>')]
        return cleaned_tokens

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/gpfs/scratch/acw769/improvnet/artifacts/ar_context/best_model.pt")
    parser.add_argument("--input", type=str, help="Path to input MIDI file for prompt context", default="/data/home/acw769/improvnet_2/improvnet/inference/debussy-clair-de-lune_original.mid")
    parser.add_argument("--output", type=str, default="/data/home/acw769/improvnet_2/improvnet/inference/generated.mid")
    parser.add_argument("--genre", type=str, default="classical")
    parser.add_argument("--prompt_len", type=int, default=256, help="Number of tokens to extract from input as the prompt")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum length of the generated sequence")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    inference = ARContextInference(model_path=args.model)
    processor = ProcessData()
    
    print(f"Reading {args.input}...")
    source_midi = processor.read_midi(args.input)
    source_tokens = processor.midi_to_tokens(source_midi)
    
    # Extract the requested prompt length
    prompt = source_tokens[:args.prompt_len]
    
    new_tokens = inference.generate(
        prompt_tokens=prompt,
        genre_str=args.genre,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    print(f"Saving to {args.output}...")
    new_midi = processor.tokens_to_midi(new_tokens)
    processor.save_midi(new_midi, args.output)
    print("Generation Complete!")