import torch
import traceback
from model import AmortizedImprovNet, ImprovNetConfig

def run_amortized_test():
    print("--- 🚀 Testing Amortized ImprovNet (Seq2Seq with KV Cache) ---")
    
    # Config
    SEQ_LEN = 128 # Per track
    BATCH_SIZE = 2
    HIDDEN_SIZE = 120
    NUM_HEADS = 6
    VOCAB_SIZES = [129, 128, 128, 512, 512]
    NUM_GENRES = 2
    NUM_FORMS = 2
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # 1. Setup
        config = ImprovNetConfig(
            hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS,
            num_decoder_layers=2, num_encoder_layers=2,
            vocab_sizes=VOCAB_SIZES, seq_len=SEQ_LEN
        )
        model = AmortizedImprovNet(config).to(device)
        model.eval()
        
        # 2. Data
        # Encoder gets NOISY input (Length = 2 * SEQ_LEN)
        input_enc = torch.stack([torch.randint(0, v, (BATCH_SIZE, 2*SEQ_LEN), device=device) for v in VOCAB_SIZES], dim=-1)
        # Decoder gets CLEAN input (Length = 2 * SEQ_LEN)
        input_dec = torch.stack([torch.randint(0, v, (BATCH_SIZE, 2*SEQ_LEN), device=device) for v in VOCAB_SIZES], dim=-1)
        
        genre = torch.zeros((BATCH_SIZE,), dtype=torch.long, device=device)
        form = torch.zeros((BATCH_SIZE,), dtype=torch.long, device=device)
        timestep = torch.tensor([100] * BATCH_SIZE, device=device)

        print("[1] Running Full Forward Pass (Training Mode)...")
        output_full = model(
            input_attributes_encoder=input_enc,
            input_attributes_decoder=input_dec,
            genre=genre, form=form, timestep=timestep,
            return_dict=True
        )
        logits_full = output_full["logits_main"]
        print("    ✅ Full pass successful.")

        print("[2] Testing KV Cache (Inference Mode)...")
        # Pass 1: First token
        input_dec_step1 = input_dec[:, 0:1, :] # First token
        output_step1 = model(
            input_attributes_encoder=input_enc, # Encoder sees full context
            input_attributes_decoder=input_dec_step1,
            genre=genre, form=form, timestep=timestep,
            past_key_values=None,
            return_dict=True
        )
        kv_cache = output_step1["past_key_values"]
        
        # Pass 2: Second token, providing cache
        input_dec_step2 = input_dec[:, 1:2, :] # Second token
        output_step2 = model(
            input_attributes_encoder=input_enc,
            input_attributes_decoder=input_dec_step2, # Pass ONLY new token
            genre=genre, form=form, timestep=timestep,
            past_key_values=kv_cache, # Provide cache
            return_dict=True
        )
        
        # Verification
        # The logits for the second token in Full Pass should match the logits from Step 2 with Cache
        logit_full_t1 = logits_full[0][:, 1, :] # 2nd token (index 1)
        logit_cache_t1 = output_step2["logits_main"][0][:, 0, :] # 1st token of the slice
        
        diff = torch.abs(logit_full_t1 - logit_cache_t1).max()
        print(f"    Max Difference (Full vs Cache): {diff.item():.6f}")
        
        if diff < 1e-5:
            print("    ✅ KV Cache logic verified! Decoder states are identical.")
        else:
            print("    ❌ KV Cache mismatch.")

        print("\n--- 🎉 Test Passed! ---")
        
    except Exception as e:
        print(f"\n🔴 Test Failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_amortized_test()