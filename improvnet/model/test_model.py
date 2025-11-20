import torch
import torch.nn.functional as F
import time
import sys
import os
import argparse
import traceback
from typing import Optional, Tuple, List

# --- IMPORTANT ---
# Make sure this import points to your model file
from model_with_cache import ImprovNet, ImprovNetConfig

# --- Constants for this Test ---
NUM_ATTRIBUTES = 6
NUM_VOICE_ATTRIBUTES = 5 
ATTR_ORDER = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
# We'll use ID 2 to represent a [MASK] token for this test
MASK_TOKEN_ID = 2 

# --- Dummy Generation Function ---
# This simulates the iterative denoising process
@torch.no_grad()
def generate_for_test(
    model: ImprovNet,
    initial_main: torch.Tensor,
    initial_accom: torch.Tensor,
    mask_main: torch.Tensor,
    mask_accom: torch.Tensor,
    steps: int = 12,
    adaptive_update_ratio: float = 0.25
) -> Tuple[torch.Tensor, torch.Tensor]:

    model.eval()
    device = next(model.parameters()).device
    B, L, _ = initial_main.shape

    # These masks select the tokens that should NOT be changed (the static prompt)
    static_mask_main = ~mask_main.bool()
    static_mask_accom = ~mask_accom.bool()
    
    # Store the original static tokens
    original_static_main = initial_main.clone()
    original_static_accom = initial_accom.clone()
    
    # xt is the "noisy" tensor that evolves over time
    xt_main = initial_main.clone()
    xt_accom = initial_accom.clone()
    
    cache = None 

    for k in range(steps):
        # On k=0, cache=None, k_step=0 -> Initialization
        # On k>0, cache!=None, k_step=k -> V-Verify
        k_step_for_model = k
        
        output = model(
            input_attributes_main=xt_main,
            input_attributes_accom=xt_accom,
            genre=torch.tensor([0], device=device), # Dummy genre
            form=torch.tensor([0], device=device),  # Dummy form
            dynamic_mask_main=mask_main,      # <-- PASS THE MASK
            dynamic_mask_accom=mask_accom,    # <-- PASS THE MASK
            cache=cache, 
            adaptive_update_ratio=adaptive_update_ratio,
            k_step=k_step_for_model, # Pass the current step
            return_dict=True
        )
        
        logits_main = output["logits_main"]
        logits_accom = output["logits_accom"]
        cache = output["cache"] # Get the new cache for the next iteration
        
        # --- Simplified Sampling (Argmax) ---
        x0_pred_main_list = []
        x0_pred_accom_list = []
        for i in range(NUM_VOICE_ATTRIBUTES):
            x0_pred_main_list.append(torch.argmax(logits_main[i], dim=-1))
            x0_pred_accom_list.append(torch.argmax(logits_accom[i], dim=-1))

        x0_pred_main = torch.stack(x0_pred_main_list, dim=-1)
        x0_pred_accom = torch.stack(x0_pred_accom_list, dim=-1)

        # --- Resample "Noise" (Simulated) ---
        # This is a simplified sampler: just replace all dynamic tokens with the prediction
        
        # Keep static tokens from the original
        xt_main = torch.where(static_mask_main, original_static_main, x0_pred_main)
        xt_accom = torch.where(static_mask_accom, original_static_accom, x0_pred_accom)

    # The final output is the last prediction
    return xt_main, xt_accom

# --- Helper to create data ---
def _create_dummy_data(B, L, vocabs, device):
    tensors = []
    for vocab_size in vocabs:
        # Ensure we don't accidentally generate the MASK_TOKEN_ID in the static part
        rand_tokens = torch.randint(MASK_TOKEN_ID + 1, vocab_size, (B, L), device=device)
        tensors.append(rand_tokens)
    return torch.stack(tensors, dim=-1)

# --- Main Test Function ---
def run_all_tests():
    print("--- 🚀 Starting ImprovNet Inference Test Suite ---")
    
    # --- Test Parameters (UPDATED) ---
    SEQ_LEN = 2048
    BATCH_SIZE = 2
    HIDDEN_SIZE = 780 
    NUM_HEADS = 30    
    NUM_LAYERS = 12   
    FFN_DIM = HIDDEN_SIZE * 4
    STEPS = 128     
    
    # [instrument, pitch, velocity, onset, duration]
    VOCAB_SIZES = [129, 128, 128, 512, 512]
    NUM_GENRES = 2
    NUM_FORMS = 2
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠️ WARNING: Running on CPU. Speed test will be slow and may not show speedup.")

    # --- 1. Model & Data Setup ---
    print(f"\n[Step 1/5] Setting up model and data...")
    try:
        config = ImprovNetConfig(
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            ffn_dim=FFN_DIM,
            vocab_sizes=VOCAB_SIZES,
            seq_len=SEQ_LEN,
            num_genres=NUM_GENRES,
            num_forms=NUM_FORMS,
            adaptive_update_ratio=0.25 
        )
        model = ImprovNet(config).to(device).eval()
        
        # --- Create a 50/50 Mask ---
        # Main stream: 50% static (prompt), 50% dynamic (to be generated)
        mask_main = (torch.rand(BATCH_SIZE, SEQ_LEN, NUM_VOICE_ATTRIBUTES, device=device) > 0.5)
        # Accompaniment stream: 100% static (full prompt)
        mask_accom = torch.full((BATCH_SIZE, SEQ_LEN, NUM_VOICE_ATTRIBUTES), False, dtype=torch.bool, device=device)
        
        initial_main = _create_dummy_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZES, device)
        initial_accom = _create_dummy_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZES, device)
        
        # --- Store original static tokens and apply MASK ---
        static_mask_main = ~mask_main.bool()
        static_mask_accom = ~mask_accom.bool()
        dynamic_mask_main = mask_main.bool()
        
        # Store a copy of the original static tokens for later comparison
        original_static_tokens_main = initial_main[static_mask_main]
        original_static_tokens_accom = initial_accom[static_mask_accom]
        
        # Overwrite the dynamic (masked) part with the MASK_TOKEN_ID
        initial_main[dynamic_mask_main] = MASK_TOKEN_ID
        
        print(f"  ✅ Setup complete. Model: {NUM_LAYERS} layers, {HIDDEN_SIZE} hidden, {SEQ_LEN} seq len.")
    except Exception as e:
        print(f"  🔴 FAILED setup: {e}")
        traceback.print_exc()
        return

    # --- 2. Test 1-3: Generation, Shape, and Mask Verification ---
    print(f"\n[Step 2/5] Running Generation ({STEPS} steps) & Verification...")
    try:
        generated_main, generated_accom = generate_for_test(
            model,
            initial_main, initial_accom,
            mask_main, mask_accom,
            steps=STEPS, # Use full steps
            adaptive_update_ratio=0.25
        )
        
        # Test 1: I/O Shapes
        expected_shape = (BATCH_SIZE, SEQ_LEN, NUM_VOICE_ATTRIBUTES)
        assert generated_main.shape == expected_shape
        assert generated_accom.shape == expected_shape
        print("  ✅ Test 1 (Shapes) PASSED.")
        
        # Test 2: Static Token Preservation
        generated_static_tokens_main = generated_main[static_mask_main]
        generated_static_tokens_accom = generated_accom[static_mask_accom]
        
        assert torch.all(original_static_tokens_main == generated_static_tokens_main), \
            "Static 'main' tokens were modified!"
        assert torch.all(original_static_tokens_accom == generated_static_tokens_accom), \
            "Static 'accom' tokens were modified!"
        print("  ✅ Test 2 (Static Preservation) PASSED.")
        
        # Test 3: All Masks are Generated
        generated_dynamic_tokens_main = generated_main[dynamic_mask_main]
        assert torch.all(generated_dynamic_tokens_main != MASK_TOKEN_ID), \
            "Generated tokens still contain MASK_TOKEN_ID!"
        print("  ✅ Test 3 (Masks Generated) PASSED.")
        
    except Exception as e:
        print(f"  🔴 FAILED Test 1, 2, or 3: {e}")
        traceback.print_exc()
        return

    # --- 3. Test 4: Caching Speed ---
    print(f"\n[Step 3/5] Running Test 4 (Caching Speed) for {STEPS} steps...")
    if device == 'cpu':
        print("  SKIPPED (Speed test is not meaningful on CPU).")
        print("\n🎉 --- Tests 1, 2 & 3 PASSED! --- 🎉")
        return
        
    try:
        # --- Run 1: Caching OFF (100% update) ---
        print(f"  Timing... ratio=1.0 (Caching OFF)")
        torch.cuda.synchronize()
        start_time_nocache = time.perf_counter()
        
        generate_for_test(
            model,
            initial_main, initial_accom,
            mask_main, mask_accom,
            steps=STEPS,
            adaptive_update_ratio=1.0 # 100% update = Caching OFF
        )
        
        torch.cuda.synchronize()
        time_no_cache = time.perf_counter() - start_time_nocache
        print(f"  Time (no cache): {time_no_cache:.4f} seconds")

        # --- Run 2: Caching ON (25% update) ---
        print(f"  Timing... ratio=0.25 (Caching ON)")
        torch.cuda.synchronize()
        start_time_cache = time.perf_counter()
        
        generate_for_test(
            model,
            initial_main, initial_accom,
            mask_main, mask_accom,
            steps=STEPS,
            adaptive_update_ratio=0.25 # 25% update
        )
        
        torch.cuda.synchronize()
        time_with_cache = time.perf_counter() - start_time_cache
        print(f"  Time (with cache): {time_with_cache:.4f} seconds")

        # --- 4. Verdict ---
        print("\n[Step 4/5] Verifying Speedup...")
        speedup_factor = time_no_cache / time_with_cache
        
        # We expect a significant speedup (at least 20% faster)
        assert time_with_cache < time_no_cache * 0.8, \
            f"Caching is not working! No-cache time: {time_no_cache:.4f}s, Cache time: {time_with_cache:.4f}s"
            
        print(f"  ✅ Test 4 (Caching Speed) PASSED!")
        print(f"  Speedup Factor: {speedup_factor:.2f}x")
        print("\n[Step 5/5] --- All tests PASSED! --- 🎉")

    except Exception as e:
        print(f"  🔴 FAILED Test 4: {e}")
        traceback.print_exc()
        return

if __name__ == "__main__":
    run_all_tests()