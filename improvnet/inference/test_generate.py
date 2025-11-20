import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from torch.utils.checkpoint import checkpoint
# from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional
from improvnet.model_FIT.model import ImprovNetModel, ImprovNetModelConfig

# --- The Generate Function ---

@torch.no_grad()
def generate(
    model: ImprovNetModel,
    initial_tokens: torch.Tensor,
    initial_mask: torch.Tensor, # True for dynamic/masked, False for static/content
    steps: int = 12, # Number of denoising steps
    temperature: float = 1.0,
    K_r: int = 2, # Response refresh interval
    rho: float = 0.25, # Adaptive update ratio
    genre_tokens: Optional[torch.Tensor] = None,
    form_tokens: Optional[torch.Tensor] = None
) -> torch.Tensor:

    model.eval()
    config = model.config
    device = next(model.parameters()).device
    MASK_TOKEN_ID = config.vocab_size - 1 # Assuming this is correct

    # static_mask = True where tokens are static (content)
    static_mask = ~initial_mask
    original_static_tokens = initial_tokens[static_mask] # Store original static values

    xt = initial_tokens.clone()

    # 1. Initialize Cache (Step K)
    print("--- Initializing Cache (Step K) ---")
    logits, cache = model.initialize_cache(
        xt,
        genre_tokens=genre_tokens,
        form_tokens=form_tokens
    )
    print("Cache initialized.")

    # Masking Schedule
    t_values = torch.linspace(0, 1, steps + 1, device=device).unsqueeze(0) # 0 to 1
    gamma = torch.cos(t_values * torch.pi / 2.0)
    num_masked_this_batch = initial_mask.sum(dim=-1).float().unsqueeze(-1)
    num_masked_per_step = (gamma * num_masked_this_batch).round().long()

    # Iterative Denoising Loop
    current_iter_mask = initial_mask.clone() # True for dynamic tokens still masked

    for k in tqdm.tqdm(range(steps), desc="Denoising Steps", ncols=80):

        # --- A. Predict Clean Sequence (x0_pred) ---
        if temperature > 0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        x0_pred = torch.multinomial(probs.view(-1, config.vocab_size), 1).view(xt.shape)

        # --- B. Decide Which Dynamic Tokens to Unmask ---
        confidence = torch.gather(probs, -1, x0_pred.unsqueeze(-1)).squeeze(-1)
        confidence[~current_iter_mask] = -1.0 # Ignore already unmasked or static tokens

        num_to_unmask = num_masked_per_step[:, k] - num_masked_per_step[:, k+1]
        max_values = current_iter_mask.sum(dim=-1)
        min_values = torch.tensor(0, device=num_to_unmask.device, dtype=num_to_unmask.dtype)
        num_to_unmask = torch.clamp(num_to_unmask, min=min_values, max=max_values)

        if num_to_unmask.max() > 0:
            k_for_topk = num_to_unmask[0].item()
            if k_for_topk > 0:
                top_k_indices = torch.topk(confidence, k=k_for_topk, dim=-1).indices
                batch_indices = torch.arange(xt.shape[0], device=device).unsqueeze(-1)

                # Update xt only at the selected dynamic token positions
                xt[batch_indices, top_k_indices] = x0_pred[batch_indices, top_k_indices]
                # Update the mask tracking remaining dynamic tokens
                current_iter_mask[batch_indices, top_k_indices] = False

        # --- C. **MANUALLY RESTORE STATIC TOKENS** ---
        # After updating xt based on predictions, force static positions back
        xt[static_mask] = original_static_tokens
        # --- END RESTORATION ---

        # --- D. Get Next Step's Logits using dLLM-Cache ---
        if k < steps - 1:
            # if (k % K_r == 0): print("Refreshing response cache.")
            # if not (k % K_r == 0): print("Using adaptive update.") # Simplified condition

            logits = model.forward_with_cache(
                input_tokens=xt,
                # static_mask=static_mask, # REMOVE
                cache=cache,
                step_k=k, # Keep step_k for potential future use or logging
                K_r=K_r,
                rho=rho,
                genre_tokens=genre_tokens,
                form_tokens=form_tokens
            )

    model.train()
    print("--- Generation Complete ---")

    # Final replacement for any remaining dynamic masks
    final_dynamic_mask = (xt == MASK_TOKEN_ID) & (~static_mask) # Check only dynamic positions
    if final_dynamic_mask.any():
        num_remaining = final_dynamic_mask.sum().item()
        print(f"Replacing {num_remaining} remaining dynamic mask(s) by re-sampling without mask...")

        # Get the logits corresponding to the remaining masked positions
        # Use the 'logits' variable from the *last* forward pass (before the loop ended)
        final_masked_logits = logits[final_dynamic_mask] # Shape: [num_remaining, vocab_size]

        # --- Prevent sampling MASK_TOKEN_ID ---
        # Set the logit for the MASK_TOKEN_ID to negative infinity
        final_masked_logits[:, MASK_TOKEN_ID] = -torch.inf

        # Sample from the modified distribution
        final_probs = F.softmax(final_masked_logits, dim=-1)
        final_replacement_tokens = torch.multinomial(final_probs, 1).squeeze(-1) # Shape: [num_remaining]

        # Place the newly sampled tokens into xt
        xt[final_dynamic_mask] = final_replacement_tokens

    # Final check/restoration of static tokens (just in case)
    xt[static_mask] = original_static_tokens

    return xt


def generate_without_cache(
    model: ImprovNetModel,
    initial_tokens: torch.Tensor,
    initial_mask: torch.Tensor, # True for dynamic/masked, False for static/content
    steps: int = 12, # Number of denoising steps
    temperature: float = 1.0,
    genre_tokens: Optional[torch.Tensor] = None,
    form_tokens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    A simplified generate function that does not use caching.
    Useful for testing or when cache is not implemented.
    """
    model.eval()
    config = model.config
    device = next(model.parameters()).device
    MASK_TOKEN_ID = config.vocab_size - 1 # Assuming this is correct

    static_mask = ~initial_mask
    original_static_tokens = initial_tokens[static_mask] # Store original static values

    xt = initial_tokens.clone()

    # Masking Schedule
    t_values = torch.linspace(0, 1, steps + 1, device=device).unsqueeze(0) # 0 to 1
    gamma = torch.cos(t_values * torch.pi / 2.0)
    num_masked_this_batch = initial_mask.sum(dim=-1).float().unsqueeze(-1)
    num_masked_per_step = (gamma * num_masked_this_batch).round().long()

    current_iter_mask = initial_mask.clone() # True for dynamic tokens still masked

    for k in tqdm.tqdm(range(steps), desc="Denoising Steps", ncols=80):

        # --- A. Predict Clean Sequence (x0_pred) ---
        logits = model(
            xt,
            genre_tokens=genre_tokens,
            form_tokens=form_tokens
        )
        if temperature > 0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        x0_pred = torch.multinomial(probs.view(-1, config.vocab_size), 1).view(xt.shape)

        # --- B. Decide Which Dynamic Tokens to Unmask ---
        confidence = torch.gather(probs, -1, x0_pred.unsqueeze(-1)).squeeze(-1)
        confidence[~current_iter_mask] = -1.0 # Ignore already unmasked or static tokens

        num_to_unmask = num_masked_per_step[:, k] - num_masked_per_step[:, k+1]
        max_values = current_iter_mask.sum(dim=-1)
        min_values = torch.tensor(0, device=num_to_unmask.device, dtype=num_to_unmask.dtype)
        num_to_unmask = torch.clamp(num_to_unmask, min=min_values, max=max_values)

        if num_to_unmask.max() > 0:
            k_for_topk = num_to_unmask[0].item()
            if k_for_topk > 0:
                top_k_indices = torch.topk(confidence, k=k_for_topk, dim=-1).indices
                batch_indices = torch.arange(xt.shape[0], device=device).unsqueeze(-1)

                # Update xt only at the selected dynamic token positions
                xt[batch_indices, top_k_indices] = x0_pred[batch_indices, top_k_indices]
                # Update the mask tracking remaining dynamic tokens
                current_iter_mask[batch_indices, top_k_indices] = False

        # --- C. **MANUALLY RESTORE STATIC TOKENS** ---
        xt[static_mask] = original_static_tokens
        # --- END RESTORATION ---
    model.train()
    print("--- Generation Complete (No Cache) ---")

    # Final replacement for any remaining dynamic masks
    final_dynamic_mask = (xt == MASK_TOKEN_ID) & (~static_mask) # Check only dynamic positions
    if final_dynamic_mask.any():
        num_remaining = final_dynamic_mask.sum().item()
        print(f"Replacing {num_remaining} remaining dynamic mask(s) by re-sampling without mask...")

        # Get the logits corresponding to the remaining masked positions
        logits = model(
            xt,
            genre_tokens=genre_tokens,
            form_tokens=form_tokens
        )
        final_masked_logits = logits[final_dynamic_mask] # Shape: [num_remaining, vocab_size]

        # --- Prevent sampling MASK_TOKEN_ID ---
        final_masked_logits[:, MASK_TOKEN_ID] = -torch.inf

        # Sample from the modified distribution
        final_probs = F.softmax(final_masked_logits, dim=-1)
        final_replacement_tokens = torch.multinomial(final_probs, 1).squeeze(-1) # Shape: [num_remaining]

        # Place the newly sampled tokens into xt
        xt[final_dynamic_mask] = final_replacement_tokens
    # Final check/restoration of static tokens (just in case)
    xt[static_mask] = original_static_tokens

    return xt

# --- Dummy Generate Call ---

if __name__ == "__main__":
    from time import time
    torch.manual_seed(1234)
    
    # 1. Setup Config
    # Using small params for a quick test
    BATCH_SIZE = 3
    SEQ_LEN = 4096
    PATCH_LEN = 32
    VOCAB_SIZE = 3000
    EMBED_DIM = 512
    LATENTS_PER_GROUP = 16
    NUM_ALTERNATIONS = 8
    NUM_LOCAL_LAYERS = 2
    NUM_GLOBAL_LAYERS = 2
    HEADS = 8
    
    config = ImprovNetModelConfig(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        seq_length=SEQ_LEN,
        patch_length=PATCH_LEN,
        local_dim=EMBED_DIM,
        global_dim=EMBED_DIM,
        latents_per_group=LATENTS_PER_GROUP,
        num_alternations=NUM_ALTERNATIONS,
        num_local_layers=NUM_LOCAL_LAYERS,
        num_global_layers=NUM_GLOBAL_LAYERS,
        heads=HEADS,
        pad_token_id=0 # Use 0 for padding
    )
    
    # 2. Instantiate Model
    model = ImprovNetModel(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model with {sum(p.numel() for p in model.parameters()):,} params created on {device}.")

    for i in ["generate_without_cache", "generate"]:
        # Time the dummy generate call
        start_time = time()

        # 3. Create Dummy Inputs
        MASK_TOKEN_ID = VOCAB_SIZE - 1 # Use 49 as mask
        
        # Create original tokens (1-48)
        original_tokens = torch.randint(1, MASK_TOKEN_ID, (BATCH_SIZE, SEQ_LEN), device=device)
        
        # Mask 70% of tokens (dynamic "response" part)
        mask_ratio = 0.7
        initial_mask = (torch.rand(original_tokens.shape, device=device) < mask_ratio)
        
        # Create the input by applying the mask
        initial_tokens = original_tokens.clone()
        initial_tokens[initial_mask] = MASK_TOKEN_ID
        
        # Unmasked tokens (the "prompt"/content)
        initial_tokens[~initial_mask] = original_tokens[~initial_mask]
        
        # Dummy conditional tokens (optional)
        genre_tokens = torch.randint(0, config.genre_vocab_size, (BATCH_SIZE,), device=device)
        form_tokens = torch.randint(0, config.form_vocab_size, (BATCH_SIZE,), device=device)

        print(f"\n--- Starting Generation Test ---")
        print(f"Batch Size: {BATCH_SIZE}, Seq Len: {SEQ_LEN}")
        print(f"Static (Content) Tokens: {(~initial_mask).sum().item()}")
        print(f"Dynamic (Masked) Tokens: {initial_mask.sum().item()}")

        if i == "generate":
            # 4. Run Dummy Generate Call
            generated_output = generate(
                model=model,
                initial_tokens=initial_tokens,
                initial_mask=initial_mask,
                steps=512,            # 8 denoising steps
                temperature=1.0,
                K_r=5,              # Refresh response every 2 steps
                rho=0.25,           # Update 25% of dynamic tokens
                genre_tokens=genre_tokens,
                form_tokens=form_tokens
            )
        else:
            generated_output = generate_without_cache(
                model=model,
                initial_tokens=initial_tokens,
                initial_mask=initial_mask,
                steps=512,            # 8 denoising steps
                temperature=1.0,
                genre_tokens=genre_tokens,
                form_tokens=form_tokens
            )
        
        # 5. Check Output
        print(f"\nOutput shape: {generated_output.shape} (Expected: ({BATCH_SIZE}, {SEQ_LEN}))")
        
        # Check if all masks are gone
        masks_left = (generated_output == MASK_TOKEN_ID).sum().item()
        print(f"Mask tokens remaining: {masks_left}")
        
        # Check if static tokens were preserved
        static_tokens_preserved = (generated_output[~initial_mask] == original_tokens[~initial_mask]).all()
        print(f"Static content tokens preserved: {static_tokens_preserved.item()}")
        
        print("\nDummy generate call successful!")

        end_time = time()
        print(f"Total time taken: {end_time - start_time:.2f} seconds")

        torch.cuda.empty_cache()