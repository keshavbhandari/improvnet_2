import tqdm
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
# Make sure this points to your new, correct model file
from improvnet.model.model_with_cache import ImprovNet, ImprovNetConfig
from improvnet.utils.utils import ProcessData
from improvnet.train.training_config import *

ATTR_ORDER = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
NUM_ATTRIBUTES = 5
PAD_TOKEN_ID = 2 

@torch.no_grad()
def generate_with_cache(
    model: ImprovNet,
    initial_main: torch.Tensor,
    initial_accom: torch.Tensor,
    mask_main: torch.Tensor,
    mask_accom: torch.Tensor,
    steps: int = 12,
    temperature: float = 1.0,
    genre_tokens: Optional[torch.Tensor] = None,
    form_tokens: Optional[torch.Tensor] = None,
    adaptive_update_ratio: float = 0.25
) -> Tuple[torch.Tensor, torch.Tensor]:

    model.eval()
    device = next(model.parameters()).device
    B, L, _ = initial_main.shape

    # --- 1. PREPARE MASKS ---
    per_token_mask_main = mask_main.any(dim=-1).bool()
    per_token_mask_accom = mask_accom.any(dim=-1).bool()
    static_mask_main = ~mask_main.bool()
    static_mask_accom = ~mask_accom.bool()
    
    original_static_main = initial_main.clone()
    original_static_accom = initial_accom.clone()
    xt_main = initial_main.clone()
    xt_accom = initial_accom.clone()

    # --- 2. SCHEDULING ---
    t_values = torch.linspace(0, 1, steps + 1, device=device).unsqueeze(0)
    gamma = torch.cos(t_values * torch.pi / 2.0)
    num_masked_main = per_token_mask_main.sum(dim=-1).float().unsqueeze(-1)
    num_masked_accom = per_token_mask_accom.sum(dim=-1).float().unsqueeze(-1)
    num_masked_per_step_main = (gamma * num_masked_main).round().long()
    num_masked_per_step_accom = (gamma * num_masked_accom).round().long()
    
    current_iter_mask_main = per_token_mask_main.clone()
    current_iter_mask_accom = per_token_mask_accom.clone()

    # --- 3. CACHE LOOP ---
    cache = None # This will be populated on the first step

    for k in tqdm.tqdm(range(steps), desc="Denoising Steps", ncols=80):

        k_step_for_model = k

        # --- A. Predict Clean Sequence (x0_pred) ---
        # On k=0, cache=None, the block runs in "Initialization" mode
        # On k=1..N, cache is not None, the block runs in "Adaptive Update" mode
        output = model(
            input_attributes_main=xt_main,
            input_attributes_accom=xt_accom,
            genre=genre_tokens,
            form=form_tokens,
            cache=cache, 
            adaptive_update_ratio=adaptive_update_ratio, # This is now being used
            k_step=k_step_for_model,
            return_dict=True
        )
        
        logits_main = output["logits_main"]
        logits_accom = output["logits_accom"]
        cache = output["cache"] # Get the new cache for the next step
        
        # (Rest of sampling logic is unchanged)
        x0_pred_main_list = []
        x0_pred_accom_list = []
        log_probs_main_list = []
        log_probs_accom_list = []

        for i in range(NUM_ATTRIBUTES):
            probs_main = F.softmax(logits_main[i] / temperature, dim=-1)
            sampled_main = torch.multinomial(probs_main.view(-1, probs_main.shape[-1]), 1).view(B, L)
            x0_pred_main_list.append(sampled_main)
            probs_accom = F.softmax(logits_accom[i] / temperature, dim=-1)
            sampled_accom = torch.multinomial(probs_accom.view(-1, probs_accom.shape[-1]), 1).view(B, L)
            x0_pred_accom_list.append(sampled_accom)
            log_probs_main = F.log_softmax(logits_main[i], dim=-1)
            log_probs_accom = F.log_softmax(logits_accom[i], dim=-1)
            log_probs_main_list.append(torch.gather(log_probs_main, -1, sampled_main.unsqueeze(-1)).squeeze(-1))
            log_probs_accom_list.append(torch.gather(log_probs_accom, -1, sampled_accom.unsqueeze(-1)).squeeze(-1))

        x0_pred_main = torch.stack(x0_pred_main_list, dim=-1)
        x0_pred_accom = torch.stack(x0_pred_accom_list, dim=-1)

        # --- B. Decide Which Dynamic Tokens to Unmask ---
        confidence_main = torch.stack(log_probs_main_list, dim=-1).sum(dim=-1)
        confidence_accom = torch.stack(log_probs_accom_list, dim=-1).sum(dim=-1)
        confidence_main[~current_iter_mask_main] = -float('inf')
        confidence_accom[~current_iter_mask_accom] = -float('inf')

        num_to_unmask_main = num_masked_per_step_main[:, k] - num_masked_per_step_main[:, k+1]
        num_to_unmask_accom = num_masked_per_step_accom[:, k] - num_masked_per_step_accom[:, k+1]

        max_main = current_iter_mask_main.sum(dim=-1)
        max_accom = current_iter_mask_accom.sum(dim=-1)
        min_val_tensor = torch.tensor(0, device=device, dtype=num_to_unmask_main.dtype)
        
        num_to_unmask_main = torch.clamp(num_to_unmask_main, min=min_val_tensor, max=max_main)
        num_to_unmask_accom = torch.clamp(num_to_unmask_accom, min=min_val_tensor, max=max_accom)
        
        batch_indices = torch.arange(B, device=device).unsqueeze(-1)

        if num_to_unmask_main.max() > 0:
            k_main = num_to_unmask_main[0].item()
            if k_main > 0:
                top_k_indices_main = torch.topk(confidence_main, k=k_main, dim=-1).indices
                xt_main[batch_indices, top_k_indices_main] = x0_pred_main[batch_indices, top_k_indices_main]
                current_iter_mask_main[batch_indices, top_k_indices_main] = False

        if num_to_unmask_accom.max() > 0:
            k_accom = num_to_unmask_accom[0].item()
            if k_accom > 0:
                top_k_indices_accom = torch.topk(confidence_accom, k=k_accom, dim=-1).indices
                xt_accom[batch_indices, top_k_indices_accom] = x0_pred_accom[batch_indices, top_k_indices_accom]
                current_iter_mask_accom[batch_indices, top_k_indices_accom] = False

        # --- C. **MANUALLY RESTORE STATIC TOKENS** ---
        xt_main = torch.where(static_mask_main, original_static_main, xt_main)
        xt_accom = torch.where(static_mask_accom, original_static_accom, xt_accom)

    model.train() 
    print("--- Generation Complete ---")
    return xt_main, xt_accom


def _collate_stream_for_inference(token_dict, mask_tensor, max_len, pad_id):
    # (Unchanged)
    try:
        input_tensor = torch.stack([token_dict[attr] for attr in ATTR_ORDER], dim=1)
    except KeyError as e:
        print(f"Collate Error: Missing key {e}. Token dict keys: {token_dict.keys()}", file=sys.stderr)
        raise
    except RuntimeError as e:
        print(f"Collate Error: Mismatched tensor lengths in dict. {e}", file=sys.stderr)
        raise
    seq_len = input_tensor.shape[0]
    pad_len = max_len - seq_len
    if pad_len > 0:
        input_pad = torch.full((pad_len, NUM_ATTRIBUTES), pad_id, dtype=torch.long)
        final_input = torch.cat([input_tensor, input_pad], dim=0)
        mask_pad = torch.full((pad_len, NUM_ATTRIBUTES), False, dtype=torch.bool)
        final_mask = torch.cat([mask_tensor, mask_pad], dim=0)
    else:
        final_input = input_tensor[:max_len]
        final_mask = mask_tensor[:max_len]
    return final_input, final_mask


def prefix_injection(generated_sequence):
    unique_note_instruments = set()
    present_prefix_instruments = set()
    
    for token_tuple in generated_sequence:
        first_attribute = token_tuple[0]
        
        if isinstance(first_attribute, tuple) and first_attribute[0] == 'instrument':
            unique_note_instruments.add(first_attribute[1])
            
        elif isinstance(first_attribute, tuple) and first_attribute[0] == 'prefix':
            if len(first_attribute) == 3 and first_attribute[1] == 'instrument':
                present_prefix_instruments.add(first_attribute[2])
                
    instruments_to_inject = unique_note_instruments - present_prefix_instruments
    
    prefix_tokens = []
    for inst_name in instruments_to_inject:
        prefix_data = ('prefix', 'instrument', inst_name)
        compound_prefix = (prefix_data, '<BLANK>', '<BLANK>', '<BLANK>', '<BLANK>')
        prefix_tokens.append(compound_prefix)

    generated_sequence = prefix_tokens + generated_sequence
    return generated_sequence

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load Model ---
    model = ImprovNet.from_pretrained(CHECKPOINTS_DIR)
    model.eval() 
    model.to(device)
    print("Model loaded.")

    processor = ProcessData()

    # --- 2. Load and Process Data ---
    (
        corrupted_main_dict,
        corrupted_accom_dict,
        mask_main_tensor,
        mask_accom_tensor,
        original_main_dict,
        original_accom_dict,
        genre_token,
        form_token
    ) = processor.pretraining_pipeline(
        file_path="/keshav/improvnet_2/improvnet/inference/debussy-clair-de-lune_original.mid", 
        genre="classical", 
        form="unknown", 
        corruption_type="pitch_velocity_mask",
        segment_length=MAX_LEN,
        ratio=0.8,
        apply_pitch_augmentation=False
    )
    print("Input data processed.")

    # --- 3. Collate and Pad ---
    corrupted_main, mask_main = _collate_stream_for_inference(
        corrupted_main_dict, mask_main_tensor, MAX_LEN, PAD_TOKEN_ID
    )
    corrupted_accom, mask_accom = _collate_stream_for_inference(
        corrupted_accom_dict, mask_accom_tensor, MAX_LEN, PAD_TOKEN_ID
    )
    corrupted_main = corrupted_main.unsqueeze(0).to(device)
    mask_main = mask_main.unsqueeze(0).to(device)
    corrupted_accom = corrupted_accom.unsqueeze(0).to(device)
    mask_accom = mask_accom.unsqueeze(0).to(device)
    genre_token = genre_token.to(device)
    form_token = form_token.to(device)
    
    # --- 4. Generate ---
    # This call now works with the new model
    generated_main, generated_accom = generate_with_cache(
        model=model,
        initial_main=corrupted_main,
        initial_accom=corrupted_accom,
        mask_main=mask_main,
        mask_accom=mask_accom,
        steps=256,
        temperature=0.5,
        adaptive_update_ratio=0.25, # <--- TRY TUNING THIS
        genre_tokens=genre_token,
        form_tokens=form_token
    )
    
    generated_main = generated_main.squeeze(0)
    generated_accom = generated_accom.squeeze(0)

    # --- 5. Decode and Save ---
    # (Unchanged)
    generated_main_dict = {ATTR_ORDER[i]: generated_main[:, i] for i in range(NUM_ATTRIBUTES)}
    generated_accom_dict = {ATTR_ORDER[i]: generated_accom[:, i] for i in range(NUM_ATTRIBUTES)}
    generated_sequence_main = processor.tensor_to_tokens(generated_main_dict)
    generated_sequence_accom = processor.tensor_to_tokens(generated_accom_dict)

    print("--- Generated Main Sequence (Head) ---")
    print(generated_sequence_main[:1024])
    print("...")

    # Inject prefix tokens for any missing instruments
    generated_sequence_main = prefix_injection(generated_sequence_main)

    # Write generated_sequence_main as a text file for inspection
    with open("/keshav/improvnet_2/improvnet/inference/generated_main.txt", "w") as f:
        for token in generated_sequence_main:
            f.write(f"{token}\n")
    print("Saved generated_main.txt")

    midi_tokens_main = processor.tokens_to_midi(generated_sequence_main)
    processor.save_midi(midi_tokens_main, "/keshav/improvnet_2/improvnet/inference/generated_main.mid")
    print("Saved generated_main.mid")
    
    is_accom_empty = (generated_accom == PAD_TOKEN_ID).all()
    
    if not is_accom_empty:
        print("Accompaniment stream found, decoding and saving...")
        generated_accom_dict = {ATTR_ORDER[i]: generated_accom[:, i] for i in range(NUM_ATTRIBUTES)}
        generated_sequence_accom = processor.tensor_to_tokens(generated_accom_dict)
        print("--- Generated Accompaniment Sequence (Head) ---")
        print(generated_sequence_accom[:15])
        print("...")
        midi_tokens_accom = processor.tokens_to_midi(generated_sequence_accom)
        processor.save_midi(midi_tokens_accom, "/keshav/improvnet_2/improvnet/inference/generated_accom.mid")
        print("Saved generated_accom.mid")
    else:
        print("Accompaniment stream is empty (all PAD tokens). Only saving main stream.")