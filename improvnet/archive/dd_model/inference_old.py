import torch
import os
from tqdm import tqdm
from improvnet.utils.utils import ProcessData
from improvnet.model.config import *
from improvnet.model.model import PrefixARModel

@torch.no_grad()
def generate_music(
    midi_filepath: str,
    output_filepath: str,
    checkpoint: str,
    chunk_size: int = 1024, 
    overlap_ratio: float = 0.125
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = ProcessData()
    print("Loading Model...")
    model = PrefixARModel().to(device)
    
    state = torch.load(checkpoint, map_location=device)
    state_dict = state['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # 1. Read Original MIDI
    print(f"Processing {os.path.basename(midi_filepath)}...")
    midi_dict = processor.read_midi(midi_filepath)
    full_tokens = processor.midi_to_tokens(midi_dict)
    
    target_length = len(full_tokens)
    print(f"Original sequence length: {target_length} tokens. Generation will match this length.")

    # Instrument is global for the whole piece
    inst_multihot = processor.get_instrument_multihot(full_tokens).unsqueeze(0).to(device)
    print(f"Extracted instrument multihot vector: {inst_multihot.cpu().numpy()}")

    # 2. Orchestration Variables
    total_generated = []
    overlap_tokens = None
    overlap_len = int(chunk_size * overlap_ratio)
    current_seq_idx = 0
    
    print("Starting chunked generation...")
    
    pbar = tqdm(total=target_length, desc="Generating Music")
    
    while sum(t.shape[1] for t in total_generated) < target_length:
        
        # --- A. Calculate Temporal Bounds ---
        start_idx = max(0, current_seq_idx - (overlap_len if overlap_tokens is not None else 0))
        end_idx = start_idx + chunk_size
        
        # --- B. Extract Local Conditions ---
        chunk_raw_tokens = full_tokens[start_idx:end_idx]
        
        mel_cond = processor.skyline_groundline(chunk_raw_tokens, algorithm="skyline")
        har_cond = processor.skyline_groundline(chunk_raw_tokens, algorithm="groundline")
        rhy_cond = processor.extract_rhythm(chunk_raw_tokens, ratio=1.0)
        
        def prepare_cond(tokens, max_len):
            tensor = processor.format_variable_sequence(tokens, max_len).unsqueeze(0).to(device)
            mask = (tensor[:, :, 0] == 2) # ID 2 is <P>
            return tensor, mask

        # Format to strict MAX lengths exactly like training
        mel_tensor, mel_mask = prepare_cond(mel_cond, MAX_MEL_LEN)
        har_tensor, har_mask = prepare_cond(har_cond, MAX_HAR_LEN)
        rhy_tensor, rhy_mask = prepare_cond(rhy_cond, MAX_RHY_LEN)
        
        # --- C. Prepare Prior Tokens ---
        prior_tokens = None
        if total_generated:
            flat_gen = torch.cat(total_generated, dim=1)
            prior_tokens = flat_gen[:, -PRIOR_LEN:, :]

        # --- D. Determine New Tokens to Generate ---
        # Generate exactly enough to fill the rest of the chunk, or the rest of the song
        current_generated_len = sum(t.shape[1] for t in total_generated)
        remaining = target_length - current_generated_len
        tokens_to_generate = min(chunk_size - (overlap_len if overlap_tokens is not None else 0), remaining)
        
        # --- E. GENERATE ---
        new_tokens = model.generate(
            melody=mel_tensor, melody_mask=mel_mask,
            harmony=har_tensor, harmony_mask=har_mask,
            rhythm=rhy_tensor, rhythm_mask=rhy_mask,
            instrument=inst_multihot,
            prior_tokens=prior_tokens,
            overlap_tokens=overlap_tokens,
            max_new_tokens=tokens_to_generate,
            temperature=1.0,
            top_k=50
        )
        
        total_generated.append(new_tokens)
        current_seq_idx += new_tokens.shape[1]
        pbar.update(new_tokens.shape[1])
        
        # --- F. Prepare Overlap for Next Chunk ---
        # Combine overlap and new tokens to extract the final N tokens for the next seed
        full_current_chunk = torch.cat([overlap_tokens, new_tokens], dim=1) if overlap_tokens is not None else new_tokens
        actual_overlap = min(overlap_len, full_current_chunk.shape[1])
        overlap_tokens = full_current_chunk[:, -actual_overlap:, :]
        
    pbar.close()

    # 3. Post-Process the Generated Output
    final_tensor = torch.cat(total_generated, dim=1).squeeze(0).cpu()  # [total, 5]
    print(f"Generated raw output shape: {final_tensor.shape}")

    attr_keys = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
    reconstructed_dict = {attr_keys[i]: final_tensor[:, i] for i in range(5)}

    print("Decoding tokens to MIDI...")
    reconstructed_tokens = processor.tensor_to_tokens(reconstructed_dict)
    print(reconstructed_tokens[0:512])

    special_tokens = {'<P>', '<S>', '<E>', '<BLANK>'}
    cleaned_tokens = [token for token in reconstructed_tokens if not any(
        (isinstance(item, tuple) and len(item) > 1 and item[1] in special_tokens) or 
        (isinstance(item, str) and item in special_tokens)
        for item in token
    )]
    
    scrubbed = len(reconstructed_tokens) - len(cleaned_tokens)
    if scrubbed > 0: print(f"Scrubbed {scrubbed} hallucinated special tokens.")

    # 4. Save
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    reconstructed_midi_dict = processor.tokens_to_midi(cleaned_tokens)
    processor.save_midi(reconstructed_midi_dict, output_filepath)
    print(f"Successfully saved AI-generated track to: {output_filepath}")

if __name__ == "__main__":
    generate_music(
        midi_filepath="/data/home/acw769/improvnet_2/improvnet/inference/debussy-clair-de-lune_original.mid",
        output_filepath="/data/home/acw769/improvnet_2/improvnet/inference/generated.mid",
        checkpoint="/gpfs/scratch/acw769/improvnet/artifacts/autoregressive_kl/best_model.pt"
    )