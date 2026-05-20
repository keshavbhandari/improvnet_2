import torch
import os

# Update these imports based on your exact file structure
from improvnet.utils.utils import ProcessData
from improvnet.fsq.compound_model import FSQAutoencoder

@torch.no_grad()
def encode_long_sequence(model, x: torch.Tensor, max_seq_per_forward: int = 1024) -> torch.Tensor:
    """Encodes an arbitrarily long sequence by chunking it to prevent VRAM OOM."""
    model.eval()
    B, T, num_attr = x.shape
    all_indices = []
    
    for i in range(0, T, max_seq_per_forward):
        # Slices perfectly, even if the last chunk is smaller than max_seq_per_forward
        x_chunk = x[:, i : i + max_seq_per_forward, :]
        
        # model.encode_to_indices returns integer codes [B, T_chunk, Num_Quantizers]
        indices_chunk = model.encode_to_indices(x_chunk) 
        all_indices.append(indices_chunk)
        
    return torch.cat(all_indices, dim=1)

@torch.no_grad()
def decode_long_sequence(model, indices: torch.Tensor, max_seq_per_forward: int = 1024) -> torch.Tensor:
    """Decodes in chunks by passing residual indices to the decoder API."""
    model.eval()
    B, T_total, num_q = indices.shape
    all_reconstructed_tokens = []
    
    for i in range(0, T_total, max_seq_per_forward):
        indices_chunk = indices[:, i : i + max_seq_per_forward, :]
        
        # model.decode_from_indices returns discrete predicted tokens [B, T_chunk, 5]
        tokens_chunk = model.decode_from_indices(indices_chunk)
        
        all_reconstructed_tokens.append(tokens_chunk)
        
    return torch.cat(all_reconstructed_tokens, dim=1)


def reconstruct_midi(
    midi_filepath: str, 
    model_checkpoint: str, 
    output_filepath: str, 
    chunk_size: int = 1024
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Data Processor
    processor = ProcessData()
    
    # 2. Load the Model
    print("Loading model checkpoint...")
    model = FSQAutoencoder() 
    state_dict = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict)
    model.to(device)
    model.eval()

    # 3. Read and Tokenize MIDI
    print(f"Reading original MIDI: {midi_filepath}")
    midi_dict = processor.read_midi(midi_filepath)
    tokens = processor.midi_to_tokens(midi_dict)

    # 4. Format for the Model (No Patching!)
    print(f"Total sequence length: {len(tokens)} notes")
    # Returns a dictionary of 1D tensors
    token_tensors_dict = processor.tokens_to_tensor(tokens)
    
    # Stack the dictionary into a flat [T, 5] tensor
    attribute_keys = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
    tensor_list = [token_tensors_dict[key] for key in attribute_keys]
    x_flat = torch.stack(tensor_list, dim=-1)
    
    # Add Batch dimension: [1, T, 5]
    x_batched = x_flat.unsqueeze(0).to(device)

    # 5. The Bottleneck: Encode and Decode
    print(f"Encoding sequence of shape {x_batched.shape}...")
    latent_codes = encode_long_sequence(model, x_batched, max_seq_per_forward=chunk_size)
    print(f"Compressed to latent codes of shape {latent_codes.shape}")

    print("Decoding latents back to tokens...")
    reconstructed_batched = decode_long_sequence(model, latent_codes, max_seq_per_forward=chunk_size)
    
    # 6. Unpack flat sequence
    # Remove batch dimension: [1, T, 5] -> [T, 5]
    reconstructed_flat = reconstructed_batched.squeeze(0).cpu()

    # Map the 5 columns back to the dictionary format ProcessData expects
    reconstructed_dict = {
        attribute_keys[i]: reconstructed_flat[:, i] for i in range(5)
    }

    # 7. Detokenize and Save
    print("Converting dictionary back to MIDI format...")
    reconstructed_tokens = processor.tensor_to_tokens(reconstructed_dict)
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    reconstructed_midi_dict = processor.tokens_to_midi(reconstructed_tokens)
    processor.save_midi(reconstructed_midi_dict, output_filepath)
    print(f"Successfully saved reconstructed MIDI to: {output_filepath}")


if __name__ == "__main__":
    # Example usage:
    reconstruct_midi(
        midi_filepath="/data/home/acw769/improvnet_2/improvnet/inference/debussy-clair-de-lune_original.mid",
        model_checkpoint="/gpfs/scratch/acw769/improvnet/artifacts/rfsq_1_patch_4_levels/best_model.pt",
        output_filepath="/data/home/acw769/improvnet_2/improvnet/inference/reconstructed.mid",
        chunk_size=1024 # Matches your training sequence length!
    )