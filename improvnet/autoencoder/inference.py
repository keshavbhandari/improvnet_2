import torch
import os

# Update these imports based on your exact file structure
from improvnet.utils.utils import ProcessData
from improvnet.autoencoder.config import PATCH_SIZE
from improvnet.autoencoder.model import ContinuousAutoencoder

@torch.no_grad()
def encode_long_sequence(model, x: torch.Tensor, max_seq_per_forward: int = 2048) -> torch.Tensor:
    """Encodes an arbitrarily long sequence by chunking it to prevent VRAM OOM."""
    model.eval()
    B, T, num_attr = x.shape
    all_latents = []
    
    # Ensure chunk size is perfectly divisible by patch size to prevent hanging tokens
    assert max_seq_per_forward % PATCH_SIZE == 0, "Chunk size must be divisible by PATCH_SIZE"
    
    for i in range(0, T, max_seq_per_forward):
        x_chunk = x[:, i : i + max_seq_per_forward, :]
        
        # model.encode_to_latents returns continuous vectors [B, T_chunk // Patch_Size, Latent_Dim]
        z_chunk = model.encode_to_latents(x_chunk) 
        all_latents.append(z_chunk)
        
    return torch.cat(all_latents, dim=1)

@torch.no_grad()
def decode_long_sequence(model, z: torch.Tensor, max_seq_per_forward: int = 2048) -> torch.Tensor:
    """Decodes in chunks by passing continuous latents to the decoder API."""
    model.eval()
    B, num_patches_total, D = z.shape
    all_reconstructed_tokens = []
    
    # The latent chunk size is the token chunk size divided by the patch size
    latent_chunk_size = max_seq_per_forward // PATCH_SIZE
    
    for i in range(0, num_patches_total, latent_chunk_size):
        z_chunk = z[:, i : i + latent_chunk_size, :]
        
        # model.decode_from_latents returns discrete predicted tokens [B, T_chunk, 5]
        tokens_chunk = model.decode_from_latents(z_chunk)
        
        all_reconstructed_tokens.append(tokens_chunk)
        
    return torch.cat(all_reconstructed_tokens, dim=1)


def reconstruct_midi(
    midi_filepath: str, 
    model_checkpoint: str, 
    output_filepath: str, 
    chunk_size: int = 2048
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Data Processor
    processor = ProcessData()
    
    # 2. Load the Model
    print("Loading model checkpoint...")
    model = ContinuousAutoencoder() 
    state_dict = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict)
    model.to(device)
    model.eval()

    # 3. Read and Tokenize MIDI
    print(f"Reading original MIDI: {midi_filepath}")
    midi_dict = processor.read_midi(midi_filepath)
    tokens = processor.midi_to_tokens(midi_dict)

    # 4. Format for the Model with Dynamic Padding
    print(f"Original sequence length: {len(tokens)} notes")
    token_tensors_dict = processor.tokens_to_tensor(tokens)
    attribute_keys = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
    
    # Calculate how many pad tokens we need to make it divisible by PATCH_SIZE
    T_original = len(tokens)
    remainder = T_original % PATCH_SIZE
    pad_len = (PATCH_SIZE - remainder) if remainder != 0 else 0

    if pad_len > 0:
        print(f"Padding {pad_len} dummy tokens to satisfy PATCH_SIZE ({PATCH_SIZE})")
        for key in attribute_keys:
            # Dynamically fetch the specific <P> pad ID for this attribute
            pad_id = getattr(processor.tokenizer, f"tok_to_id_{key}")['<P>']
            padding_tensor = torch.full((pad_len,), pad_id, dtype=torch.long)
            token_tensors_dict[key] = torch.cat([token_tensors_dict[key], padding_tensor])

    # Stack the dictionary into a flat [T_padded, 5] tensor
    tensor_list = [token_tensors_dict[key] for key in attribute_keys]
    x_flat = torch.stack(tensor_list, dim=-1)
    
    # Add Batch dimension: [1, T_padded, 5]
    x_batched = x_flat.unsqueeze(0).to(device)

    # 5. The Bottleneck: Encode and Decode
    print(f"Encoding sequence of shape {x_batched.shape}...")
    latent_vectors = encode_long_sequence(model, x_batched, max_seq_per_forward=chunk_size)
    print(f"Compressed to continuous latents of shape {latent_vectors.shape}")

    print("Decoding latents back to tokens...")
    reconstructed_batched = decode_long_sequence(model, latent_vectors, max_seq_per_forward=chunk_size)
    
    # 6. Unpack flat sequence and Filter Out Padding
    # Remove batch dimension: [1, T_padded, 5] -> [T_padded, 5]
    reconstructed_flat = reconstructed_batched.squeeze(0).cpu()

    # Identify the <P> token ID for just one of the attributes (e.g., instrument)
    inst_pad_id = getattr(processor.tokenizer, "tok_to_id_instrument")['<P>']
    
    # Create a boolean mask of valid notes (where instrument is NOT the pad token)
    # This safely drops the exact number of dummy tokens we added earlier
    valid_mask = reconstructed_flat[:, 0] != inst_pad_id
    reconstructed_clean = reconstructed_flat[valid_mask]
    
    print(f"Filtered out padding. Final reconstructed length: {reconstructed_clean.shape[0]} notes")

    # Map the 5 columns back to the dictionary format ProcessData expects
    reconstructed_dict = {
        attribute_keys[i]: reconstructed_clean[:, i] for i in range(5)
    }

    # 7. Detokenize and Save
    print("Converting dictionary back to MIDI format...")
    reconstructed_tokens = processor.tensor_to_tokens(reconstructed_dict)
    print(reconstructed_tokens[0:100])
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    reconstructed_midi_dict = processor.tokens_to_midi(reconstructed_tokens)
    processor.save_midi(reconstructed_midi_dict, output_filepath)
    print(f"Successfully saved reconstructed MIDI to: {output_filepath}")

if __name__ == "__main__":
    # Example usage:
    reconstruct_midi(
        midi_filepath="/data/home/acw769/improvnet_2/improvnet/inference/debussy-clair-de-lune_original.mid",
        model_checkpoint="/gpfs/scratch/acw769/improvnet/artifacts/autoencoder_8patch_1024latent/best_model.pt",
        output_filepath="/data/home/acw769/improvnet_2/improvnet/inference/reconstructed.mid",
        chunk_size=1024 
    )