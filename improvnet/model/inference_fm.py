import torch
import os
from tqdm import tqdm
from improvnet.utils.utils import ProcessData
from improvnet.autoencoder.config import PATCH_SIZE
from improvnet.model.config_fm import *
from improvnet.autoencoder.model import ContinuousAutoencoder
from improvnet.model.model_fm import FlowMatchingModel

@torch.no_grad()
def generate_music(
    midi_filepath: str,
    output_filepath: str,
    ae_checkpoint: str,
    fm_checkpoint: str,
    num_target_notes: int = 1024, # How long the generated song should be
    cond_notes: int = 128,        # The 128-note conditioning chunks
    cfg_scales: dict = {
        "melody": 3.0, 
        "harmony": 3.0, 
        "rhythm": 3.0, 
        "inst": 3.0
    },
    inference_steps: int = 50
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Processor and Load Models
    processor = ProcessData()
    
    print("Loading frozen Stage 1 Autoencoder...")
    autoencoder = ContinuousAutoencoder().to(device)
    ae_state = torch.load(ae_checkpoint, map_location=device)
    autoencoder.load_state_dict(ae_state['model_state_dict'] if 'model_state_dict' in ae_state else ae_state)
    autoencoder.eval()

    print("Loading Stage 2 Flow Matching Model...")
    fm_model = FlowMatchingModel(
        latent_dim=LATENT_DIM, hidden_dim=FM_HIDDEN_DIM, 
        num_layers=FM_LAYERS, num_heads=FM_HEADS, num_inst_classes=NUM_INSTRUMENT_CLASSES
    ).to(device)
    fm_state = torch.load(fm_checkpoint, map_location=device)
    fm_model.load_state_dict(fm_state['model_state_dict'])
    fm_model.eval()

    # 2. Extract Conditions from Clair de Lune
    print(f"Extracting {cond_notes} notes from {os.path.basename(midi_filepath)}...")
    midi_dict = processor.read_midi(midi_filepath)
    tokens = processor.midi_to_tokens(midi_dict)
    
    # Grab the first 'cond_notes' (e.g., 128) notes of the piece
    source_slice = tokens[:cond_notes]
    
    melody_cond = processor.skyline_groundline(source_slice, algorithm="skyline")
    harmony_cond = processor.skyline_groundline(source_slice, algorithm="groundline")
    rhythm_cond = processor.extract_rhythm(source_slice, ratio=1.0)
    
    # We want Piano for Clair de Lune (Class index 0)
    inst_multihot = processor.get_instrument_multihot(source_slice).unsqueeze(0).to(device)
    print(inst_multihot)

    # 3. Format and Encode Conditions
    def prepare_cond(cond_tokens):
        # Format pads/truncates to exactly cond_notes length
        tensor_dict = processor.tokens_to_tensor(cond_tokens)
        formatted = processor.format_sequence(tensor_dict, cond_notes).unsqueeze(0).to(device)
        # Encode to latents
        z = autoencoder.encode_to_latents(formatted)
        # Create mask (False means "pay attention to everything")
        mask = torch.zeros((1, cond_notes // PATCH_SIZE), dtype=torch.bool, device=device)
        return z, mask

    z_mel, mel_mask = prepare_cond(melody_cond)
    z_har, har_mask = prepare_cond(harmony_cond)
    z_rhy, rhy_mask = prepare_cond(rhythm_cond)

    # 4. Set up the Target Latent Space (Pure Noise)
    target_patches = num_target_notes // PATCH_SIZE
    z_t = torch.randn((1, target_patches, LATENT_DIM), device=device)
    
    # 5. The ODE Solver Loop (Euler Method)
    print(f"Generating latents using {inference_steps} ODE steps with CFG Scale {cfg_scales}...")
    dt = 1.0 / inference_steps

    with torch.amp.autocast('cuda', enabled=True):
        for step in tqdm(range(inference_steps), desc="ODE Integration"):
            t_val = step * dt
            t_tensor = torch.tensor([t_val], device=device)
            
            # 1. Unconditional Pass (Drop everything)
            v_uncond = fm_model(
                z_t, t_tensor, z_mel, mel_mask, z_har, har_mask, 
                z_rhy, rhy_mask, inst_multihot, 
                cfg_drops={"melody": True, "harmony": True, "rhythm": True, "inst": True}
            )
            
            # 2. Melody Only Pass
            v_mel = fm_model(
                z_t, t_tensor, z_mel, mel_mask, z_har, har_mask, 
                z_rhy, rhy_mask, inst_multihot, 
                cfg_drops={"melody": False, "harmony": True, "rhythm": True, "inst": True}
            )
            
            # 3. Harmony Only Pass
            v_har = fm_model(
                z_t, t_tensor, z_mel, mel_mask, z_har, har_mask, 
                z_rhy, rhy_mask, inst_multihot, 
                cfg_drops={"melody": True, "harmony": False, "rhythm": True, "inst": True}
            )
            
            # 4. Rhythm Only Pass
            v_rhy = fm_model(
                z_t, t_tensor, z_mel, mel_mask, z_har, har_mask, 
                z_rhy, rhy_mask, inst_multihot, 
                cfg_drops={"melody": True, "harmony": True, "rhythm": False, "inst": True}
            )
            
            # 5. Instrument Only Pass
            v_inst = fm_model(
                z_t, t_tensor, z_mel, mel_mask, z_har, har_mask, 
                z_rhy, rhy_mask, inst_multihot, 
                cfg_drops={"melody": True, "harmony": True, "rhythm": True, "inst": False}
            )
            
            # --- COMPOSITIONAL CFG MATH ---
            # Calculate the isolated direction each condition wants to push the noise
            dir_mel = v_mel - v_uncond
            dir_har = v_har - v_uncond
            dir_rhy = v_rhy - v_uncond
            dir_inst = v_inst - v_uncond
            
            # Multiply each direction by its independent scale and sum them together!
            v_final = v_uncond \
                      + cfg_scales["melody"] * dir_mel \
                      + cfg_scales["harmony"] * dir_har \
                      + cfg_scales["rhythm"] * dir_rhy \
                      + cfg_scales["inst"] * dir_inst
            
            # Euler Step: Push the noisy latents toward the combined music
            z_t = z_t + v_final * dt

    # 6. Decode the Final Latents Back to MIDI
    print("Generation complete! Decoding latents back to MIDI tokens...")
    # Because target might be long (e.g. 1024), we decode in chunks just to be safe
    all_reconstructed = []
    chunk_size = 1024 // PATCH_SIZE 
    for i in range(0, target_patches, chunk_size):
        z_chunk = z_t[:, i:i+chunk_size, :]
        tokens_chunk = autoencoder.decode_from_latents(z_chunk)
        all_reconstructed.append(tokens_chunk)
        
    reconstructed_batched = torch.cat(all_reconstructed, dim=1)
    reconstructed_flat = reconstructed_batched.squeeze(0).cpu()

    attribute_keys = ['instrument', 'pitch', 'velocity', 'onset', 'duration']
    reconstructed_dict = {
        attribute_keys[i]: reconstructed_flat[:, i] for i in range(5)
    }

    print("Saving MIDI file...")
    reconstructed_tokens = processor.tensor_to_tokens(reconstructed_dict)
    print(reconstructed_tokens)

    # Manually delete any hallucinated Pad or Start tokens from the middle of the song
    special_tokens = {'<P>', '<S>', '<E>', '<BLANK>'}
    cleaned_tokens = [token for token in tokens if not any(
        (isinstance(item, tuple) and len(item) > 1 and item[1] in special_tokens) or 
        (isinstance(item, str) and item in special_tokens)
        for item in token
    )]
    print(f"Scrubbed {len(reconstructed_tokens) - len(cleaned_tokens)} ghost padding tokens.")

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    reconstructed_midi_dict = processor.tokens_to_midi(cleaned_tokens)
    processor.save_midi(reconstructed_midi_dict, output_filepath)
    print(f"Successfully saved AI-generated Clair de Lune to: {output_filepath}")

if __name__ == "__main__":
    generate_music(
        midi_filepath="/data/home/acw769/improvnet_2/improvnet/inference/debussy-clair-de-lune_original.mid",
        output_filepath="/data/home/acw769/improvnet_2/improvnet/inference/clair_de_lune_generated.mid",
        ae_checkpoint="/gpfs/scratch/acw769/improvnet/artifacts/autoencoder/best_model.pt",
        fm_checkpoint="/gpfs/scratch/acw769/improvnet/artifacts/flow_matching/best_model.pt",
        num_target_notes=1024, # Generate ~1.5 minutes of music
        cond_notes=128,        # Condition on the first 128 notes of Debussy
        cfg_scales={
            "melody": 1.0, 
            "harmony": 1.0, 
            "rhythm": 1.0, 
            "inst": 4.0
        },
        inference_steps=100     # Number of Euler integration steps
    )