import torch
import os

RUN_NAME = "omni_caddi_diffusion_v1"
SAVE_DIR = "/gpfs/scratch/acw769/improvnet/artifacts/omni_caddi"
os.makedirs(SAVE_DIR, exist_ok=True)

RESUME_TRAINING = False

# --- VOCABULARY ---
# Bumped to a multiple of 64 for Tensor Cores
VOCAB_SIZE = 67761  
GENRES = ["classical", "jazz", "blues", "unknown"]
NUM_GENRES = len(GENRES)

# --- SEQUENCE MATH ---
SEQ_LEN = 2048 
BLOCK_SIZE = 512 
PROMPT_MAX = 1024 

# 1D Special Token IDs
PAD_ID = 2
MASK_ID = 5
BLANK_ID = 6
SEP_ID = 7

# --- ARCHITECTURE MATH ---
EMBED_DIM = 1024
N_HEADS = 16       
N_KV_HEADS = 4     
N_LAYERS = 20

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 16
ACCUM_STEPS = 1
LR = 1e-4 
WARMUP_STEPS = 5000 
N_STEPS = 1_000_000 
GRAD_CLIP = 1.0

LOG_EVERY = 10
VAL_EVERY = 20000

# Total number of diffusion steps to simulate (Smooth Trajectory)
DIFFUSION_STEPS = 16 

JSONL_FILES = [
    "/data/scratch/acw769/improvnet/artifacts/data/misc_data_tokenized.jsonl"
]

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gb < 60.0:
        BATCH_SIZE = 8
        ACCUM_STEPS = 4
        RUN_NAME = RUN_NAME + "_small"
        SAVE_DIR = os.path.join(SAVE_DIR, "small")
        print(f"Omni Config: Detected {vram_gb:.1f}GB VRAM. Scaling to BATCH_SIZE={BATCH_SIZE}, ACCUM_STEPS={ACCUM_STEPS}.")
    else:
        print(f"Omni Config: Detected {vram_gb:.1f}GB VRAM. Keeping defaults.")
else:
    print("Omni Config: CUDA not initialized yet.")