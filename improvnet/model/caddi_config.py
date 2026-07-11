import torch
import os

RUN_NAME = "caddi_ar_diffusion_v1_1d"
SAVE_DIR = "/gpfs/scratch/acw769/improvnet/artifacts/caddi_diffusion_small"
os.makedirs(SAVE_DIR, exist_ok=True)

RESUME_TRAINING = False

# Flattened vocab size
VOCAB_SIZE = 67761 
GENRES = ["classical", "jazz", "blues", "unknown"]
NUM_GENRES = len(GENRES)

# --- SEQUENCE MATH (Updated for 1D Flattening) ---
SEQ_LEN = 2048 
BLOCK_SIZE = 256 
PROMPT_MAX = 1024 

# 1D Special Token IDs
PAD_ID = 2
MASK_ID = 5
BLANK_ID = 6
SEP_ID = 7

# --- ARCHITECTURE MATH (Updated for 1D Concentration) ---
EMBED_DIM = 1024
N_HEADS = 16       # 1024 / 16 = 64 head_dim (Perfect for Tensor Cores and RoPE!)
N_KV_HEADS = 4     # Grouped Query Attention (4 query heads per KV head)
N_LAYERS = 20

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 32
ACCUM_STEPS = 1
LR = 1e-4 
WARMUP_STEPS = 5000 
N_STEPS = 150000 
GRAD_CLIP = 1.0

LOG_EVERY = 10
VAL_EVERY = 10000

JSONL_FILES = [
    "/data/scratch/acw769/improvnet/artifacts/data/misc_data_tokenized.jsonl"
]

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gb < 60.0:
        BATCH_SIZE = 16
        ACCUM_STEPS = 2
        RUN_NAME = RUN_NAME + "_small"
        SAVE_DIR = os.path.join(SAVE_DIR, "small")
        print(f"CaDDi Config: Detected {vram_gb:.1f}GB VRAM. Scaling to BATCH_SIZE={BATCH_SIZE}, ACCUM_STEPS={ACCUM_STEPS}.")
    else:
        print(f"CaDDi Config: Detected {vram_gb:.1f}GB VRAM. Keeping defaults.")
else:
    print("CaDDi Config: CUDA not initialized yet.")