import torch
import os

RUN_NAME = "ar_context_pretrain_v1"
SAVE_DIR = "/gpfs/scratch/acw769/improvnet/artifacts/ar_context"
os.makedirs(SAVE_DIR, exist_ok=True)

RESUME_TRAINING = True

# --- VOCABULARY ---
VOCAB_SIZE = 67761 
GENRES = ["classical", "jazz", "blues", "unknown"]
NUM_GENRES = len(GENRES)

# --- SEQUENCE MATH ---
# We train the AR context model on full 2048-token sequences.
SEQ_LEN = 2048 

# Special Tokens
PAD_ID = 2
MASK_ID = 5
BLANK_ID = 6
SEP_ID = 7

# --- ARCHITECTURE MATH ---
EMBED_DIM = 1024
N_HEADS = 16       # 1024 / 16 = 64 head_dim
N_KV_HEADS = 4     # Grouped Query Attention (4 queries per KV)
N_LAYERS = 20

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
# AR training is highly efficient, so we can use larger batch sizes 
# or sequences compared to the complex unrolled diffusion model.
BATCH_SIZE = 32 
ACCUM_STEPS = 1
LR = 1.5e-4 
WARMUP_STEPS = 5000 
N_STEPS = 200000 
GRAD_CLIP = 1.0

LOG_EVERY = 10
VAL_EVERY = 20000

JSONL_FILES = [
    "/data/scratch/acw769/improvnet/artifacts/data/misc_data_tokenized.jsonl"
]

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gb < 60.0:
        BATCH_SIZE = 8
        ACCUM_STEPS = 2
        RUN_NAME = RUN_NAME + "_small"
        SAVE_DIR = os.path.join(SAVE_DIR, "small")
        print(f"AR Config: Detected {vram_gb:.1f}GB VRAM. Scaling to BATCH_SIZE={BATCH_SIZE}, ACCUM_STEPS={ACCUM_STEPS}.")
    else:
        print(f"AR Config: Detected {vram_gb:.1f}GB VRAM. Keeping defaults.")
else:
    print("AR Config: CUDA not initialized yet.")