import torch
import os

RUN_NAME = "ar_context_pretrain_v1" # Old run
SAVE_DIR = "/gpfs/scratch/acw769/improvnet/artifacts/ar_context" # Old run
# RUN_NAME = "ar_context_pretrain_v1_optimized" # Optimized run
# SAVE_DIR = "/gpfs/scratch/acw769/improvnet/artifacts/ar_context_optimized" # Optimized run
os.makedirs(SAVE_DIR, exist_ok=True)

RESUME_TRAINING = True

# --- VOCABULARY ---
VOCAB_SIZE = 67761 
GENRES = ["classical", "jazz", "blues", "unknown"]
NUM_GENRES = len(GENRES)

# --- SEQUENCE MATH ---
# We train the AR context model on full 2048-token sequences.
SEQ_LEN = 8192 

# Special Tokens
PAD_ID = 2
MASK_ID = 5
BLANK_ID = 6
SEP_ID = 7

# --- ARCHITECTURE MATH ---
EMBED_DIM = 1536
N_HEADS = 16       # 1024 / 16 = 64 head_dim
N_KV_HEADS = 4     # Grouped Query Attention (4 queries per KV)
N_LAYERS = 20

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
# AR training is highly efficient, so we can use larger batch sizes 
# or sequences compared to the complex unrolled diffusion model.
BATCH_SIZE = 19 # Old run #20
# BATCH_SIZE = 20 # Optimized run
ACCUM_STEPS = 1 # Optimized run, 2 for old run
LR = 1.0e-4 # Old run
# LR = 1.5e-4 # Optimized run
WARMUP_STEPS = 10000 
N_STEPS = 800000 
GRAD_CLIP = 1.0
LM_HEAD_CHUNK_SIZE = 2048

# OPTIMIZER_BACKEND = "paged_adamw8bit"
# ALLOW_OPTIMIZER_MIGRATION_TO_8BIT = True

OPTIMIZER_BACKEND = "adamw"
ALLOW_OPTIMIZER_MIGRATION_TO_8BIT = False

# Set these only when resuming a legacy checkpoint that predates saved batch/accum metadata.
# RESUME_CHECKPOINT_BATCH_SIZE = 7
# RESUME_CHECKPOINT_ACCUM_STEPS = 2
# RESUME_CHECKPOINT_WORLD_SIZE = 4

LOG_EVERY = 1
VAL_EVERY = 10000

JSONL_FILES = [
    "/data/scratch/acw769/improvnet/artifacts/data/misc_data_tokenized.jsonl",
    "/data/scratch/acw769/improvnet/artifacts/data/gigamidi_data_tokenized.jsonl"
]

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gb < 60.0:
        BATCH_SIZE = 7
        ACCUM_STEPS = 2
        print(f"AR Config: Detected {vram_gb:.1f}GB VRAM. Scaling to BATCH_SIZE={BATCH_SIZE}, ACCUM_STEPS={ACCUM_STEPS}.")
    else:
        print(f"AR Config: Detected {vram_gb:.1f}GB VRAM. Keeping defaults.")
else:
    print("AR Config: CUDA not initialized yet.")
