import torch
import os
from improvnet.tokenizer.absolute import AbsTokenizer

RUN_NAME = "ar_context_split_instrument_v1"
SAVE_DIR = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/ar_context_split_instrument"
os.makedirs(SAVE_DIR, exist_ok=True)

RESUME_TRAINING = False

# --- VOCABULARY ---
VOCAB_SIZE = AbsTokenizer().vocab_size
GENRES = ["classical", "jazz", "blues", "unknown"]
NUM_GENRES = len(GENRES)

# --- DATA SPLITS ---
DATA_SPLIT_SEED = 42
DATA_SPLIT_RATIOS = {"train": 0.97, "validation": 0.02, "test": 0.01}

# --- SEQUENCE MATH ---
# We train the AR context model on full 2048-token sequences.
SEQ_LEN = 8192 

# Special Tokens
PAD_ID = 2
MASK_ID = 5
BLANK_ID = 6
SEP_ID = 7

# --- ARCHITECTURE MATH ---
EMBED_DIM = 2048
N_HEADS = 16       # 1024 / 16 = 64 head_dim
N_KV_HEADS = 4     # Grouped Query Attention (4 queries per KV)
N_LAYERS = 20

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
# AR training is highly efficient, so we can use larger batch sizes 
# or sequences compared to the complex unrolled diffusion model.
BATCH_SIZE = 24
ACCUM_STEPS = 1 # Optimized run, 2 for old run
LR = 2e-4 # Optimized run
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
WARMUP_STEPS = 10000 
N_STEPS = 800000 
GRAD_CLIP = 1.0

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
    "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/data/misc_data_tokenized.jsonl",
    # "/data/scratch/acw769/improvnet/artifacts/data/gigamidi_data_tokenized.jsonl"
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
