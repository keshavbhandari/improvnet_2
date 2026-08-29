import torch
import os

RUN_NAME = "twotower_caddi_hybrid_v1"
SAVE_DIR = "/gpfs/scratch/acw769/improvnet/artifacts/twotower_hybrid"
os.makedirs(SAVE_DIR, exist_ok=True)

RESUME_TRAINING = False

# --- VOCABULARY ---
VOCAB_SIZE = 67761 
GENRES = ["classical", "jazz", "blues", "unknown"]
NUM_GENRES = len(GENRES)
NUM_INSTRUMENTS = 41 # Matches AR Context config

# --- SEQUENCE MATH ---
# By shifting prefix to 1024, we leave ~1024 tokens for 4 sequential drafts
BLOCK_SIZE = 256 
PROMPT_MAX = 1024
NUM_DRAFTS = 4 

# Special Tokens
PAD_ID = 2
MASK_ID = 5
BLANK_ID = 6
SEP_ID = 7

# --- ARCHITECTURE MATH ---
# Must EXACTLY match the AR Context Tower so the KV caches align perfectly
EMBED_DIM = 1536
N_HEADS = 16       
N_KV_HEADS = 4     
N_LAYERS = 20

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 8 
ACCUM_STEPS = 4
LR = 1e-4 
WARMUP_STEPS = 10000
N_STEPS = 800000
GRAD_CLIP = 1.0
DIFFUSION_STEPS = 16 
OPTIMIZER_BACKEND = "adamw"
ALLOW_OPTIMIZER_MIGRATION_TO_8BIT = False
# Set these only when resuming a legacy checkpoint that predates saved batch/accum/world metadata.
# RESUME_CHECKPOINT_BATCH_SIZE = 8
# RESUME_CHECKPOINT_ACCUM_STEPS = 4
# RESUME_CHECKPOINT_WORLD_SIZE = 4

AR_MODEL_PATH = "/gpfs/scratch/acw769/improvnet/artifacts/ar_context/latest_checkpoint.pt"

LOG_EVERY = 10
VAL_EVERY = 20000

JSONL_FILES = [
    "/data/scratch/acw769/improvnet/artifacts/data/misc_data_tokenized.jsonl"
]

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gb < 60.0:
        BATCH_SIZE = 8
        ACCUM_STEPS = 4
else:
    print("TwoTower Config: CUDA not initialized yet.")
