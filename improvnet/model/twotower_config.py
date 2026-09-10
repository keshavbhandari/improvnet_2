import torch
import os
from improvnet.tokenizer.absolute import AbsTokenizer

RUN_NAME = "twotower_split_instrument_v1"
SAVE_DIR = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/twotower_split_instrument"
os.makedirs(SAVE_DIR, exist_ok=True)

# Start from latest_checkpoint.pt when present; otherwise start a fresh run.
RESUME_TRAINING = True

# --- VOCABULARY ---
VOCAB_SIZE = AbsTokenizer().vocab_size
GENRES = ["classical", "jazz", "blues", "unknown"]
NUM_GENRES = len(GENRES)
NUM_INSTRUMENTS = 41 # Matches AR Context config

# --- DATA SPLITS ---
DATA_SPLIT_SEED = 42
DATA_SPLIT_RATIOS = {"train": 0.97, "validation": 0.02, "test": 0.01}

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
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
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

AR_MODEL_PATH = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/ar_context_split_instrument/latest_checkpoint.pt"

LOG_EVERY = 10
VAL_EVERY = 20000
# Save independently of validation so long Slurm jobs can always resume.
CHECKPOINT_EVERY = 2000

JSONL_FILES = [
    "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/data/misc_data_tokenized.jsonl"
]

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gb < 60.0:
        BATCH_SIZE = 8
        ACCUM_STEPS = 4
else:
    print("TwoTower Config: CUDA not initialized yet.")
