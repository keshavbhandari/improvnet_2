import torch
import os

RUN_NAME = "caddi_ar_diffusion_v1"
SAVE_DIR = "/gpfs/scratch/acw769/improvnet/artifacts/caddi_diffusion"
os.makedirs(SAVE_DIR, exist_ok=True)

RESUME_TRAINING = False

VOCAB_SIZES = [91, 137, 22, 509, 510]
GENRES = ["classical", "jazz", "blues", "unknown"]
NUM_GENRES = len(GENRES)

SEQ_LEN = 2048 # Max length of the unrolled trajectory (Prompt + 4 Blocks)
BLOCK_SIZE = 256 # Size of the active target block
PROMPT_MAX = 1024 # Maximum tokens retained for the prompt context
PAD_ID = 2
MASK_ID = 8

# --- ARCHITECTURE MATH ---
EMBED_DIM = 1920

# Perfectly balanced for 6 MRA coordinate groups (4 heads per group)
N_HEADS = 24  
# Retains Grouped Query Attention (GQA) efficiency while aligning with MRA
N_KV_HEADS = 6 
# Depth
N_LAYERS = 20

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 48 
ACCUM_STEPS = 1
LR = 1e-4 
WARMUP_STEPS = 5000 
N_STEPS = 150000 
GRAD_CLIP = 1.0

LOG_EVERY = 10
VAL_EVERY = 10000

JSONL_FILES = [
    # "/data/scratch/acw769/improvnet/artifacts/data/gigamidi_data_tokenized.jsonl",
    "/data/scratch/acw769/improvnet/artifacts/data/misc_data_tokenized.jsonl"
]

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gb < 60.0:
        BATCH_SIZE = 16
        ACCUM_STEPS = 4
        RUN_NAME = RUN_NAME + "_small"
        SAVE_DIR = os.path.join(SAVE_DIR, "small")
        print(f"CaDDi Config: Detected {vram_gb:.1f}GB VRAM. Scaling to BATCH_SIZE={BATCH_SIZE}, ACCUM_STEPS={ACCUM_STEPS}.")
    else:
        print(f"CaDDi Config: Detected {vram_gb:.1f}GB VRAM. Keeping defaults.")
else:
    print("CaDDi Config: CUDA not initialized yet.")