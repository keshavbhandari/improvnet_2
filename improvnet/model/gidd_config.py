import torch

VOCAB_SIZES    = [91, 137, 22, 509, 510] # instrument, pitch, velocity, onset, dur
NUM_INSTRUMENT_CLASSES = 40

# --- GENRE CONDITIONING ---
GENRES = ["classical", "jazz", "blues", "unknown"]
NUM_GENRES = len(GENRES)

# --- DISCRETE DIFFUSION HYPERPARAMETERS ---
SEQ_LEN = 2048
BLOCK_SIZE = 256
PAD_ID = 2
MASK_ID = 8

# --- ARCHITECTURE MATH ---
EMBED_DIM = 1920 #960
N_HEADS = 24 # 6 MRA coordinate groups (4 heads per group)
N_KV_HEADS = 6 

# --- DEPTH ---
N_LAYERS = 20

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 48 
ACCUM_STEPS = 1
LR = 1e-4 
N_STEPS = 1_000_000
WARMUP_STEPS = 5000
LOG_EVERY = 10
VAL_EVERY = 20_000
GRAD_CLIP = 1.0
USE_CKPT = True
RESUME_TRAINING = False   
SAVE_DIR = "/gpfs/scratch/acw769/improvnet/artifacts/block_diffusion_gidd"
RUN_NAME = "model_diffusion_v1_gidd"
JSONL_FILES = [
    # "/data/scratch/acw769/improvnet/artifacts/data/gigamidi_data_tokenized.jsonl",
    "/data/scratch/acw769/improvnet/artifacts/data/misc_data_tokenized.jsonl"
]

# Dynamically scale batch size for ~40-48GB GPUs (e.g., A40, RTX 6000)
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gb < 60.0:
        BATCH_SIZE = 16
        ACCUM_STEPS = 2
        RUN_NAME = RUN_NAME + "_small"
        print(f"Config: Detected {vram_gb:.1f}GB VRAM. Scaling to BATCH_SIZE={BATCH_SIZE}, ACCUM_STEPS={ACCUM_STEPS}.")
    else:
        print(f"Config: Detected {vram_gb:.1f}GB VRAM. Keeping 80GB+ defaults.")
else:
    print("Config: CUDA not initialized yet. Defaulting to BATCH_SIZE=64.")