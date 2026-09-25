import torch
import os
from improvnet.tokenizer.absolute import AbsTokenizer

RUN_NAME = "twotower_split_instrument_v1_test"
SAVE_DIR = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/twotower_split_instrument_test"
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
# The target block is sampled first; up to PROMPT_MAX preceding tokens become
# Tower A context. Tower B refines the same target across NUM_DRAFTS drafts.
BLOCK_SIZE = 256
PROMPT_MAX = 4096
NUM_DRAFTS = 4 

# Structured denoising augmentations. Elastic examples reserve a variable
# fraction of the target for four-token <BLANK> event slots. The realized
# fraction is passed to Tower B as an explicit generation-time control.
ELASTICITY_MIN_RATIO = 0.10
ELASTICITY_MAX_RATIO = 0.20
ELASTICITY_MULTITRACK_PROB = 0.8
ELASTICITY_SOLO_PROB = 0.3

# On multi-instrument examples, remove one complete, conditioned instrument
# stem from every draft and train the denoiser to reconstruct it.
STEM_REMOVAL_PROB = 0.8

# Train Tower B to operate without any Tower A context for this fraction of
# batches. Genre and desired instruments are still supplied directly to B.
PROMPTLESS_BATCH_PROB = 0.05

# Visible-token corruption teaches later drafts to revise plausible mistakes,
# not only fill masks. Most replacements preserve token type; a small fraction
# deliberately uses the wider vocabulary to expose the model to bad structure.
NON_MASK_CORRUPTION_PROBS = (0.15, 0.10, 0.05, 0.025)
FULL_VOCAB_CORRUPTION_PROB = 0.075

# If both <D>/<E> are in the target, one is kept visible while the other is
# masked. If only one is present, ordinary diffusion plus these additional
# probabilities provide explicit reconstruction practice.
BOUNDARY_TOKEN_MASK_PROBS = (0.75, 0.50, 0.25, 0.10)

# Simple weighted reconstruction objective.
CLEAN_TOKEN_LOSS_WEIGHT = 0.05
CORRUPTED_TOKEN_LOSS_WEIGHT = 1.0
MASKED_TOKEN_LOSS_WEIGHT = 2.0

# Special Tokens
PAD_ID = 2
MASK_ID = 5
BLANK_ID = 6
SEP_ID = 7

# --- ARCHITECTURE MATH ---
# Must EXACTLY match the AR Context Tower so the KV caches align perfectly
EMBED_DIM = 2048
N_HEADS = 16       
N_KV_HEADS = 4     
N_LAYERS = 20

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 24
ACCUM_STEPS = 1
LR = 1.5e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
WARMUP_STEPS = 4000 #8000
MIN_LR = 1e-5
DECAY_STEPS = 4000
N_STEPS = 40000 #200000
GRAD_CLIP = 1.0
DIFFUSION_STEPS = 16 

AR_MODEL_PATH = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/ar_context_split_instrument/latest_checkpoint.pt"

LOG_EVERY = 1
VAL_EVERY = 20000
# Save independently of validation so long Slurm jobs can always resume.
CHECKPOINT_EVERY = 2000

MISC_JSONL = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/data/misc_data_tokenized.jsonl"
GIGAMIDI_JSONL = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/data/gigamidi_data_tokenized.jsonl"

JSONL_FILES = [
    MISC_JSONL,
    GIGAMIDI_JSONL,
]

# Keep no-drums and all-instruments-with-drums; exclude only drums-only files.
MIDI_FILEPATH_EXCLUDE_SUBSTRINGS = {
    GIGAMIDI_JSONL: ("/drums-only/",),
}

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gb < 60.0:
        BATCH_SIZE = 8
        ACCUM_STEPS = 4
else:
    print("TwoTower Config: CUDA not initialized yet.")
