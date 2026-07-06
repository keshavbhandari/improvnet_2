# ==========================================
# STAGE 1: FROZEN AUTOENCODER SPECS
# ==========================================
# AE_CHECKPOINT = "/gpfs/scratch/acw769/improvnet/artifacts/autoencoder/latest_checkpoint.pt"
AE_CHECKPOINT = "/gpfs/scratch/acw769/improvnet/artifacts/autoencoder_v2_8patch_128latent_0.05noise/latest_checkpoint.pt"
PATCH_SIZE = 8
LATENT_DIM = 128

# ==========================================
# STAGE 2: FLOW MATCHING SPECS
# ==========================================
# The FM model processes the CONTINUOUS latents, not the raw tokens.
# 2048 latents * 8 tokens/patch = generating up to 16,384 compound notes!
MAX_LATENT_SEQ_LEN = 2048 

# FM Transformer Architecture
FM_HIDDEN_DIM = 1024
FM_LAYERS = 16
FM_HEADS = 16
FM_FFN_MULT = 4.0

# Continuous Flow Parameters
SIGMA_MIN = 1e-5     # Minimum noise level
NUM_INFERENCE_STEPS = 50  # Steps to run the ODE solver during generation

# ==========================================
# CONDITIONING & CLASSIFIER-FREE GUIDANCE
# ==========================================
NUM_INSTRUMENT_CLASSES = 40

# Dropout probabilities during training (to give the model creative autonomy)
# If a condition is dropped, we replace its latents with a learned [NULL_COND] vector
P_DROP_MELODY = 0.15
P_DROP_HARMONY = 0.15
P_DROP_RHYTHM = 0.15
P_DROP_INST = 0.10

# Probability of completely unconditional generation (dropping ALL conditions)
P_UNCOND = 0.10 

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 64 
ACCUM_STEPS = 1
LR = 1e-4
N_STEPS = 1_000_000
WARMUP_STEPS = 5000
LOG_EVERY = 10
VAL_EVERY = 10_000
GRAD_CLIP = 1.0
USE_CKPT = True  # Highly recommended at 16 layers/1024 dim
RESUME_TRAINING = False   # Set to True to load from checkpoint
SAVE_DIR = "/gpfs/scratch/acw769/improvnet/artifacts/flow_matching_v2"
RUN_NAME = "fm_conditional_v2"
JSONL_FILES = [
    # "/data/scratch/acw769/improvnet/artifacts/data/gigamidi_data.jsonl",
    # "/data/scratch/acw769/improvnet/artifacts/data/misc_data.jsonl"
    "/data/scratch/acw769/improvnet/artifacts/data/gigamidi_data_tokenized.jsonl",
    "/data/scratch/acw769/improvnet/artifacts/data/misc_data_tokenized.jsonl"
]