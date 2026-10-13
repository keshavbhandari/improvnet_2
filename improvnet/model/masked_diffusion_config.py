"""AR-sized masked diffusion; values only, never launches training."""
from improvnet.model.ar_config import (
    VOCAB_SIZE, GENRES, NUM_GENRES, SEQ_LEN, PAD_ID, MASK_ID, BLANK_ID,
    EMBED_DIM, N_HEADS, N_KV_HEADS, N_LAYERS, JSONL_FILES,
    DATA_SPLIT_SEED, DATA_SPLIT_RATIOS, MIDI_FILEPATH_EXCLUDE_SUBSTRINGS,
)

RUN_NAME = "masked_diffusion_v1"
SAVE_DIR = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/" + RUN_NAME
RESUME_TRAINING = True
SEED = 42
NUM_WORKERS = 4
# Draw elasticity uniformly from [0, ELASTICITY_MAX] during training.
ELASTICITY_MAX = 1.0
JAZZ_BLUES_WEIGHT = 4.0
RHYTHM_DROP_PROB = .1

BATCH_SIZE = 25
ACCUM_STEPS = 1 # Optimized run, 2 for old run
LR = 1.5e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
WARMUP_STEPS = 8000
MIN_LR = 1e-5
DECAY_STEPS = 16000
N_STEPS = 140000
GRAD_CLIP = 1.0

LOG_EVERY = 1
VAL_EVERY = 10000
# Save independently of validation so long Slurm jobs can always resume.
CHECKPOINT_EVERY = 2000