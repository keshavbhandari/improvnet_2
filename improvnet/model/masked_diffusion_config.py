"""AR-sized masked diffusion; values only, never launches training."""
from improvnet.model.ar_config import (
    VOCAB_SIZE, GENRES, NUM_GENRES, SEQ_LEN, PAD_ID, MASK_ID, BLANK_ID,
    EMBED_DIM, N_HEADS, N_KV_HEADS, N_LAYERS, BATCH_SIZE, ACCUM_STEPS, LR,
    WEIGHT_DECAY, BETAS, WARMUP_STEPS, MIN_LR, DECAY_STEPS, N_STEPS,
    GRAD_CLIP, LOG_EVERY, VAL_EVERY, CHECKPOINT_EVERY, JSONL_FILES,
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
