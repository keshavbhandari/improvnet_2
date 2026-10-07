"""Independent SFT settings; importing this file does not start a job.

The batch was smoke-tested on a shared GH200 with full 4096-token history.
The learning rate and total update budget still require learning-curve tuning.
"""

from improvnet.archive.uniform_diffusion.model.uniform_diffusion_config import UniformDiffusionConfig

MODEL = UniformDiffusionConfig()
AR_MODEL_PATH = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/ar_context_split_instrument/latest_checkpoint.pt"
SAVE_DIR = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/uniform_diffusion_v1"
JSONL_FILES = [
    "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/data/misc_data_tokenized.jsonl",
    "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/data/gigamidi_data_tokenized.jsonl",
]
MIDI_FILEPATH_EXCLUDE_SUBSTRINGS = {JSONL_FILES[1]: ("/drums-only/",)}
DATA_SPLIT_SEED = 42
DATA_SPLIT_RATIOS = {"train": 0.97, "validation": 0.02, "test": 0.01}
SEED = 42
PROMPT_MAX = 4096
PROMPTLESS_PROB = 0.05
END_ANCHORED_TARGET_PROB = 0.05
SELF_CONDITIONING_PROB = 0.5
# Released SFT encodes raw previous logits with softmax (temperature 1).
# The paper's 0.8 -> 0.4 schedule belongs to inference, not this SFT pass.
SELF_CONDITIONING_TEMPERATURE = 1.0
AR_LOSS_WEIGHT = 1.0
CE_CHUNK_SIZE = 256
PITCH_AUGMENTATION = True

# Per GPU; retains headroom for distributed buffers on a dedicated 95 GiB GPU.
BATCH_SIZE = 32
ACCUM_STEPS = 1
VALIDATION_BATCH_SIZE = 16
MAX_VALIDATION_BATCHES = 100
NUM_WORKERS = 4
N_STEPS = 40000
LR = 2e-5
MIN_LR = 2e-6
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
LOG_EVERY = 1
VALIDATE_EVERY = 1000
SAVE_EVERY = 1000
RESUME_TRAINING = True
