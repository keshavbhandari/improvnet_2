"""Values and paths only; importing settings never starts a job."""

from pathlib import Path

JSONL_FILES = (
    "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/data/misc_data_tokenized.jsonl",
    "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/data/gigamidi_data_tokenized.jsonl",
)
# Match the AR/diffusion corpus filtering; retain normal mixtures with drums.
MIDI_FILEPATH_EXCLUDE_SUBSTRINGS = {JSONL_FILES[1]: ("/drums-only/",)}
ARTIFACT_DIR = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/continuity"
MANIFEST = str(Path(ARTIFACT_DIR) / "dataset_manifest.json")
SAVE_DIR = str(Path(ARTIFACT_DIR) / "encoder_run")

EMBED_DIM = 2048
N_HEADS = 16
N_KV_HEADS = 4
N_LAYERS = 20
FFN_MULT = 8 / 3
DROPOUT = 0.1
GRADIENT_CHECKPOINTING = True
PROJECTION_DIM = 256
MIN_LENGTH = 128
MAX_LENGTH = 512
EVAL_LENGTH = 256
PITCH_AUGMENTATION = True
MAX_SEMITONE_SHIFT = 7
TEMPERATURE = 0.1  # Fixed contrastive-logit temperature, not sampling temperature.

SEED = 42
SAMPLES_PER_EPOCH = 200000  # Global pair draws; no genre balancing/conditioning.
BATCH_SIZE = 160  # Pairs per GPU => twice as many encoded excerpts; benchmark first.
ACCUM_STEPS = 1
VALIDATION_BATCH_SIZE = 48
MAX_VALIDATION_BATCHES = 100
NUM_WORKERS = 4
N_STEPS = 30000
LR = 1.5e-4
MIN_LR = 1e-5
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
LOG_EVERY = 1
VALIDATE_EVERY = 1000
SAVE_EVERY = 1000
RESUME_TRAINING = True
TENSORBOARD = True
