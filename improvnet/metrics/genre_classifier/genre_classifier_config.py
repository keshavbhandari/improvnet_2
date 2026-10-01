"""Values and paths for genre-classifier architecture, data and training."""

from pathlib import Path

JSONL_FILES = (
    "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/data/misc_data_tokenized.jsonl",
    "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/data/gigamidi_data_tokenized.jsonl",
)
ARTIFACT_DIR = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/genre_classifier"

# Independent encoder architecture, matching the uniform-diffusion backbone.
# Vocabulary, padding ID and genre count are derived by the trainer.
EMBED_DIM = 2048
N_HEADS = 16
N_KV_HEADS = 4
N_LAYERS = 20
FFN_MULT = 8 / 3
DROPOUT = 0.1
GRADIENT_CHECKPOINTING = True

MANIFEST = str(Path(ARTIFACT_DIR) / "dataset_manifest.json")
SAVE_DIR = str(Path(ARTIFACT_DIR) / "encoder_run")
SEED = 42
MIN_LENGTH = 128
MAX_LENGTH = 4096  # Training crop upper bound and encoder/scoring length limit.
EVAL_LENGTH = 256
PITCH_AUGMENTATION = True
MAX_SEMITONE_SHIFT = 7

# Candidate-pool limits apply ONLY to training. Validation/test keep all
# known labels; unknown/unlabeled records are excluded from all folds.
TRAIN_CLASS_CAPS = {"classical": 30000}
# Class-balanced draws with replacement: retain every known category, give
# jazz/blues four times each other category's draw probability (~22.2% each
# with 12 labels). No second layer of inverse-frequency CE weighting.
FOCUS_GENRES = ("jazz", "blues")
FOCUS_SAMPLING_WEIGHT = 4.0
SAMPLES_PER_EPOCH = 200000  # Global draws, divided equally across DDP ranks.

BATCH_SIZE = 48  # Per GPU; 4096-token bidirectional encoder needs benchmarking.
ACCUM_STEPS = 1
VALIDATION_BATCH_SIZE = 16
MAX_VALIDATION_BATCHES = 0  # Full shard; rare classes matter.
NUM_WORKERS = 4
N_STEPS = 30000
# Fresh encoder training, not the low-rate AR-initialized diffusion SFT.
LR = 1.5e-4
MIN_LR = 1e-5
WARMUP_STEPS = 1000
WEIGHT_DECAY = .01
BETAS = (.9, .95)
GRAD_CLIP = 1.0
LOG_EVERY = 1
VALIDATE_EVERY = 1000
SAVE_EVERY = 1000
RESUME_TRAINING = True
TENSORBOARD = True
CALIBRATE = True
