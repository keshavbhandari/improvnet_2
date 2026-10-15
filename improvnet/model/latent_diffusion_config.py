"""Stage-2 values and paths. Importing this module never starts training."""
from improvnet.model.local_vae_config import (
    DATA_ROOT, DATASETS, ARIA_METADATA, GIGAMIDI_METADATA, INDEX_PATH,
    DATA_SPLIT_SEED, DATA_SPLIT_RATIOS, COARSE_MS, FINE_MS, PIANO_PROGRAMS, MAX_FRAMES,
)

# Stage-2-only metadata/index. The frozen VAE's original index stays unchanged.
GIGAMIDI_METADATA = DATASETS["gigamidi"] + "/Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv"
GIGAMIDI_GENRE_FIELDS = ("music_styles_curated",)
INDEX_PATH = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/local_vae_dataset/midi_index_gigamidi_genres.sqlite"

VAE_CHECKPOINT = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/vae/vae176_vector512/best_model.pt"
# Pinned best-checkpoint snapshot: ongoing VAE training cannot replace this file.
# Fresh stage-2 run: older diffusion checkpoints use a different frozen encoder.
SAVE_DIR = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/latent_diffusion"
EVALUATION_DIR = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/latent_diffusion_evaluation"

HISTORY_BINS = 8  # Up to 8 seconds of clean context, separate from the target.
TARGET_BINS = 30  # 30 seconds generated; 38 tokens total for vector [512,1,1].
PATCH_PITCH = 1
PATCH_TIME = 1
D_MODEL = 512
N_LAYERS = 16
N_HEADS = 8
FF_MULTIPLIER = 4
GRADIENT_CHECKPOINTING = True
MAX_ACCOMPANIMENT_TRACKS = 3
MAX_ACCOMPANIMENT_EVENTS = 1024  # One compound token per note, across the window.
RHYTHM_FRAMES_PER_TOKEN = 10  # 100 ms per conditioning token with the default clock.
ACCOMPANIMENT_MODE_PROBS = {"full": .45, "rhythm": .45, "none": .10}
PROMPTLESS_PROB = .05
PITCH_DROP_PROB = .10  # Whole reference unavailable; not silence.
PITCH_NOTE_DROP_PROB = .10  # Erase selected reference bits; never restrict outputs.
GENRE_DROP_PROB = .10

SEED = 42
SAMPLES_PER_EPOCH = 50000
VALIDATION_SAMPLES = 1024
COLLECTION_WEIGHTS = {}
GIGAMIDI_DRUM_ONLY_FRACTION = .05
ACTIVE_BIN_PROB = .90
PITCH_AUGMENTATION = True
PITCH_SHIFT_RANGE = (-5, 6)
CACHE_MB_PER_WORKER = 128
MAX_DATA_RETRIES = 16
LOG_DATA_ERRORS = False
NUM_WORKERS = 16
LATENT_CALIBRATION_SAMPLES = 512  # Training windows;
VAE_BATCH_SIZE = 16

BATCH_SIZE = 32  # Windows per GPU, not individual VAE bins; not yet benchmarked.
VALIDATION_BATCH_SIZE = 4
MAX_VALIDATION_BATCHES = 100
ACCUM_STEPS = 1
N_STEPS = 100000  # Optimizer updates; independent of the 30-second audio window.
LR = 1e-4
MIN_LR = 1e-5
WARMUP_STEPS = 1000
BETAS = (.9, .95)
WEIGHT_DECAY = .01
GRAD_CLIP = 1.
EMA_DECAY = .999
LOG_EVERY = 1
VALIDATE_EVERY = 1000
SAVE_EVERY = 1000
RESUME_TRAINING = True
ACTIVITY_THRESHOLD = .95  # Inference uses the frozen VAE's saved calibration by default.
ONSET_THRESHOLD = .985
