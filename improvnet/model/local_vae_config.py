"""Local roll VAE values/paths only. No model creation or import-time writes."""

DATA_ROOT = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/data"
DATASETS = {
    "gigamidi": DATA_ROOT + "/Final_GigaMIDI_V2.0_Final",
    "aria": DATA_ROOT + "/aria-midi-v1-deduped-ext",
    "pijama": DATA_ROOT + "/pijama-retranscribed",
    "doug": DATA_ROOT + "/doug_mcenzie_jazz",
    "classical_archives": DATA_ROOT + "/ClassicalArchives-MIDI-Collection",
    "maestro": DATA_ROOT + "/maestro-v3.0.0-midi",
    "symphonynet": DATA_ROOT + "/SymphonyNet_Dataset",
}
ARIA_METADATA = DATA_ROOT + "/aria-midi-v1-deduped-ext/aria-midi-v1-deduped-ext/metadata.json"
# Separate upstream genre CSV is not included in the uploaded MIDI ZIP.
GIGAMIDI_METADATA = None
INDEX_PATH = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/local_vae_dataset/midi_index.sqlite"
# Checkpoints: vae/<run>/; logs: vae/tensorboard/<run>/.
RUNS_DIR = "/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/vae"
TENSORBOARD_ROOT = RUNS_DIR + "/tensorboard"
RUN_NAME = "vae176_vector512"
SAVE_DIR = RUNS_DIR + "/" + RUN_NAME
DATA_SPLIT_SEED = 42
DATA_SPLIT_RATIOS = {"train": .97, "validation": .02, "test": .01}

COARSE_MS = 1000
FINE_MS = 10
PIANO_PROGRAMS = tuple(range(8))
MAX_FRAMES = 360000
# No genre/instrument/accompaniment conditioning in the VAE.
BASE_CHANNELS = 176
CHANNEL_MULTIPLIERS = (1, 2, 4)
LATENT_LAYOUT = "vector"
LATENT_CHANNELS = 512  # Vector width; stored as [512,1,1], one vector per second.
VECTOR_PROJECTION_CHANNELS = 4  # Internal grid features, not additional latent codes.
RES_BLOCKS = 2

SEED = 42
# Virtual epoch; sample collections uniformly, then files uniformly within them.
SAMPLES_PER_EPOCH = 50000
VALIDATION_SAMPLES = 3200
COLLECTION_WEIGHTS = {}  # Optional relative weights; missing collections use 1.
GIGAMIDI_DRUM_ONLY_FRACTION = .05  # Fixed file subset, not a per-batch probability.
ACTIVE_BIN_PROB = .90  # Keep 10% uniformly sampled bins, including real silence.
PITCH_AUGMENTATION = True
PITCH_SHIFT_RANGE = (-5, 6)  # Inclusive; drums never pitch-transposed.
CACHE_MB_PER_WORKER = 128
MAX_DATA_RETRIES = 16
LOG_DATA_ERRORS = False  # No per-file logs by default; retry metric remains visible.
NUM_WORKERS = 16

BATCH_SIZE = 64  # Per GPU; conservative initial setting, not a measured maximum.
VALIDATION_BATCH_SIZE = 16
MAX_VALIDATION_BATCHES = 100
ACCUM_STEPS = 1
N_STEPS = 200000
LR = 1e-4
MIN_LR = 1e-5
WARMUP_STEPS = 1000
BETAS = (.9, .95)
WEIGHT_DECAY = .01
GRAD_CLIP = 1.
KL_WEIGHT = 1e-4
KL_WARMUP_STEPS = 5000
ACTIVITY_WEIGHT = 1.
ONSET_WEIGHT = 2.
VELOCITY_WEIGHT = 5.
PEDAL_WEIGHT = 1.
CONSISTENCY_WEIGHT = .1
# Class-balanced loss shares, with extra weight on suppressing false positives.
# These are class loss shares, NOT calibrated onset/activity probabilities.
ACTIVITY_POSITIVE_FRACTION = .40
ONSET_POSITIVE_FRACTION = .20  # 80% of class-balanced onset loss penalizes negatives.

# Conservative event-level denoising. Targets/pedal/accompaniment remain clean.
SEMANTIC_CORRUPTION = True
SEMANTIC_NOTE_PROB = .10  # Expected proposal rate; no minimum per bin.
SEMANTIC_OPERATION_WEIGHTS = {"pitch": .2, "addition": .3, "timing": .4, "deletion": .1}
SEMANTIC_PITCH_SHIFT_MAX = 1  # Semitones; isolated notes/carries aren't shifted.
SEMANTIC_TIME_SHIFT_MS = 30  # Complete note shifts, no duration change/wraparound.
SEMANTIC_MIN_NOTES_FOR_DELETION = 4  # Never delete an isolated note or a carry.
SEMANTIC_WEIGHT = .5
LATENT_JITTER_WEIGHT = .5
LATENT_JITTER_MAX_STD = .10  # Uniform 0..10% of detached clean-batch channel std.
LATENT_JITTER_VALIDATION_STD = .10  # Fixed 10% stress test, not diffusion time.
ROBUSTNESS_WARMUP_STEPS = 5000  # Ramp auxiliary weights and training jitter std.
ACTIVITY_THRESHOLD = .95  # Initial metric threshold; recalibrate on the new run.
ONSET_THRESHOLD = .985
# Held-note continuation/gap settings require validation calibration after training;
# the previous spatial VAE's calibration does not certify this new architecture.
# Joint validation calibration; fixed-threshold tags above remain comparable.
# Select on clean + latent-jitter branches, not ambiguous semantic recovery.
CALIBRATE_THRESHOLDS = True
CALIBRATION_ACTIVITY_THRESHOLDS = (.3, .5, .6, .7, .8, .9, .95, .99)
CALIBRATION_ONSET_THRESHOLDS = (.5, .8, .9, .95, .98, .99, .995, .999, .9999)
CALIBRATION_MIN_RECALL = .99  # If unattainable, report it and select best F1.
LOG_EVERY = 10
VALIDATE_EVERY = 1000
SAVE_EVERY = 1000
RESUME_TRAINING = True
