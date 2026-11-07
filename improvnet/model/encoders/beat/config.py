"""Beat encoder: aligned disjoint instrument groups; standalone training settings."""

# Data and checkpoint paths
MANIFEST = '/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/jazz_blues_data/prepared/melody_manifest_v5.json'  # Dataset catalogue: source locations, piece splits and deduplication metadata; fixed for this encoder cohort.
SAVE_DIR = '/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/beat'  # Directory for model checkpoints, metrics and TensorBoard logs.
DATASET_CLASS = 'improvnet.model.encoders.beat.data.BeatPairDataset'  # Dataset implementation that constructs contrastive pairs.
COLLATOR_CLASS = 'improvnet.model.encoders.beat.data.BeatCollator'  # Batch builder including same-piece temporal negatives.
CHECKPOINT_FORMAT = 'beat_v1'  # Checkpoint identifier checked when loading the frozen encoder.
OBJECTIVE = 'beat'  # Use the beat objective with temporal and cross-piece negatives.
AUGMENTATION_DESCRIPTION = 'disjoint aligned instrument groups; independent +/-7 pitch shifts and 15-45% span masking; same-piece distant-window and other-piece negatives'  # Human-readable augmentation policy saved in checkpoints.

# Encoder architecture
EMBED_DIM = 768  # Transformer hidden width.
N_HEADS = 12  # Number of attention query heads.
N_KV_HEADS = 4  # Number of shared key/value heads.
N_LAYERS = 8  # Number of bidirectional transformer blocks.
FFN_MULT = 8 / 3  # Feed-forward width relative to the hidden width.
DROPOUT = .1  # Dropout probability during training.
GRADIENT_CHECKPOINTING = True  # Recompute block activations to reduce GPU memory.
PROJECTION_DIM = 256  # Size of the unit-normalized contrastive embedding.

# Token windows and augmentation
MIN_LENGTH = 64  # Minimum number of tokens per sampled view.
MAX_LENGTH = 512  # Maximum number of tokens per training view.
EVAL_LENGTH = 512  # Token limit per validation view.
PITCH_AUGMENTATION = True  # Independently transpose positive views to discourage key matching.
MAX_SEMITONE_SHIFT = 7  # Maximum absolute transposition in semitones.
TEMPERATURE = .1  # Contrastive softmax temperature.
SEED = 42  # Seed for reproducible sampling and initialization.

# Batches and data loading
SAMPLES_PER_EPOCH = 196608  # Number of sampled training pairs in each virtual epoch.
BATCH_SIZE = 64  # Pairs per GPU; four GPUs give 256 pairs per optimizer update.
ACCUM_STEPS = 1  # Batches accumulated before each optimizer update.
VALIDATION_BATCH_SIZE = 48  # Validation pairs per GPU batch.
MAX_VALIDATION_BATCHES = 0  # Zero evaluates the entire held-out fold, distributed across GPUs.
NUM_WORKERS = 4  # Data-loader workers per GPU process.

# Complete learning-rate schedule and optimizer
N_STEPS = 30000  # Total optimizer updates from fresh initialization.
DECAY_STEPS = 20000  # Reach MIN_LR at this update before the final cosine tail.
TAIL_END_LR = 5e-6  # Learning rate at the end of the full training schedule.
LR = 1.5e-4  # Peak learning rate after warmup.
MIN_LR = 1e-5  # Learning rate reached at the end of the main cosine decay.
WARMUP_STEPS = 500  # Initial updates used to increase the learning rate.
WEIGHT_DECAY = .01  # AdamW parameter regularization.
BETAS = (.9, .95)  # AdamW first- and second-moment decay coefficients.
GRAD_CLIP = 1.  # Maximum gradient norm.

# Reporting and checkpointing
LOG_EVERY = 10  # Optimizer updates between training metric reports.
VALIDATE_EVERY = 500  # Optimizer updates between validation evaluations.
SAVE_EVERY = 500  # Optimizer updates between checkpoint saves.
RESUME_TRAINING = True  # Resume the latest checkpoint if present; otherwise start fresh.
TENSORBOARD = True  # Write training and validation events for TensorBoard.

# Positive-pair sampling and masking
ALIGNED_PAIR_PROB = 1.  # Fraction of positive pairs sampled from the same time window.
PIANO_DRUM_DRAW_PROB = .5  # Probability of preferentially drawing a piano-and-drums piece.
JAZZ_PREFERENCE_WITHIN_PIANO_DRUMS = .5  # Jazz preference within the preferential piano-and-drums draw.
MASK_PROB = .9  # Probability of applying span masking independently to each view.
MASK_RATE_RANGE = (.15, .45)  # Uniform range of interior-token masking fractions when masking is applied.

NEGATIVE_GAP_MS = 1000  # Minimum temporal gap for a same-piece negative window.
DATASET_OPTIONS = dict(  # Explicit arguments supplied to the dataset constructor.
    mask_prob=MASK_PROB,  # Probability of applying token-span masking to a view.
    mask_rate_range=MASK_RATE_RANGE,  # Range of masked token fractions for augmented views.
    piano_drum_draw_prob=PIANO_DRUM_DRAW_PROB,  # Preferential sampling probability for piano-and-drums pieces.
    jazz_piano_drum_prob=JAZZ_PREFERENCE_WITHIN_PIANO_DRUMS,  # Jazz preference within piano-and-drums sampling.
    negative_gap_ms=NEGATIVE_GAP_MS,  # Temporal separation required for same-piece negatives.
)  # End of dataset sampling and augmentation options.
