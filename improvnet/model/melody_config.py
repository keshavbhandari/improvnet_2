"""V14: original AR decoder, paired-window reference prefix and a genre curriculum."""
from pathlib import Path
from improvnet.utils.melody import MelodyTokenizer, TASKS


# Tasks and sequence format
PROTOCOL_VERSION = 18  # Projected-prefix reference-memory training/inference format.
TASK_NAMES = TASKS  # Canonical refinement, harmonize and stem task names.
TASK_WEIGHTS = (.20, .40, .40)  # Relative sampling probabilities for refinement, harmonize and stem.

# Corpus paths and sampling
DATA_ROOT = Path('/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/jazz_blues_data')  # Root of the MIDI corpus and prepared metadata.
DATA_FILE = DATA_ROOT / 'prepared' / 'melody_manifest_v7.json'  # Finetuning split and sampling manifest.
EXTRA_PERFORMANCE_INDEX = DATA_ROOT / 'prepared' / 'jazz_performance_v5.index.json'  # Index of additional jazz performance MIDI recordings.
PREPARED_DATA_FILE = DATA_ROOT / 'prepared' / 'performances_stems_v4.jsonl'  # Prepared multitrack performance metadata.
BASE_DATA_FILE = DATA_ROOT / 'prepared' / 'performances_v3.jsonl'  # Original performance metadata fallback.
RECOVERED_DATA_FILE = DATA_ROOT / 'prepared' / 'recovered_labelled.jsonl'  # Recovered files with usable genre labels.
PARSED_CACHE = DATA_ROOT / 'prepared' / 'round2_lazy_cache'  # Reuse parsed notes; construct task examples dynamically.
FALLBACK_PARSED_CACHE = Path('/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/beat_annotations/parsed_cache')  # Reuse existing plain-note parsing cache; no annotation database is read.
CLASSICAL_INDEX = Path('/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/local_vae_dataset/midi_index_gigamidi_genres.sqlite')  # SQLite index used to sample additional non-jazz/blues MIDI files.
CLASSICAL_RATIO = 2  # Non-jazz/blues sampling slots per original jazz/blues file.
JAZZ_BLUES_REPEATS = 4  # Final jazz/blues sampling weight after the curriculum ramp.
JAZZ_RAMP_START_STEP = 10000  # Keep jazz/blues weight at one through this optimizer step.
JAZZ_RAMP_END_STEP = 15000  # Linearly increase the weight to JAZZ_BLUES_REPEATS by this step.
MULTI_INSTRUMENT_PROB = .75  # Preference for multitrack files in eligible sampling paths.
STEM_DRUM_PROB = .85  # Probability of selecting drums among eligible missing stems.
KIT_DRUM_EXAMPLE_PROB = .50  # Request authentic kit-only windows in half of drum-target training draws; other draws retain the full corpus.
JAZZ_INSTRUMENT_WEIGHT = 3.  # Sampling weight for jazz-associated instrument choices.
AR_MIXTURE_PROB = .80  # Probability of drawing from the indexed AR mixture pool.
AR_JSONL_FILES = [DATA_ROOT.parent / 'data_old' / name for name in  # Original AR datasets scanned for multitrack examples.
                  ('gigamidi_data_tokenized.jsonl', 'misc_data_tokenized.jsonl')]  # Original AR tokenized metadata filenames.
AR_MIXTURE_LIMIT = 100000  # Maximum number of indexed AR mixture records.
AR_MIXTURE_SCAN_PER_FILE = 200000  # Maximum records inspected in each original AR dataset.
AR_MIXTURE_INDEX = DATA_ROOT / 'prepared' / 'ar_mixtures_v3.json'  # Cached index of eligible AR mixtures.

# Output paths and initialization
SAVE_DIR = '/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/melody_refinement_v14'  # Fresh V14 output directory; Earlier checkpoints remain available for comparison.
RUN_NAME = 'melody_refinement_v14'  # Name identifying this finetuning run.
INIT_CHECKPOINT = '/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/ar_context_split_instrument/best_model.pt'  # Pretrained autoregressive weights used for a fresh run.
RESUME_TRAINING = True  # Resume the latest checkpoint if present; otherwise start fresh.
GENRES = ['classical', 'jazz', 'blues', 'unknown']  # Genre labels supported by the conditioning vocabulary.
VOCAB_SIZE = MelodyTokenizer().vocab_size  # Vocabulary size supplied by the music tokenizer.
MODEL_KWARGS = dict(  # Pretrained AR architecture and reference-prefix dimensions.
    vocab_size=VOCAB_SIZE,  # Number of music token types.
    embed_dim=2048,  # Transformer hidden width.
    n_heads=16,  # Number of attention query heads.
    n_kv_heads=4,  # Number of shared key/value heads.
    n_layers=20,  # Number of autoregressive transformer blocks.
    num_genres=len(GENRES),  # Number of supported genre conditions.
    num_instruments=41,  # Number of supported instrument conditions.
    dropout=.1,  # Dropout probability during training.
    style_dim=256,  # Frozen style-feature width.
    beat_dim=256,  # Frozen rhythm-feature width.
    memory_tokens=64,  # Two independent 32-token banks, ordered by reference-window time.
    memory_dim=256,  # Reference resampler width before projection to the decoder.
    memory_heads=4,  # Heads in the reference resampler; decoder blocks stay unchanged.
)  # End of model architecture settings.

# Window sizes and timing masks
SEQ_LEN = 8192  # Maximum combined prompt and response length in tokens.
CHUNK_TOKENS = 512  # Maximum source tokens per task window.
REFINE_MIN_MASK_RATIO = .90  # Minimum fraction of corrupted melody attack groups with onset/duration masked.
REFINE_FULL_MASK_PROB = .25  # Probability of masking every corrupted-melody attack group during refinement.
HARMONIZE_MIN_MASK_RATIO = .90  # Minimum fraction of melody timing attributes masked for harmonization.
HARMONIZE_FULL_MASK_PROB = .25  # Probability of masking all melody timing attributes.

# Prompt history and robustness
HISTORY_TOKENS = 2048  # Maximum preceding clean-target history tokens.
HISTORY_PROB = .75  # Probability of providing monophonic history for refinement.
POLY_HISTORY_PROB = .75  # Probability of providing polyphonic target history for other tasks.
HISTORY_CORRUPTION_PROB = .30  # Probability of corrupting supplied non-refinement history during training.
HISTORY_EDIT_FRACTION = .10  # Fraction of history notes eligible for robustness edits.
HISTORY_ADDITION_WEIGHT = 2.  # Relative weight of adding history notes.
HISTORY_DELETION_WEIGHT = 1.  # Relative weight of deleting history notes.

# Frozen style and beat references
REFERENCE_PROB = 1.0  # Condition both encoders on safe neighboring windows for every task; resample when no safe pair exists.
REFERENCE_EXACT_PROB = .50  # Harmonize/stem references: half match all target instruments, half their jazz-trio subset; refinement always matches exactly.
STYLE_DIM = 256  # Frozen style embedding dimension.
BEAT_DIM = 256  # Frozen beat embedding dimension.
REFERENCE_WINDOWS = 2  # Consecutive nearby windows on one side of the target; inference selects a random pair.
REFERENCE_PITCH_SEMITONES = 7  # One pitch shift per encoder shared across both windows; drums remain unchanged.
REFERENCE_LOCAL_BINS = 8  # Time bins of local frozen features retained within each reference window.
REFERENCE_ENCODER_BATCH_SIZE = 32  # Bound frozen-encoder activation memory independently of the AR batch.
REFERENCE_TOKENS = 512  # Maximum reference-window tokens passed to each encoder.
REFERENCE_GUARD_MS = 250  # Minimum separation between target and reference windows to prevent overlap.
REFERENCE_MAX_DISTANCE_MS = 30000  # Maximum neighbor distance used to select a reference window.
STYLE_ENCODER = '/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/style/best_checkpoint.pt'  # Trained style checkpoint; encoder weights stay frozen during finetuning.
BEAT_ENCODER = '/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/beat/best_checkpoint.pt'  # Trained beat checkpoint required before finetuning; weights stay frozen.

# Melody extraction and input corruption
SKYLINE_TOLERANCE_MS = 50  # Onset grouping tolerance for extracting performance-MIDI melody.
MAX_CORRUPTIONS = 1  # Maximum additional edit after merging detected ornaments.
GENERIC_MERGE_PROB = .10  # Probability of an ordinary merge; other edits include contour, pitch and timing.
PITCH_CORRUPTION_SEMITONES = 7  # Maximum absolute refinement/history pitch error in semitones.
TIMING_CORRUPTION_MS = 50  # Maximum absolute onset/duration error, quantized to 10 ms.
STEM_SOURCE_CORRUPTION_PROB = 0.0  # Probability of corrupting the retained source window; zero keeps pitches, onsets and durations clean.
STEM_TIMING_JITTER_MS = TIMING_CORRUPTION_MS  # Stem-input timing error limit, shared with the other tasks.
STEM_PITCH_SHIFT_PROB = .05  # Upper bound on the per-example fraction of accompaniment pitches corrupted.
STEM_PITCH_SHIFT_MIN_PROB = 0.  # Lower bound on the sampled accompaniment pitch-corruption fraction.
STEM_PITCH_CLEAN_PROB = .20  # Probability of leaving all stem-input pitches unchanged.
STEM_PITCH_SHIFT_SEMITONES = PITCH_CORRUPTION_SEMITONES  # Maximum absolute accompaniment pitch error; skyline melody is protected.

# Data loading and held-out splits
WORKER_RECORD_CACHE = 4  # Maximum cached parsed records per worker.
WORKER_CACHE_NOTES = 100000  # Per-worker cached-note budget; larger files remain usable without caching.
PREFETCH_FACTOR = 1  # Number of prefetched batches per data-loader worker.
DATA_SPLIT_SEED = 42  # Seed used to assign pieces to dataset splits.
DATA_SPLIT_RATIOS = {'train': .98, 'validation': .01, 'test': .01}  # Piece-level training, validation and test fractions.

# Loss and batch settings
PAD_ID = 2  # Tokenizer padding token identifier.
IGNORE_INDEX = -100  # Loss label used to exclude prompt and padding positions.
BATCH_SIZE = 24  # Training examples per GPU; kept at the existing run setting.
ACCUM_STEPS = 1  # Batches accumulated before each optimizer update.
NUM_WORKERS = 4  # Data-loader workers per GPU process.

# Optimizer and full training schedule
CONDITIONING_LR = 1e-4  # Peak LR for new resampler, prefix projection and other conditioning parameters.
LR = 2e-5  # Peak learning rate after warmup.
MIN_LR = 2e-6  # Learning-rate floor after the main cosine decay.
WEIGHT_DECAY = .1  # AdamW parameter regularization.
BETAS = (.9, .95)  # AdamW first- and second-moment decay coefficients.
WARMUP_STEPS = 500  # Initial updates used to increase the learning rate.
DECAY_STEPS = 5000  # Number of final updates used to cosine-decay to MIN_LR.
N_STEPS = 20000  # Total optimizer updates from fresh initialization.
GRAD_CLIP = 1.  # Maximum gradient norm.

# Reporting and checkpointing
LOG_EVERY = 1  # Optimizer updates between training metric reports.
DATA_LOG_EVERY = 50  # Training steps between text-only data diagnostics and drum-palette loss reports.
VAL_EVERY = 500  # Optimizer updates between validation evaluations.
CHECKPOINT_EVERY = 500  # Optimizer updates between checkpoint saves.
