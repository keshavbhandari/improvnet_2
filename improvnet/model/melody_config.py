"""AR fine-tuning settings. Batch size is per GPU; no beat features are used."""
from pathlib import Path
from improvnet.utils.melody import MelodyTokenizer

DATA_ROOT = Path('/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/jazz_blues_data')
DATA_FILE = DATA_ROOT / 'prepared' / 'performances.jsonl'
SAVE_DIR = '/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/melody_refinement_v1'
RUN_NAME = 'melody_refinement_v1'
INIT_CHECKPOINT = '/e/scratch/e-dev-2026d09-047/bhandari1/improvnet/artifacts/ar_context_split_instrument/best_model.pt'
RESUME_TRAINING = True  # Resume this run if present; otherwise load AR weights only.

GENRES = ['classical', 'jazz', 'blues', 'unknown']
VOCAB_SIZE = MelodyTokenizer().vocab_size
MODEL_KWARGS = dict(vocab_size=VOCAB_SIZE, embed_dim=2048, n_heads=16,
                    n_kv_heads=4, n_layers=20, num_genres=len(GENRES),
                    num_instruments=41, dropout=0.1)
SEQ_LEN = 8192  # Includes prompt + response; two conditioning positions are reserved.
PAD_ID = 2
IGNORE_INDEX = -100

WINDOW_MS = 5000
CONTEXT_WINDOWS = 1  # Left and right performance context, each this many windows.
SKYLINE_TOLERANCE_MS = 50
MAX_CORRUPTIONS = 3
TASK_WEIGHTS = (0.35, 0.30, 0.35)  # refine, rhythm, harmonize
DATA_SPLIT_SEED = 42
DATA_SPLIT_RATIOS = {'train': .90, 'validation': .05, 'test': .05}

BATCH_SIZE = 24
ACCUM_STEPS = 1
NUM_WORKERS = 16
LR = 2e-5
MIN_LR = 2e-6
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
WARMUP_STEPS = 500
DECAY_STEPS = 5000
N_STEPS = 20000  # Optimizer updates; initial full fine-tuning budget.
GRAD_CLIP = 1.0
LOG_EVERY = 1
VAL_EVERY = 500
CHECKPOINT_EVERY = 500
