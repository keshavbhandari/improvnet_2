# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VOCAB_SIZES    = [91, 137, 22, 509, 510] # instrument, pitch, velocity, onset, dur
NUM_ATTRS      = len(VOCAB_SIZES)
PATCH_SIZE     = 16    # Increased to capture more context per patch
N_PATCHES      = 512   # Adjust based on average sequence length
EMBED_DIM      = 1000  # Transformer width (1000 / 5 attrs = 200) & (1000 / 20 heads = 50 head_dim)
LATENT_DIM     = 10    # FSQ dimensions (Wide bandwidth)
LEVELS         = 4     # levels per FSQ dim → 4^10 = 1,048,576 codes
NUM_QUANTIZERS = 4     # How many FSQ layers in the residual stack
N_HEADS        = 20    # query heads
N_KV_HEADS     = 5     # key/value heads for GQA (4 Q heads per 1 KV head)
N_LAYERS       = 12    # transformer layers in encoder and decoder each
FFN_MULT       = 8/3   # SwiGLU hidden mult
DROPOUT        = 0.0
BATCH_SIZE     = 12     # per-step micro-batch (Adjust based on VRAM)
ACCUM_STEPS    = 2     # gradient accumulation
N_STEPS        = 500_000
WARMUP_STEPS   = 500
LOG_EVERY      = 20
GRAD_CLIP      = 1.0
USE_CKPT       = True  # Highly recommended at 12 layers/1000 dim
VAL_EVERY        = 10000   # Run validation every N steps
SAVE_DIR         = "/gpfs/scratch/acw769/improvnet/artifacts/rfsq"
VAL_LOG_FILE     = "val_metrics.txt"
RESUME_TRAINING  = False   # Set to True to load from checkpoint

DEFAULT_MRA_BASE_VALUES = [100.0, 150.0, 30.0, 1031.0, 1031.0]
JSONL_FILES = [
    # "/data/scratch/acw769/improvnet/artifacts/data/gigamidi_data.jsonl",
    # "/data/scratch/acw769/improvnet/artifacts/data/misc_data.jsonl"
    "/data/scratch/acw769/improvnet/artifacts/data/gigamidi_data_tokenized.jsonl",
    "/data/scratch/acw769/improvnet/artifacts/data/misc_data_tokenized.jsonl"
]