VOCAB_SIZES    = [91, 137, 22, 509, 510] # instrument, pitch, velocity, onset, dur
NUM_ATTRS      = len(VOCAB_SIZES)
SEQ_LEN        = 1024  # Max sequence length
EMBED_DIM      = 1000  # Transformer width (1000 / 5 attrs = 200) & (1000 / 20 heads = 50 head_dim)
LATENT_DIM     = 8    # FSQ dimensions (Wide bandwidth)
LEVELS         = 3     # levels per FSQ dim → total compression = 3^8 = 6561x
NUM_QUANTIZERS = 2     # How many FSQ layers in the residual stack
N_HEADS        = 20    # query heads
N_KV_HEADS     = 5     # key/value heads for GQA (4 Q heads per 1 KV head)
IS_CAUSAL      = False # Causal (autoregressive) or non-causal (bidirectional) attention
N_LAYERS       = 12    # transformer layers in encoder and decoder each
FFN_MULT       = 8/3   # SwiGLU hidden mult
ATTN_DROPOUT   = 0.05
FFN_DROPOUT    = 0.15
BATCH_SIZE     = 64     # per-step micro-batch (Adjust based on VRAM)
ACCUM_STEPS    = 1     # gradient accumulation
N_STEPS        = 100_000
WARMUP_STEPS   = 1000
LOG_EVERY      = 10
GRAD_CLIP      = 1.0
USE_CKPT       = True  # Highly recommended at 12 layers/1000 dim
VAL_EVERY      = 5000   # Run validation every N steps
SAVE_DIR       = f"/gpfs/scratch/acw769/improvnet/artifacts/rfsq"
VAL_LOG_FILE   = f"val_metrics.txt"
RUN_NAME       = f"rfsq"
RESUME_TRAINING = False   # Set to True to load from checkpoint

DEFAULT_MRA_BASE_VALUES = [100.0, 150.0, 30.0, 1031.0, 1031.0]
JSONL_FILES = [
    # "/data/scratch/acw769/improvnet/artifacts/data/gigamidi_data.jsonl",
    # "/data/scratch/acw769/improvnet/artifacts/data/misc_data.jsonl"
    "/data/scratch/acw769/improvnet/artifacts/data/gigamidi_data_tokenized.jsonl",
    "/data/scratch/acw769/improvnet/artifacts/data/misc_data_tokenized.jsonl"
]