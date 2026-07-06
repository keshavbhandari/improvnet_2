VOCAB_SIZES    = [91, 137, 22, 509, 510] # instrument, pitch, velocity, onset, dur
NUM_ATTRS      = len(VOCAB_SIZES)
SEQ_LEN        = 1024  # Max sequence length
EMBED_DIM      = 1000  # Transformer width (1000 / 5 attrs = 200) & (1000 / 20 heads = 50 head_dim)
PATCH_SIZE     = 8     # How many tokens to compress into a single continuous vector
LATENT_DIM     = 1024   # Continuous dimensions (bounded -1 to 1 via tanh)
N_HEADS        = 20    # query heads
N_KV_HEADS     = 5     # key/value heads for GQA (4 Q heads per 1 KV head)
IS_CAUSAL      = False # Causal (autoregressive) or non-causal (bidirectional) attention
N_LAYERS       = 12    # transformer layers in encoder and decoder each
FFN_MULT       = 8/3   # SwiGLU hidden mult
ATTN_DROPOUT   = 0.05
FFN_DROPOUT    = 0.15
TARGET_BETA    = 0.05   # KL weight for VAE loss (can be annealed during training)
BATCH_SIZE     = 64     # per-step micro-batch (Adjust based on VRAM)
ACCUM_STEPS    = 1     # gradient accumulation
N_STEPS        = 200_000
WARMUP_STEPS   = 1000
LOG_EVERY      = 10
GRAD_CLIP      = 1.0
USE_CKPT       = True  # Highly recommended at 12 layers/1000 dim
VAL_EVERY      = 25000   # Run validation every N steps
SAVE_DIR       = f"/gpfs/scratch/acw769/improvnet/artifacts/autoencoder_{PATCH_SIZE}patch_{LATENT_DIM}latent"
VAL_LOG_FILE   = f"val_metrics_autoencoder_{PATCH_SIZE}patch_{LATENT_DIM}latent.txt"
RUN_NAME       = f"autoencoder_{PATCH_SIZE}patch_{LATENT_DIM}latent"
RESUME_TRAINING = False   # Set to True to load from checkpoint

DEFAULT_MRA_BASE_VALUES = [100.0, 150.0, 30.0, 1031.0, 1031.0]
JSONL_FILES = [
    "/data/scratch/acw769/improvnet/artifacts/data/gigamidi_data_tokenized.jsonl",
    "/data/scratch/acw769/improvnet/artifacts/data/misc_data_tokenized.jsonl"
]







# VOCAB_SIZES    = [91, 137, 22, 509, 510] # instrument, pitch, velocity, onset, dur
# NUM_ATTRS      = len(VOCAB_SIZES)
# SEQ_LEN        = 1024  # Max sequence length
# EMBED_DIM      = 1000  # Transformer width (1000 / 5 attrs = 200) & (1000 / 20 heads = 50 head_dim)
# PATCH_SIZE     = 8     # How many tokens to compress into a single continuous vector
# LATENT_DIM     = 128   # Continuous dimensions (bounded -1 to 1 via tanh)
# N_HEADS        = 20    # query heads
# N_KV_HEADS     = 5     # key/value heads for GQA (4 Q heads per 1 KV head)
# IS_CAUSAL      = False # Causal (autoregressive) or non-causal (bidirectional) attention
# N_LAYERS       = 12    # transformer layers in encoder and decoder each
# FFN_MULT       = 8/3   # SwiGLU hidden mult
# ATTN_DROPOUT   = 0.05
# FFN_DROPOUT    = 0.15
# BATCH_SIZE     = 64     # per-step micro-batch (Adjust based on VRAM)
# ACCUM_STEPS    = 1     # gradient accumulation
# N_STEPS        = 100_000
# WARMUP_STEPS   = 1000
# LOG_EVERY      = 10
# GRAD_CLIP      = 1.0
# USE_CKPT       = True  # Highly recommended at 12 layers/1000 dim
# VAL_EVERY      = 5000   # Run validation every N steps
# SAVE_DIR       = f"/gpfs/scratch/acw769/improvnet/artifacts/autoencoder"
# VAL_LOG_FILE   = f"val_metrics_autoencoder.txt"
# RUN_NAME       = f"autoencoder"
# RESUME_TRAINING = False   # Set to True to load from checkpoint

# DEFAULT_MRA_BASE_VALUES = [100.0, 150.0, 30.0, 1031.0, 1031.0]
# JSONL_FILES = [
#     # "/data/scratch/acw769/improvnet/artifacts/data/gigamidi_data.jsonl",
#     # "/data/scratch/acw769/improvnet/artifacts/data/misc_data.jsonl"
#     "/data/scratch/acw769/improvnet/artifacts/data/gigamidi_data_tokenized.jsonl",
#     "/data/scratch/acw769/improvnet/artifacts/data/misc_data_tokenized.jsonl"
# ]