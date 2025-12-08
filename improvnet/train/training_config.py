import os
import glob

# Configuration for the data
# DATA_DIRS = glob.glob(os.path.join("../improvnet/data", '*.jsonl'))
# Path to the data folder relative to this script
data_folder = os.path.join(os.path.dirname(__file__), "..", "data")

# Expand to absolute path
data_folder = os.path.abspath(data_folder)

# Get all JSONL files
DATA_DIRS = glob.glob(os.path.join(data_folder, '*.jsonl'))
datasets_to_include = ["misc_data.jsonl", "gigamidi_data.jsonl"]
DATA_DIRS = [d for d in DATA_DIRS if any(ds in os.path.basename(d) for ds in datasets_to_include)]

TRAIN_TYPE = "pretraining"                       # Type of training: 'pretraining' or 'finetuning'
LOGS_PATH = f"improvnet/artifacts/{TRAIN_TYPE}/logs.txt"              # Path for logs
TENSORBOARD_LOG_DIR = f"improvnet/artifacts/{TRAIN_TYPE}/tensorboard"  # Directory for TensorBoard logs
CHECKPOINTS_DIR = f"improvnet/artifacts/{TRAIN_TYPE}/checkpoint"      # Directory for saving checkpoints

# Configuration for the training
MAX_LEN = 2048 #8192                              # Maximum length of token sequences
EMBED_DIM = 1152                               # Dimension of token embeddings
NUM_DECODER_LAYERS = 14                               # Number of transformer decoder layers
NUM_ENCODER_LAYERS = 6                                # Number of transformer encoder layers
HEADS = 18                                      # Number of attention heads
MLP_MULT = 4                                   # Multiplier for the feedforward network dimension
NO_BIAS = False                               # Whether to use no bias in Linear layers
USE_CHECKPOINTING = True                       # Whether to use gradient checkpointing

BATCH_SIZE = 30 #8                                # Batch size for training
NUM_WORKERS = 4                                 # Number of workers for data loading
LR = 2e-4                                       # Learning rate for the optimizer
NUM_EPOCHS = 10                                 # Number of epochs for training
GRADIENT_ACCUMULATION_STEPS = 2                 # Number of gradient accumulation steps
GRADIENT_CHECKPOINTING = True                   # Whether to use gradient checkpointing
LOAD_FROM_CHECKPOINT = True                     # Whether to load weights from a checkpoint
USE_TENSORBOARD = True                          # Whether to use TensorBoard for logging
LOG_STEP = 5                                    # Step interval for logging training progress
DEBUG = False                                   # Whether to run in debug mode

if TRAIN_TYPE == "finetuning":
    LOAD_FROM_CHECKPOINT = True
    LR = 5e-5