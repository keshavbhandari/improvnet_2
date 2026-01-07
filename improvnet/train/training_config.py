import os
import glob

# Configuration for the data
data_folder = os.path.join(os.path.dirname(__file__), "..", "data")
data_folder = os.path.abspath(data_folder)

# Get all JSONL files
DATA_DIRS = glob.glob(os.path.join(data_folder, '*.jsonl'))
datasets_to_include = ["misc_data.jsonl", "gigamidi_data.jsonl"]
DATA_DIRS = [d for d in DATA_DIRS if any(ds in os.path.basename(d) for ds in datasets_to_include)]

TRAIN_TYPE = "pretraining"                       # 'pretraining' or 'finetuning'
LOGS_PATH = f"improvnet/artifacts/{TRAIN_TYPE}/logs.txt"
TENSORBOARD_LOG_DIR = f"improvnet/artifacts/{TRAIN_TYPE}/tensorboard"
CHECKPOINTS_DIR = f"improvnet/artifacts/{TRAIN_TYPE}/checkpoint"

# Model Architecture Parameters
MAX_LEN = 2048 
EMBED_DIM = 1152                               
NUM_DECODER_LAYERS = 14                               
NUM_ENCODER_LAYERS = 6                                
HEADS = 18                                      
MLP_MULT = 4                                   
NO_BIAS = False                               
USE_CHECKPOINTING = True                       

# Training Hyperparameters
BATCH_SIZE = 15                                 
NUM_WORKERS = 4                                 
LR = 2e-4                                       
NUM_EPOCHS = 10                                 
GRADIENT_ACCUMULATION_STEPS = 5                 
GRADIENT_CHECKPOINTING = True                   
LOAD_FROM_CHECKPOINT = True                     
USE_TENSORBOARD = True                          
LOG_STEP = 5                                    
DEBUG = False                                   

# --- Learning Rate Schedule Settings ---
WARMUP_RATIO = 0.03            # Ratio of total steps for warmup
MAX_WARMUP_STEPS = 5000         # Hard cap for warmup steps to prevent excessive warmup on large datasets

if TRAIN_TYPE == "finetuning":
    LOAD_FROM_CHECKPOINT = True
    LR = 5e-5
