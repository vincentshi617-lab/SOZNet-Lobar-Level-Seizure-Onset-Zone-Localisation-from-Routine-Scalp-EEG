"""Central configuration for SOZNet.

Paths are resolved from environment variables so that no absolute paths
are committed to the repository. Example:

    export SOZNET_DATA_ROOT=/path/to/eeg_epochs_output
    export SOZNET_ANNOTATION_CSV=/path/to/Final_List_GT.csv
    export SOZNET_OUTPUT_DIR=/path/to/outputs
"""
import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("SOZNET_DATA_ROOT", "data/processed")
ANNOTATION_CSV = os.environ.get("SOZNET_ANNOTATION_CSV", "data/labels.csv")
OUTPUT_DIR = os.environ.get("SOZNET_OUTPUT_DIR", "outputs")

# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------
NUM_ELECTRODES = 18
CHUNK_SIZE = 2048            # 8 s at 256 Hz
RAW_WINDOW_SIZE = 4096       # 16 s at 256 Hz, before centre / random crop
NUM_CLASSES = 6
NUM_REGIONS = 6

# Six anatomical regions used throughout the package:
# 0 = Left Frontal (LF)
# 1 = Right Frontal (RF)
# 2 = Left Temporo-Parietal (LTP)
# 3 = Right Temporo-Parietal (RTP)
# 4 = Left Occipital (LO)
# 5 = Right Occipital (RO)
REGION_NAMES = ["LF", "RF", "LTP", "RTP", "LO", "RO"]
LEFT_REGIONS = [0, 2, 4]
RIGHT_REGIONS = [1, 3, 5]

# Mapping from the "Blocks" column in the clinical CSV to internal indices.
# 1 -> LF, 2 -> RF, 3 -> LTP, 4 -> RTP, 5 -> LO, 6 -> RO
BLOCK_TO_INDEX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
SAMPLES_PER_SESSION = 500
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 20
TEST_RATIO = 0.1
VAL_RATIO = 0.1
SEED = 42

WARMUP_EPOCHS = 5
COSINE_ETA_MIN = 5e-5

# ---------------------------------------------------------------------------
# Loss / post-processing
# ---------------------------------------------------------------------------
FOCAL_GAMMA = 2.0
POS_WEIGHT_FACTOR = 3.0
HEMI_PENALTY_LAMBDA = 0.1
WINDOW_PROB_THRESHOLD = 0.3
SESSION_PROB_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
F1 = 8
F2 = 16
D = 2
KERNEL_1 = 64
KERNEL_2 = 16
DROPOUT = 0.5

# ---------------------------------------------------------------------------
# Fine-tuning / model soup
# ---------------------------------------------------------------------------
FINETUNE_LR = 1e-5
FINETUNE_EPOCHS = 2
FINETUNE_WARMUP_STEPS = 50
FINETUNE_M = 3

# ---------------------------------------------------------------------------
# Augmentation defaults
# ---------------------------------------------------------------------------
AUG_NUM_AUGMENTED = 40000
AUG_SEIZURE_TARGET = 1000
AUG_NOISE_STD = 0.1
AUG_SCALE_RANGE = (0.7, 1.3)
USE_SWAP_AUG = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
