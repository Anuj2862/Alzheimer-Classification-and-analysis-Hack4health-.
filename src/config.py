import torch

# Data paths
BASE_PATH = "Datasets"  # Assumes Datasets folder is in root
TRAIN_PARQUET = f"{BASE_PATH}/train.parquet"
TEST_PARQUET = f"{BASE_PATH}/test.parquet"

# Image processing
IMG_SIZE = 160 # From notebook
NUM_CLASSES = 4 # 0=Healthy, 1=Very Mild, 2=Mild, 3=Moderate (mapped to 0-3 internally if needed)

# Training config
BATCH_SIZE = 16
EPOCHS = 35
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

# System
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
