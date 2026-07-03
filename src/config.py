import torch
import numpy as np
import random
from pathlib import Path

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths (modify before running)
DATA_PATH  = "path/to/your/dataset.csv"
OUTPUT_DIR = Path("path/to/your/output_directory")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Infusion duration
TINF = 1.0  # hours

# PK warm-start initial values
PK_INIT = torch.tensor([
    0.00762,  # CL (L/h)
    4.27,     # V1 (L)
    0.0171,   # Q  (L/h)
    5.44,     # V2 (L)
], dtype=torch.float32)

# Hyperparameters
CONFIG = {
    "input_dim":      7,
    "hidden_dims":    [256, 256, 128, 64],
    "pk_hidden":      [64, 32],
    "max_epochs":     500,
    "batch_size":     32,
    "lr":             1e-3,
    "weight_decay":   1e-5,
    "lambda_data":    1.0,
    "lambda_physics": 0.05,
    "lambda_param":   0.25,
    "lr_factor":      0.5,
    "lr_patience":    5,
    "lr_min":         1e-6,
    "es_patience":    50,
    "es_min_delta":   1e-6,
    "n_folds":        5,
    "test_ratio":     0.2,
    "print_every":    10,
}
