import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Training config
RANDOM_SEED = 42
TEST_DAYS = 30  # last 30 days of data for test set
TOP_K_VALUES = [5, 10, 20]

# Model hyperparameters
HYBRID_ALPHA = 0.5   # collaborative weight
HYBRID_BETA = 0.3    # content-based weight
HYBRID_GAMMA = 0.2   # popularity weight

# Interaction weights
INTERACTION_WEIGHTS = {"view": 1, "cart": 3, "purchase": 5}

# Catalog
NUM_USERS = 500
NUM_ITEMS = 200
NUM_CATEGORIES = 10
