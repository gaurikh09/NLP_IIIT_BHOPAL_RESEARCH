"""Application configuration."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model", "saved_model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Model settings
MODEL_NAME = "google/mt5-small"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64
BEAM_NUM = 4

# Training settings
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
EVAL_STEPS = 5000
TRAIN_SAMPLES = 45000
VAL_SAMPLES = 5000
TOTAL_SAMPLES = 50000
MIN_CONTENT_LENGTH = 100

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
