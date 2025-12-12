import os

# --- Data ---
DATA_DIR = "CSV_train"
NUM_CLIENTS = 9
FEATURE_COLS_TO_DROP = ["label"]  # Adjust if needed

# --- Training ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5
NUM_ROUNDS = 10

# --- Device ---
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")