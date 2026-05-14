"""
config.py
---------
Central configuration file for the Breast Cancer Ruleout AI project.
All tunable constants (paths, image sizes, training parameters) live here
so you only need to change values in ONE place.
"""

import os

# ──────────────────────────────────────────────
# 1. PATHS
# ──────────────────────────────────────────────
# Root of the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Folder that contains the raw .pgm mammogram images and Info.txt
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "all-mias")

# Where we save preprocessed numpy arrays so we don't re-read images every run
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Where trained model weights are saved
MODEL_DIR = os.path.join(PROJECT_ROOT, "model_training", "saved_models")

# Where evaluation plots (accuracy/loss curves, confusion matrix) are saved
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation", "results")

# ──────────────────────────────────────────────
# 2. IMAGE PREPROCESSING
# ──────────────────────────────────────────────
IMAGE_SIZE = (64, 64)           # Width x Height each image is resized to
VISUALIZE_SIZE = (224, 224)     # Larger size used only for sample visualization
NUM_ROTATION_ANGLES = 360       # Rotate from 0 to 359 degrees
ROTATION_STEP = 2               # Rotate every 2 degrees → 180 augmented copies per image

# ──────────────────────────────────────────────
# 3. DATA SPLITTING
# ──────────────────────────────────────────────
TEST_SIZE = 0.15                # 15 % of data reserved for testing
RANDOM_STATE = 2021             # Fixed seed so results are reproducible

# ──────────────────────────────────────────────
# 4. MODEL / TRAINING
# ──────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS = 100
VALIDATION_SPLIT = 0.2          # 20 % of training data used for validation

# Hyperparameter search bounds (used by Keras Tuner)
HP_CONV1_FILTERS = (32, 128, 32)    # (min, max, step)
HP_CONV1_KERNEL = [3, 5]            # Kernel size choices
HP_CONV2_FILTERS = (32, 128, 16)    # (min, max, step)
HP_DROPOUT = (0.0, 0.5, 0.1)       # (min, max, step)
HP_LEARNING_RATES = [1e-2, 1e-3, 1e-4]

# Keras Tuner search settings
TUNER_MAX_TRIALS = 5
TUNER_EXECUTIONS_PER_TRIAL = 3

# Early stopping
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MONITOR = "val_accuracy"
EARLY_STOP_RESTORE_BEST = True

# ──────────────────────────────────────────────
# 5. BEST HYPERPARAMETERS (from tuner results)
#    Used when you skip the tuner and train directly
# ──────────────────────────────────────────────
BEST_CONV1_FILTERS = 32
BEST_CONV1_KERNEL = 3
BEST_CONV2_FILTERS = 48
BEST_DROPOUT = 0.1
BEST_LEARNING_RATE = 1e-4
