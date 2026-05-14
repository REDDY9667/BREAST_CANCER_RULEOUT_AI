"""
main.py
-------
The single entry point that runs the ENTIRE pipeline from start to finish:

    Step 1 → Preprocess data  (read images, augment, split)
    Step 2 → Build the CNN model
    Step 3 → Train the model
    Step 4 → Evaluate on the test set
    Step 5 → Save the trained model

Usage:
    python main.py                  # train with known-best hyperparameters
    python main.py --tune           # run hyperparameter search first
    python main.py --skip-preprocess  # reuse previously saved .npy data

HOW THE PIECES FIT TOGETHER:
────────────────────────────
    main.py  calls →  data_preprocessing/preprocess.py  (load images)
                  →  model_training/model.py            (build CNN)
                  →  model_training/train.py             (train it)
                  →  evaluation/evaluate.py              (test & report)
"""

import argparse
import sys
import os

# Add project root to Python path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing.preprocess import (
    prepare_datasets,
    save_processed_data,
    load_processed_data,
    processed_data_exists,
)
from model_training.model import build_model_from_best_hp, run_hyperparameter_search
from model_training.train import train_model, save_model
from evaluation.evaluate import run_evaluation


def main():
    # ── Parse command-line arguments ──────────────────────────
    parser = argparse.ArgumentParser(
        description="Breast Cancer Ruleout AI — full training pipeline"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run hyperparameter search before training (slow but thorough)."
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip image loading; reuse saved .npy files from a previous run."
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to the folder containing .pgm images and Info.txt. "
             "Defaults to data/all-mias inside the project."
    )
    args = parser.parse_args()

    # ── STEP 1 : DATA PREPROCESSING ──────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 / 5 : DATA PREPROCESSING")
    print("=" * 60)

    if args.skip_preprocess and processed_data_exists():
        print("  Reusing saved preprocessed data …")
        x_train, x_test, y_train, y_test = load_processed_data()
    else:
        x_train, x_test, y_train, y_test = prepare_datasets(args.data_dir)
        save_processed_data(x_train, x_test, y_train, y_test)

    # ── STEP 2 : BUILD MODEL ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2 / 5 : BUILDING THE CNN MODEL")
    print("=" * 60)

    if args.tune:
        print("  Running hyperparameter search (this may take a while) …")
        model, best_hp = run_hyperparameter_search(x_train, y_train)
    else:
        print("  Using known-best hyperparameters …")
        model = build_model_from_best_hp()

    # ── STEP 3 : TRAIN ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 / 5 : TRAINING")
    print("=" * 60)

    history = train_model(model, x_train, y_train)

    # ── STEP 4 : EVALUATE ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 / 5 : EVALUATION")
    print("=" * 60)

    metrics = run_evaluation(model, x_test, y_test, history=history)

    # ── STEP 5 : SAVE MODEL ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5 / 5 : SAVING MODEL")
    print("=" * 60)

    save_model(model, "final_model.h5")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Final test accuracy: {metrics['Accuracy']}")
    print("  Saved artifacts:")
    print("    - model_training/saved_models/best_model.h5")
    print("    - model_training/saved_models/final_model.h5")
    print("    - evaluation/results/confusion_matrix.png")
    print("    - evaluation/results/training_history.png")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
