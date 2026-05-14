"""
evaluate.py
-----------
Evaluates the trained model on the held-out test set and generates:
    1. Numeric metrics  (accuracy, precision, recall, F1, Cohen's Kappa)
    2. Classification report (per-class breakdown)
    3. Confusion matrix heatmap (saved as image)
    4. Training history plots — accuracy & loss curves (saved as image)

KEY METRICS EXPLAINED (for beginners):
──────────────────────────────────────
Accuracy   – % of all predictions that were correct.
Precision  – Of everything the model called "Malignant", how many truly were?
             High precision = few false alarms.
Recall     – Of all actual malignant cases, how many did the model catch?
             High recall = few missed cancers (critical in medical AI).
F1 Score   – Harmonic mean of precision and recall; balances both.
Cohen's Kappa – Measures agreement beyond random chance (1.0 = perfect).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works on servers without a display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EVAL_DIR


# ──────────────────────────────────────────────────────────────
# 1. COMPUTE METRICS
# ──────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    """
    Calculate and print all evaluation metrics.

    Parameters
    ----------
    y_true – ground-truth labels (0 or 1)
    y_pred – model's predicted labels (0 or 1)

    Returns
    -------
    dict with metric names → values
    """
    metrics = {
        "Accuracy":          np.round(accuracy_score(y_true, y_pred), 4),
        "Precision":         np.round(precision_score(y_true, y_pred, average="weighted"), 4),
        "Recall":            np.round(recall_score(y_true, y_pred, average="weighted"), 4),
        "F1 Score":          np.round(f1_score(y_true, y_pred, average="weighted"), 4),
        "Cohen Kappa Score": np.round(cohen_kappa_score(y_true, y_pred), 4),
    }

    print("\n" + "=" * 45)
    print("         EVALUATION METRICS")
    print("=" * 45)
    for name, value in metrics.items():
        print(f"  {name:<22}: {value}")
    print("=" * 45)

    target_names = ["Benign (B)", "Malignant (M)"]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\n  Classification Report:\n")
    print(report)

    return metrics


# ──────────────────────────────────────────────────────────────
# 2. CONFUSION MATRIX
# ──────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save=True):
    """
    Draw a heatmap of the confusion matrix.

    The confusion matrix shows:
        True Negatives  | False Positives
        ─────────────────────────────────
        False Negatives | True Positives

    In our case:
        - True Negative  = correctly predicted Benign
        - True Positive  = correctly predicted Malignant
        - False Positive = predicted Malignant but was actually Benign
        - False Negative = predicted Benign but was actually Malignant
                           (the most dangerous error in cancer screening!)
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if save:
        os.makedirs(EVAL_DIR, exist_ok=True)
        path = os.path.join(EVAL_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150)
        print(f"[evaluate] Confusion matrix saved to {path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 3. TRAINING HISTORY PLOTS
# ──────────────────────────────────────────────────────────────
def plot_training_history(history, save=True):
    """
    Plot accuracy and loss curves for training vs. validation.

    WHY THIS MATTERS:
    -----------------
    • If training accuracy keeps rising but validation accuracy plateaus
      or drops → the model is OVERFITTING (memorising training data
      instead of learning general patterns).
    • If both curves rise together and level off → good generalisation.
    • The gap between training loss and validation loss shows how much
      the model struggles with unseen data.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Training History", fontsize=14)

    # Accuracy subplot
    ax1.plot(epochs_range, acc, label="Training Accuracy", linewidth=2)
    ax1.plot(epochs_range, val_acc, label="Validation Accuracy", linewidth=2)
    ax1.set_title("Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss subplot
    ax2.plot(epochs_range, loss, label="Training Loss", linewidth=2)
    ax2.plot(epochs_range, val_loss, label="Validation Loss", linewidth=2)
    ax2.set_title("Loss over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        os.makedirs(EVAL_DIR, exist_ok=True)
        path = os.path.join(EVAL_DIR, "training_history.png")
        plt.savefig(path, dpi=150)
        print(f"[evaluate] Training history plot saved to {path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# 4. FULL EVALUATION PIPELINE
# ──────────────────────────────────────────────────────────────
def run_evaluation(model, x_test, y_test, history=None):
    """
    End-to-end evaluation:
        1. Compute test loss & accuracy via model.evaluate()
        2. Generate predictions
        3. Print all metrics
        4. Save confusion matrix plot
        5. Save training history plot (if history is provided)

    Parameters
    ----------
    model   – trained Keras model
    x_test  – test images
    y_test  – test labels
    history – Keras History object from model.fit() (optional)

    Returns
    -------
    dict of metrics
    """
    print("\n[evaluate] Evaluating model on test set …")

    # model.evaluate returns [loss, accuracy]
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_accuracy:.4f}")

    # Generate binary predictions
    y_pred_proba = model.predict(x_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Metrics
    metrics = compute_metrics(y_test, y_pred)

    # Plots
    plot_confusion_matrix(y_test, y_pred)

    if history is not None:
        plot_training_history(history)

    return metrics
