"""
train.py
--------
Handles the actual training loop: fitting the model on training data,
using callbacks to prevent overfitting, and saving the best weights.

KEY CONCEPTS FOR BEGINNERS:
───────────────────────────
Epoch      – One full pass through the entire training dataset.
Batch size – Number of images the model processes before updating its
             internal weights.  Larger batches = faster but need more RAM.
Callback   – A function that Keras calls automatically at certain points
             during training (e.g. after every epoch).  We use:
                • EarlyStopping  → stop training if the model stops improving
                • ModelCheckpoint → save the model whenever it hits a new best
"""

import os
import sys

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, MODEL_DIR,
    EARLY_STOP_PATIENCE, EARLY_STOP_MONITOR, EARLY_STOP_RESTORE_BEST,
)


def get_callbacks():
    """
    Create and return the list of Keras callbacks used during training.

    EarlyStopping
    -------------
    Monitors validation accuracy after every epoch.  If it hasn't improved
    for *patience* consecutive epochs, training stops automatically.
    `restore_best_weights=True` means the model rolls back to the epoch
    that had the highest validation accuracy.

    ModelCheckpoint
    ---------------
    Saves the model file (best_model.h5) every time a new best validation
    accuracy is achieved, so you never lose your best result even if
    training crashes later.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_DIR, "best_model.h5")

    early_stop = EarlyStopping(
        monitor=EARLY_STOP_MONITOR,
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=EARLY_STOP_RESTORE_BEST,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        filepath=best_model_path,
        monitor=EARLY_STOP_MONITOR,
        save_best_only=True,
        verbose=1,
    )

    return [early_stop, checkpoint]


def train_model(model, x_train, y_train):
    """
    Train the given model on x_train / y_train.

    What happens inside .fit():
        1. Keras divides x_train into mini-batches of size BATCH_SIZE.
        2. For each batch it:
           a. Feeds the images through the network (forward pass).
           b. Computes the loss (how wrong the predictions are).
           c. Computes gradients (which direction to adjust weights).
           d. Updates the weights (backpropagation).
        3. After processing all batches, one epoch is complete.
        4. Keras evaluates the model on the validation set (20% held out).
        5. Callbacks check if we should stop or save.
        6. Repeat for up to EPOCHS times (or until early stopping fires).

    Parameters
    ----------
    model   – compiled Keras model (from model.py)
    x_train – numpy array of training images
    y_train – numpy array of training labels

    Returns
    -------
    history – a History object containing loss & accuracy per epoch
              (used later for plotting)
    """
    callbacks = get_callbacks()

    print(f"[train] Starting training for up to {EPOCHS} epochs …")
    print(f"        Batch size       : {BATCH_SIZE}")
    print(f"        Validation split : {VALIDATION_SPLIT * 100:.0f}%")
    print(f"        Early stop after : {EARLY_STOP_PATIENCE} epochs without improvement")

    history = model.fit(
        x_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
    )

    print("[train] Training complete.")
    return history


def save_model(model, filename="final_model.h5"):
    """Save the full model (architecture + weights) to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    model.save(path)
    print(f"[train] Model saved to {path}")


def load_saved_model(filename="best_model.h5"):
    """Load a previously saved model from disk."""
    from tensorflow.keras.models import load_model
    path = os.path.join(MODEL_DIR, filename)
    model = load_model(path)
    print(f"[train] Model loaded from {path}")
    return model
