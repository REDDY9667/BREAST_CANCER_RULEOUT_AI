"""
model.py
--------
Defines the CNN (Convolutional Neural Network) architecture used to
classify mammogram images as Benign (0) or Malignant (1).

Two ways to use this module:
    1. build_model_from_best_hp()  – builds the model with the best
       hyperparameters already found (fast, no search).
    2. run_hyperparameter_search() – uses Keras Tuner to try many
       combinations and find the best one automatically.

ARCHITECTURE OVERVIEW (6 Conv blocks → Flatten → Dense → Sigmoid):
──────────────────────────────────────────────────────────────────
    Input (64×64×1 grayscale image)
        │
        ▼
    Conv2D  →  ReLU  →  MaxPool2D              ← Block 1 (tunable filters & kernel)
        │
        ▼
    Conv2D  →  ReLU  →  MaxPool2D  → Dropout   ← Block 2 (tunable filters & dropout)
        │
        ▼
    Conv2D(128) → ReLU → MaxPool2D → Dropout   ← Block 3
        │
        ▼
    Conv2D(128) → ReLU → MaxPool2D → Dropout   ← Block 4
        │
        ▼
    Conv2D(64)  → ReLU → MaxPool2D → Dropout   ← Block 5
        │
        ▼
    Conv2D(32)  → ReLU → MaxPool2D → Dropout   ← Block 6
        │
        ▼
    Flatten  →  Dense(1, sigmoid)               ← Output (0 or 1)
"""

import os
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    HP_CONV1_FILTERS, HP_CONV1_KERNEL,
    HP_CONV2_FILTERS, HP_DROPOUT, HP_LEARNING_RATES,
    TUNER_MAX_TRIALS, TUNER_EXECUTIONS_PER_TRIAL,
    BEST_CONV1_FILTERS, BEST_CONV1_KERNEL,
    BEST_CONV2_FILTERS, BEST_DROPOUT, BEST_LEARNING_RATE,
    IMAGE_SIZE,
)


# ──────────────────────────────────────────────────────────────
# OPTION A : Build with known-best hyperparameters (no search)
# ──────────────────────────────────────────────────────────────
def build_model_from_best_hp():
    """
    Construct and compile the CNN using the hyperparameters that were
    already determined to be optimal by a previous tuner run.

    Returns
    -------
    tensorflow.keras.Model  –  compiled model ready for .fit()
    """
    input_shape = (*IMAGE_SIZE, 1)  # (64, 64, 1)

    model = Sequential([
        # --- Block 1 ---
        Conv2D(BEST_CONV1_FILTERS, (BEST_CONV1_KERNEL, BEST_CONV1_KERNEL),
               padding="same", strides=(1, 1), input_shape=input_shape),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"),

        # --- Block 2 ---
        Conv2D(BEST_CONV2_FILTERS, (3, 3), padding="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"),
        Dropout(BEST_DROPOUT),

        # --- Block 3 ---
        Conv2D(128, (3, 3), padding="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"),
        Dropout(0.2),

        # --- Block 4 ---
        Conv2D(128, (3, 3), padding="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"),
        Dropout(0.2),

        # --- Block 5 ---
        Conv2D(64, (3, 3), padding="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"),
        Dropout(0.2),

        # --- Block 6 ---
        Conv2D(32, (3, 3), padding="same"),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"),
        Dropout(0.2),

        # --- Output ---
        Flatten(),
        Dense(1),
        Activation("sigmoid"),   # Sigmoid squashes output to [0, 1]
    ])

    model.compile(
        optimizer=Adam(learning_rate=BEST_LEARNING_RATE),
        loss=BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ──────────────────────────────────────────────────────────────
# OPTION B : Hyperparameter search with Keras Tuner
# ──────────────────────────────────────────────────────────────
def _build_model_for_tuner(hp):
    """
    Model-building function consumed by Keras Tuner.
    'hp' is a HyperParameters object that lets us declare search ranges.

    WHY HYPERPARAMETER TUNING?
    --------------------------
    Different filter counts, kernel sizes, dropout rates, and learning
    rates all affect how well the model learns.  Instead of guessing,
    Keras Tuner systematically tries many combinations and picks the
    best one based on validation accuracy.
    """
    input_shape = (*IMAGE_SIZE, 1)

    model = Sequential()

    # Block 1 — tunable filters and kernel size
    model.add(Conv2D(
        filters=hp.Int("conv_1_filter",
                        min_value=HP_CONV1_FILTERS[0],
                        max_value=HP_CONV1_FILTERS[1],
                        step=HP_CONV1_FILTERS[2]),
        kernel_size=hp.Choice("conv_1_kernel", values=HP_CONV1_KERNEL),
        padding="same", strides=(1, 1), input_shape=input_shape,
    ))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"))

    # Block 2 — tunable filters and dropout
    model.add(Conv2D(
        filters=hp.Int("conv_2_filter",
                        min_value=HP_CONV2_FILTERS[0],
                        max_value=HP_CONV2_FILTERS[1],
                        step=HP_CONV2_FILTERS[2]),
        kernel_size=(3, 3), padding="same",
    ))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"))
    model.add(Dropout(
        rate=hp.Float("dropout_2",
                       min_value=HP_DROPOUT[0],
                       max_value=HP_DROPOUT[1],
                       step=HP_DROPOUT[2]),
    ))

    # Blocks 3-6 — fixed architecture
    for filters in [128, 128, 64, 32]:
        model.add(Conv2D(filters, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="same"))
        model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(
        optimizer=Adam(
            learning_rate=hp.Choice("learning_rate", HP_LEARNING_RATES)
        ),
        loss=BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def run_hyperparameter_search(x_train, y_train):
    """
    Use Keras Tuner's RandomSearch to try different hyperparameter
    combinations and return the best model + its hyperparameters.

    Parameters
    ----------
    x_train : np.ndarray – training images
    y_train : np.ndarray – training labels

    Returns
    -------
    best_model            – compiled Keras model with best weights
    best_hyperparameters  – dict-like object with winning values
    """
    from keras_tuner import RandomSearch

    tuner = RandomSearch(
        _build_model_for_tuner,
        objective="val_accuracy",
        max_trials=TUNER_MAX_TRIALS,
        executions_per_trial=TUNER_EXECUTIONS_PER_TRIAL,
        directory="tuner_results",
        project_name="breast_cancer_hp_search",
    )

    print("[model] Starting hyperparameter search …")
    tuner.search(x_train, y_train, epochs=5, validation_split=0.2)

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters()[0]

    print(f"""
    ╔══════════════════════════════════════════╗
    ║   Hyperparameter Search Complete         ║
    ╠══════════════════════════════════════════╣
    ║  Conv1 filters : {best_hp.get('conv_1_filter'):<23} ║
    ║  Conv1 kernel  : {best_hp.get('conv_1_kernel'):<23} ║
    ║  Conv2 filters : {best_hp.get('conv_2_filter'):<23} ║
    ║  Dropout       : {best_hp.get('dropout_2'):<23} ║
    ║  Learning rate : {best_hp.get('learning_rate'):<23} ║
    ╚══════════════════════════════════════════╝
    """)

    return best_model, best_hp
