"""
preprocess.py
-------------
Handles everything related to loading raw mammogram images, reading their
labels, applying data augmentation (rotation), and splitting into
train / test sets.

Pipeline at a glance:
    raw .pgm images  -->  read & resize  -->  rotate (augment)  -->  pair with labels
    -->  shuffle & split into train / test  -->  save as .npy files
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Import our central configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, PROCESSED_DIR,
    IMAGE_SIZE, NUM_ROTATION_ANGLES, ROTATION_STEP,
    TEST_SIZE, RANDOM_STATE,
)


# ──────────────────────────────────────────────────────────────
# STEP 1 : READ RAW IMAGES
# ──────────────────────────────────────────────────────────────
def read_images(data_dir: str) -> dict:
    """
    Read all 322 mammogram .pgm images from *data_dir*, resize each to
    IMAGE_SIZE, then create rotated copies for data augmentation.

    WHY ROTATE?
    -----------
    The MIAS dataset has only 322 images — far too few to train a deep
    neural network.  By rotating each image at 180 different angles
    (0, 2, 4, …, 358 degrees) we multiply the dataset size by 180,
    giving the model more varied examples to learn from.

    Returns
    -------
    dict  –  {image_name: {angle: numpy_array, ...}, ...}
    """
    print("[preprocess] Reading images from:", data_dir)
    images = {}

    for i in range(322):
        # Build the image filename: mdb001, mdb002, …, mdb322
        if i < 9:
            image_name = f"mdb00{i + 1}"
        elif i < 99:
            image_name = f"mdb0{i + 1}"
        else:
            image_name = f"mdb{i + 1}"

        image_path = os.path.join(data_dir, image_name + ".pgm")

        # Read in GRAYSCALE (0 means grayscale in OpenCV)
        # Mammograms are inherently grayscale — no colour information needed.
        img = cv2.imread(image_path, 0)
        if img is None:
            print(f"  WARNING: could not read {image_path}, skipping.")
            continue

        # Resize to a small, uniform size so every image is the same shape.
        img = cv2.resize(img, IMAGE_SIZE)
        rows, cols = img.shape

        # Create rotated copies
        images[image_name] = {}
        for angle in range(0, NUM_ROTATION_ANGLES, ROTATION_STEP):
            # getRotationMatrix2D returns a 2×3 transformation matrix
            rotation_matrix = cv2.getRotationMatrix2D(
                (cols / 2, rows / 2),   # centre of rotation
                angle,                   # degrees
                1,                       # scale (1 = keep original size)
            )
            rotated = cv2.warpAffine(img, rotation_matrix, (cols, rows))
            images[image_name][angle] = rotated

    print(f"  Loaded {len(images)} images, each with "
          f"{NUM_ROTATION_ANGLES // ROTATION_STEP} augmented rotations.")
    return images


# ──────────────────────────────────────────────────────────────
# STEP 2 : READ LABELS
# ──────────────────────────────────────────────────────────────
def read_labels(data_dir: str) -> dict:
    """
    Parse the Info.txt file that ships with the MIAS dataset.

    Each line looks like:
        mdb001 G CIRC B 535 425 197
    The 4th column tells us the abnormality class:
        B = Benign   → label 0
        M = Malignant → label 1
    Images marked NORM (normal) are skipped because they have no
    abnormality label relevant to our binary classification.

    Returns
    -------
    dict  –  {image_name: {angle: label, ...}, ...}
    """
    print("[preprocess] Reading labels …")
    label_path = os.path.join(data_dir, "Info.txt")
    with open(label_path, "r") as f:
        text = f.read()

    labels = {}
    for line in text.split("\n"):
        words = line.split()
        if len(words) > 3:
            abnormality = words[3]
            if abnormality == "B":
                labels[words[0]] = {
                    angle: 0
                    for angle in range(0, NUM_ROTATION_ANGLES, ROTATION_STEP)
                }
            elif abnormality == "M":
                labels[words[0]] = {
                    angle: 1
                    for angle in range(0, NUM_ROTATION_ANGLES, ROTATION_STEP)
                }

    # The file contains a header row keyed "Truth-Data:" — remove it.
    labels.pop("Truth-Data:", None)

    benign_count = sum(1 for v in labels.values() if list(v.values())[0] == 0)
    malignant_count = len(labels) - benign_count
    print(f"  Found {len(labels)} labelled images "
          f"({benign_count} benign, {malignant_count} malignant).")
    return labels


# ──────────────────────────────────────────────────────────────
# STEP 3 : PAIR IMAGES WITH LABELS AND SPLIT
# ──────────────────────────────────────────────────────────────
def prepare_datasets(data_dir: str = None):
    """
    Orchestrates the full preprocessing pipeline:
        1. Read labels   → know which images are B / M
        2. Read images   → load pixels + augment
        3. Pair them     → create X (pixels) and Y (labels) arrays
        4. Split         → 85 % train, 15 % test
        5. Reshape       → add a channel dimension (needed by Conv2D)

    Returns
    -------
    x_train, x_test, y_train, y_test  –  numpy arrays ready for the model
    """
    if data_dir is None:
        data_dir = DATA_DIR

    labels = read_labels(data_dir)
    images = read_images(data_dir)

    X, Y = [], []
    for image_name in labels:
        if image_name not in images:
            continue  # skip if image was unreadable
        for angle in range(0, NUM_ROTATION_ANGLES, ROTATION_STEP):
            X.append(images[image_name][angle])
            Y.append(labels[image_name][angle])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # Normalise pixel values from 0-255 → 0-1 (helps the model converge faster)
    X = X / 255.0

    print(f"[preprocess] Total samples after augmentation: {len(X)}")

    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=TEST_SIZE,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Conv2D expects shape (samples, height, width, channels).
    # Our images are grayscale so channels = 1.
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    print(f"  x_train shape: {x_train.shape}")
    print(f"  x_test  shape: {x_test.shape}")

    return x_train, x_test, y_train, y_test


# ──────────────────────────────────────────────────────────────
# STEP 4 (optional) : SAVE / LOAD preprocessed data
# ──────────────────────────────────────────────────────────────
def save_processed_data(x_train, x_test, y_train, y_test):
    """Save numpy arrays so future runs skip the slow image-reading step."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(PROCESSED_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)
    print(f"[preprocess] Saved processed data to {PROCESSED_DIR}")


def load_processed_data():
    """Load previously saved numpy arrays."""
    x_train = np.load(os.path.join(PROCESSED_DIR, "x_train.npy"))
    x_test = np.load(os.path.join(PROCESSED_DIR, "x_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    print(f"[preprocess] Loaded processed data from {PROCESSED_DIR}")
    return x_train, x_test, y_train, y_test


def processed_data_exists() -> bool:
    """Check whether saved .npy files already exist."""
    return all(
        os.path.exists(os.path.join(PROCESSED_DIR, f))
        for f in ["x_train.npy", "x_test.npy", "y_train.npy", "y_test.npy"]
    )


# ──────────────────────────────────────────────────────────────
# CLI entry point — run this file directly to preprocess data
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = prepare_datasets()
    save_processed_data(x_train, x_test, y_train, y_test)
    print("Done!")
