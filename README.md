# Breast Cancer Ruleout AI

A deep learning project that uses a **Convolutional Neural Network (CNN)** to classify mammogram images as **Benign** or **Malignant**, achieving **98.36% test accuracy**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [How It Works (Step by Step)](#how-it-works-step-by-step)
3. [Project Structure](#project-structure)
4. [Setup & Installation](#setup--installation)
5. [Dataset](#dataset)
6. [Running the Pipeline](#running-the-pipeline)
7. [Model Architecture](#model-architecture)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Results](#results)
10. [Key Concepts Explained](#key-concepts-explained)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

**Goal:** Given a mammogram X-ray image, predict whether the tissue is **Benign (non-cancerous)** or **Malignant (cancerous)**.

**Approach:**
- Use the MIAS (Mammographic Image Analysis Society) dataset of 322 mammogram images.
- Apply **data augmentation** (rotation) to increase the dataset to ~20,700 samples.
- Train a 6-layer CNN with **hyperparameter tuning** to find optimal settings.
- Evaluate with accuracy, precision, recall, F1-score, and confusion matrix.

**Final Results:**
| Metric | Score |
|--------|-------|
| Accuracy | 98.36% |
| Precision | 98.39% |
| Recall | 98.36% |
| F1 Score | 98.36% |
| Cohen Kappa | 96.69% |

---

## How It Works (Step by Step)

### Step 1: Data Preprocessing (`data_preprocessing/preprocess.py`)

This is where raw images become training-ready data:

1. **Read images** — Load all 322 `.pgm` mammogram files in grayscale (no colour needed for X-rays).
2. **Resize** — Scale every image to 64x64 pixels so they're all the same size (neural networks require uniform input dimensions).
3. **Data augmentation (rotation)** — The dataset is too small (only 322 images) to train a deep network. We rotate each image at 180 different angles (0, 2, 4, ..., 358 degrees), creating 180 copies per image. This gives us ~20,700 total samples.
4. **Read labels** — Parse the `Info.txt` file to find which images are Benign (B → label 0) and which are Malignant (M → label 1). Normal images are skipped.
5. **Normalise** — Divide pixel values by 255 so they fall in the range [0, 1]. This helps the optimizer converge faster.
6. **Split** — Divide into 85% training and 15% testing sets with a fixed random seed for reproducibility.
7. **Reshape** — Add a channel dimension so shape becomes (samples, 64, 64, 1) — the "1" means one colour channel (grayscale).

### Step 2: Model Building (`model_training/model.py`)

The CNN architecture has 6 convolutional blocks followed by a dense output layer:

```
Input (64x64x1)
    |
Conv2D(32) -> ReLU -> MaxPool2D                    # Block 1
    |
Conv2D(48) -> ReLU -> MaxPool2D -> Dropout(0.1)    # Block 2
    |
Conv2D(128) -> ReLU -> MaxPool2D -> Dropout(0.2)   # Block 3
    |
Conv2D(128) -> ReLU -> MaxPool2D -> Dropout(0.2)   # Block 4
    |
Conv2D(64) -> ReLU -> MaxPool2D -> Dropout(0.2)    # Block 5
    |
Conv2D(32) -> ReLU -> MaxPool2D -> Dropout(0.2)    # Block 6
    |
Flatten -> Dense(1, sigmoid)                        # Output
```

**What each layer does:**
- **Conv2D** — Slides small filters across the image to detect features (edges, textures, shapes). Early layers find simple patterns; deeper layers combine them into complex features.
- **ReLU** — An activation function that outputs the input directly if positive, otherwise outputs zero. It introduces non-linearity so the network can learn complex patterns.
- **MaxPool2D** — Shrinks the image by keeping only the maximum value in each 2x2 patch. This reduces computation and makes feature detection more robust to small shifts.
- **Dropout** — Randomly deactivates a percentage of neurons during training. This prevents overfitting by forcing the network to not rely on any single neuron.
- **Flatten** — Converts the 2D feature maps into a 1D vector for the final classification layer.
- **Dense(1, sigmoid)** — A single neuron with sigmoid activation that outputs a probability between 0 (Benign) and 1 (Malignant).

### Step 3: Training (`model_training/train.py`)

Training is the process of adjusting the model's internal weights so it makes better predictions:

1. **Forward pass** — Feed a batch of 128 images through the network, get predictions.
2. **Compute loss** — Compare predictions to true labels using Binary Cross-Entropy loss (measures how wrong the model is).
3. **Backpropagation** — Calculate how to adjust each weight to reduce the loss.
4. **Update weights** — Apply the adjustments using the Adam optimizer (an adaptive learning rate algorithm).
5. **Repeat** — Process all batches to complete one epoch. Repeat for up to 100 epochs.

**Callbacks used:**
- **EarlyStopping** — If validation accuracy hasn't improved for 10 epochs, stop training and restore the best weights. This prevents wasting time and overfitting.
- **ModelCheckpoint** — Save the model every time it achieves a new best validation accuracy, so you never lose your best result.

### Step 4: Evaluation (`evaluation/evaluate.py`)

After training, the model is tested on the 15% of data it has never seen:

1. **Predict** — Run the test images through the trained model.
2. **Threshold** — If the output probability > 0.5, classify as Malignant; otherwise, Benign.
3. **Compute metrics** — Calculate accuracy, precision, recall, F1, and Cohen's Kappa.
4. **Confusion matrix** — A 2x2 grid showing True Positives, True Negatives, False Positives, and False Negatives.
5. **Training plots** — Graphs of accuracy and loss over epochs (training vs. validation) to visualise learning progress.

---

## Project Structure

```
BREAST_CANCER_RULEOUT_AI/
|
|-- config.py                          # All constants and settings in one place
|-- main.py                            # Entry point — runs the full pipeline
|-- requirements.txt                   # Python package dependencies
|-- .gitignore                         # Files excluded from version control
|-- README.md                          # This documentation
|
|-- data/
|   |-- all-mias/                      # Raw .pgm images + Info.txt (from dataset)
|   |-- processed/                     # Saved .npy arrays (generated after first run)
|
|-- data_preprocessing/
|   |-- __init__.py
|   |-- preprocess.py                  # Image loading, augmentation, splitting
|
|-- model_training/
|   |-- __init__.py
|   |-- model.py                       # CNN architecture & hyperparameter tuning
|   |-- train.py                       # Training loop, callbacks, model saving
|   |-- saved_models/                  # Trained .h5 model files (generated)
|
|-- evaluation/
|   |-- __init__.py
|   |-- evaluate.py                    # Metrics, classification report, plots
|   |-- results/                       # Saved plots (generated)
```

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

```bash
# 1. Clone or download this project
cd BREAST_CANCER_RULEOUT_AI

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset

This project uses the **MIAS (Mammographic Image Analysis Society)** dataset:
- **322 mammogram images** in `.pgm` format (grayscale, 1024x1024 originally)
- **Labels** provided in `Info.txt`:
  - `B` = Benign (non-cancerous)
  - `M` = Malignant (cancerous)
  - `NORM` = Normal (no abnormality — excluded from our binary classification)

### Setting up the data

1. Download or extract the MIAS dataset.
2. Place all `.pgm` files and `Info.txt` into `data/all-mias/`:

```
data/
  all-mias/
    mdb001.pgm
    mdb002.pgm
    ...
    mdb322.pgm
    Info.txt
```

If your data is in a different location, use the `--data-dir` flag:
```bash
python main.py --data-dir "C:\path\to\your\mias\folder"
```

---

## Running the Pipeline

### Full pipeline (recommended for first run)

```bash
python main.py
```

This runs all 5 steps: preprocess → build model → train → evaluate → save.

### With hyperparameter tuning (slower, finds optimal settings)

```bash
python main.py --tune
```

### Skip preprocessing (reuse saved data from a previous run)

```bash
python main.py --skip-preprocess
```

### Custom data directory

```bash
python main.py --data-dir "C:\Users\you\Downloads\all-mias"
```

### Run individual modules

```bash
# Only preprocess data
python data_preprocessing/preprocess.py
```

---

## Model Architecture

The CNN (Convolutional Neural Network) processes images through 6 convolutional blocks:

| Layer | Type | Output Shape | Parameters | Purpose |
|-------|------|-------------|------------|---------|
| 1 | Conv2D(32, 3x3) | 64x64x32 | 320 | Detect basic edges/textures |
| 2 | MaxPool2D | 32x32x32 | 0 | Reduce spatial size |
| 3 | Conv2D(48, 3x3) | 32x32x48 | 13,872 | Detect more complex patterns |
| 4 | MaxPool2D + Dropout | 16x16x48 | 0 | Reduce size + prevent overfitting |
| 5-10 | 4 more Conv blocks | ... | ... | Detect increasingly abstract features |
| 11 | Flatten | 32 | 0 | Reshape for classification |
| 12 | Dense(1, sigmoid) | 1 | 33 | Final binary prediction |

**Total parameters:** ~200K (relatively lightweight model)

---

## Hyperparameter Tuning

Keras Tuner was used with **RandomSearch** to explore these parameters:

| Parameter | Search Range | Best Value Found |
|-----------|-------------|-----------------|
| Conv1 Filters | 32 - 128 (step 32) | 32 |
| Conv1 Kernel Size | 3 or 5 | 3 |
| Conv2 Filters | 32 - 128 (step 16) | 48 |
| Dropout Rate | 0.0 - 0.5 (step 0.1) | 0.1 |
| Learning Rate | 0.01, 0.001, 0.0001 | 0.0001 |

- **5 trials** with **3 executions per trial** (results averaged)
- **Objective:** maximise validation accuracy

---

## Results

### Training Progress
- Model was trained for 76 epochs (early stopping triggered at epoch 76, best at epoch 66)
- Final validation accuracy: **98.44%**

### Test Set Performance

```
Accuracy:           0.9836
Precision:          0.9839
Recall:             0.9836
F1 Score:           0.9836
Cohen Kappa Score:  0.9669
```

### Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Benign (B) | 1.00 | 0.97 | 0.98 | 1719 |
| Malignant (M) | 0.97 | 1.00 | 0.98 | 1386 |

**Key observation:** The model achieves 100% recall for Malignant cases — it catches every cancer case in the test set. This is critical for a medical screening tool where missing a cancer is far worse than a false alarm.

---

## Key Concepts Explained

### What is a CNN?
A Convolutional Neural Network is a type of deep learning model designed specifically for images. Instead of looking at every pixel independently, it slides small "filters" across the image to detect patterns like edges, textures, and shapes.

### What is Data Augmentation?
A technique to artificially increase the size of a small dataset by creating modified versions of existing data. In this project, we rotate each image at 180 different angles, turning 322 images into ~20,700 samples.

### What is Overfitting?
When a model memorises the training data instead of learning general patterns. It performs well on training data but poorly on new, unseen data. We combat this with Dropout layers and Early Stopping.

### What is Binary Cross-Entropy?
The loss function used for binary (two-class) classification. It measures how far the model's predicted probability is from the true label (0 or 1). Lower loss = better predictions.

### What is the Adam Optimiser?
An algorithm that adjusts the model's weights during training. "Adam" stands for Adaptive Moment Estimation. It adapts the learning rate for each weight individually, making training faster and more stable than basic gradient descent.

### What is Early Stopping?
A regularisation technique that monitors a metric (validation accuracy) during training. If the metric stops improving for a set number of epochs (patience=10), training halts automatically and the best weights are restored.

---

## Troubleshooting

### "Could not read image" warnings
Ensure the `.pgm` files are in the correct directory (`data/all-mias/`) and are not corrupted.

### Out of memory errors
Reduce `BATCH_SIZE` in `config.py` (try 64 or 32). You can also reduce `NUM_ROTATION_ANGLES` to use fewer augmented copies.

### Low accuracy
- Make sure images are being read correctly (check for `None` returns from `cv2.imread`).
- Verify the `Info.txt` file is present and correctly formatted.
- Try increasing `EPOCHS` or adjusting the learning rate in `config.py`.

### TensorFlow GPU issues
If you see CUDA errors, ensure your TensorFlow version matches your installed CUDA/cuDNN versions. Alternatively, force CPU mode:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```
