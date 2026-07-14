"""
predict.py — Prediction Script
--------------------------------
This script takes a traffic sign image and tells you what sign it is.

HOW TO USE FROM COMMAND LINE:
    python predict.py path/to/your/image.jpg

HOW IT WORKS:
    1. Load the saved model (trained by train.py)
    2. Load the class labels (from labels.json)
    3. Read and preprocess the image exactly the same way as during training
    4. Pass it through the model → get probabilities for each class
    5. Pick the class with the highest probability

WHY must preprocessing match training?
    The model was trained on 64x64, normalized images.
    If we feed it a different size or un-normalized image, predictions will be wrong.
"""

import os
import sys
import json
import numpy as np
import cv2

# We import Keras just for loading the model
from tensorflow.keras.models import load_model

# ─── File Paths ───────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("model", "traffic_sign_model.h5")
LABELS_PATH = os.path.join("model", "labels.json")

# Image size must match what was used in training
IMG_SIZE = 64
# ──────────────────────────────────────────────────────────────────────────────


def load_model_and_labels():
    """
    Loads the trained model and class label mapping from disk.

    RETURNS:
        model   — The trained Keras model
        labels  — A dict like {"0": "Stop", "1": "No Entry", ...}

    RAISES:
        FileNotFoundError if model or labels file is missing
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'.\n"
            "Please run 'python train.py' first to train and save the model."
        )

    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"Labels file not found at '{LABELS_PATH}'.\n"
            "Please run 'python train.py' first to generate the labels file."
        )

    # Load the trained Keras model
    model = load_model(MODEL_PATH)

    # Load the class labels from JSON
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    return model, labels


def preprocess_image(image_path):
    """
    Reads and preprocesses a single image for prediction.

    This MUST match the preprocessing done in train.py:
    - Read with OpenCV
    - Resize to 64x64
    - Convert BGR → RGB
    - Normalize pixel values to 0-1 range
    - Add a batch dimension (model expects a batch, not a single image)

    RETURNS:
        Preprocessed image ready for model.predict(), shape: (1, 64, 64, 3)

    RAISES:
        FileNotFoundError if the image file doesn't exist
        ValueError if OpenCV can't read the image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: '{image_path}'")

    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(
            f"Could not read image at '{image_path}'.\n"
            "Make sure it is a valid image file (jpg, jpeg, png)."
        )

    # Resize to 64x64 (same as training)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert BGR to RGB (same as training)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize: pixel values 0-255 → 0.0-1.0 (same as training)
    img = img.astype("float32") / 255.0

    # Add batch dimension: (64, 64, 3) → (1, 64, 64, 3)
    # The model expects a batch of images, even if it's just one
    img = np.expand_dims(img, axis=0)

    return img


def predict_image(image_path, model=None, labels=None):
    """
    Predicts the class of a traffic sign image.

    This function is designed to be:
    - Called from the command line (via __main__ below)
    - Imported and called from app.py (to avoid duplicating code)

    ARGUMENTS:
        image_path — Path to the image file
        model      — Optional: pre-loaded Keras model (avoids reloading every call)
        labels     — Optional: pre-loaded labels dict (avoids reloading every call)

    RETURNS:
        predicted_class — String name of the predicted traffic sign
        confidence      — Float between 0 and 1 (e.g., 0.97 means 97% confident)
    """
    # If model/labels weren't passed in, load them now
    if model is None or labels is None:
        model, labels = load_model_and_labels()

    # Preprocess the image
    processed_img = preprocess_image(image_path)

    # Run the image through the model
    # predictions is an array of probabilities for each class
    # e.g., [0.02, 0.01, 0.91, 0.04, 0.02] means class 2 is most likely
    predictions = model.predict(processed_img, verbose=0)

    # Get the index with the highest probability
    predicted_index = np.argmax(predictions[0])

    # Get the confidence (probability) for that prediction
    confidence = float(predictions[0][predicted_index])

    # Look up the class name using the index
    predicted_class = labels[str(predicted_index)]

    return predicted_class, confidence


# ── Command Line Interface ────────────────────────────────────────────────────
# This block runs only when you call the script directly:
#     python predict.py path/to/image.jpg
# It does NOT run when predict.py is imported by app.py
if __name__ == "__main__":
    # Check that the user provided an image path
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        print("Example: python predict.py dataset/Stop/Stop_001.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Predicting class for: {image_path}")
    print("-" * 40)

    try:
        predicted_class, confidence = predict_image(image_path)
        print(f"Predicted Sign : {predicted_class}")
        print(f"Confidence     : {confidence * 100:.2f}%")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Something went wrong: {e}")
        sys.exit(1)
