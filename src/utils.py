import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ─── Shared Paths ─────────────────────────────────────────────────────────────
BASE_DIR           = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR          = os.path.join(BASE_DIR, "models")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# ─── Label Helpers (class_indices.json only) ──────────────────────────────────
def get_class_names():
    """
    Reads class_indices.json and returns an {index: class_name} mapping.
    Example: { 0: 'Tomato___Bacterial_spot', 1: 'Tomato___Early_blight', ... }
    """
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)    # {"Tomato___Bacterial_spot": 0, ...}
    return {v: k for k, v in class_indices.items()}  # {0: "Tomato___...", ...}


def get_class_indices():
    """
    Reads class_indices.json and returns the raw {class_name: index} dict.
    """
    with open(CLASS_INDICES_PATH, "r") as f:
        return json.load(f)


def decode_prediction(pred_index):
    """
    Converts a predicted integer index to its class name string.

    Args:
        pred_index (int): Index output from model.predict / np.argmax.

    Returns:
        str: Human-readable class name.
    """
    index_to_class = get_class_names()
    return index_to_class.get(pred_index, "Unknown")


# ─── Image Preprocessing ──────────────────────────────────────────────────────
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image from disk, converts to RGB, resizes to target_size,
    normalizes pixel values to [0, 1], and adds a batch dimension.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target (height, width). Default is (224, 224).

    Returns:
        np.ndarray: Preprocessed image array of shape (1, H, W, 3).
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


# ─── EDA Plotting Helpers ─────────────────────────────────────────────────────
def plot_class_distribution(class_counts, title="Class Distribution", save_path=None):
    """
    Plots a horizontal bar chart of image counts per class.
    """
    import seaborn as sns
    plt.figure(figsize=(12, 6))
    sns.barplot(y=list(class_counts.keys()), x=list(class_counts.values()))
    plt.title(title)
    plt.xlabel("Number of Images")
    plt.ylabel("Disease Class")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Chart saved → {save_path}")
    plt.show()


def plot_sample_images(image_paths_by_class, num_classes=9, save_path=None):
    """
    Plots a 3x3 grid of sample images, one per disease class.
    """
    import random
    classes = list(image_paths_by_class.keys())[:num_classes]
    plt.figure(figsize=(15, 10))

    for i, cls in enumerate(classes):
        sample_path = random.choice(image_paths_by_class[cls])
        img = Image.open(sample_path)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(cls[:25])
        plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Sample grid saved → {save_path}")
    plt.show()


# ─── Output Formatting ────────────────────────────────────────────────────────
def format_prediction_output(pred_class, confidence):
    """
    Returns a clean, formatted string for displaying prediction results.
    """
    return (
        f"\n╔══════════════════════════════════════╗\n"
        f"  Predicted Class : {pred_class}\n"
        f"  Confidence      : {confidence:.2f}%\n"
        f"╚══════════════════════════════════════╝"
    )
