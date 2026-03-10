import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DIR    = os.path.join(BASE_DIR, "data", "test")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
MODEL_PATH  = os.path.join(MODEL_DIR, "mobilenetv2_model.pth")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logger():
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"evaluation_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("evaluate")

def main():
    logger = setup_logger()
    logger.info(f"Using device: {DEVICE}")

    # ─── Load Class Indices ───────────────────────────────────────────────────
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    num_classes = len(class_indices)
    index_to_class = {v: k for k, v in class_indices.items()}
    class_labels = [index_to_class[i] for i in range(num_classes)]

    # ─── Load Model ───────────────────────────────────────────────────────────
    logger.info(f"Loading model from: {MODEL_PATH}")
    model = models.mobilenet_v2()
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # ─── Test Data DataLoader ─────────────────────────────────────────────────
    test_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ─── Evaluation ───────────────────────────────────────────────────────────
    logger.info("--- Running Evaluation on Test Set ---")
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_loss = running_loss / len(test_dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    logger.info(f"Test Loss:     {total_loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")

    # ─── Advanced Metrics: F1-Score + Confusion Matrix ────────────────────────
    logger.info("--- Generating Deep Metrics (F1, Confusion Matrix) ---")
    
    logger.info("\nClassification Report:\n" + classification_report(all_labels, all_preds, target_names=class_labels))

    # ─── Confusion Matrix Plot ────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title("Confusion Matrix — MobileNetV2 (PyTorch)")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    logger.info(f"Confusion Matrix saved → {cm_path}")

if __name__ == '__main__':
    main()
