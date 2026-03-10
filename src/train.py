import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import copy
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import copy
import mlflow
import mlflow.pytorch
import logging
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR  = os.path.join(BASE_DIR, "data", "train")
VAL_DIR    = os.path.join(BASE_DIR, "data", "val")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")

# ─── Config & Hyperparameters ─────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 20
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logger():
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("train")

def main():
    logger = setup_logger()
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Using device: {DEVICE}")

    # Set MLflow experiment
    mlflow.set_experiment("Tomato_Disease_Classification")
    
    with mlflow.start_run() as run:
        logger.info(f"Active MLflow Run ID: {run.info.run_id}")
        
        # Log hyperparameters
        mlflow.log_params({
            "image_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "device": str(DEVICE),
            "model_architecture": "MobileNetV2"
        })

        # ─── Data Transformations ─────────────────────────────────────────────────
    train_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=20, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ─── Datasets & DataLoaders ───────────────────────────────────────────────
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

    # Windows num_workers often causes issues if not 0, but using it under __main__ is safe. 
    # To be totally safe, set num_workers=0 or 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Save Class Indices → class_indices.json
    class_indices = train_dataset.class_to_idx
    class_indices_path = os.path.join(MODEL_DIR, "class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(class_indices, f)
    logger.info(f"Class indices saved → {class_indices_path}")

    # ─── Architecture Selection (MobileNetV2) ─────────────────────────────────
    num_classes = len(class_indices)

    logger.info("Loading pre-trained MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all pre-trained base layers
    for param in model.parameters():
        param.requires_grad = False

    # Custom Classification Head
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    model = model.to(DEVICE)

    # ─── Criterion, Optimizer, Scheduler ──────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-6, verbose=True
    )

    # ─── Training Loop ────────────────────────────────────────────────────────
    logger.info("Beginning Training...")
    best_val_accuracy = 0.0
    patience_counter = 0
    early_stopping_patience = 5

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch+1}/{EPOCHS}")
        logger.info("-" * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            logger.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Log metrics to MLflow per phase
            mlflow.log_metrics({
                f"{phase}_loss": epoch_loss,
                f"{phase}_accuracy": epoch_acc.item()
            }, step=epoch)
            
            if phase == 'val':
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()
                
                if epoch_acc > best_val_accuracy:
                    best_val_accuracy = epoch_acc
                    patience_counter = 0
                    
                    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "mobilenetv2_model.pth"))
                    logger.info("Saved new best model.")
                else:
                    patience_counter += 1
                    
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    logger.info(f"Best val Acc: {best_val_accuracy:.4f}")
    logger.info("Training complete! Best model saved to models/mobilenetv2_model.pth")
    
    # After training loop completes, log the architecture as an MLflow artifact
    # Reload best model weights logically and log via PyTorch MLflow flavor
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mobilenetv2_model.pth")))
    mlflow.pytorch.log_model(model, "mobilenetv2_model")
    mlflow.log_artifact(class_indices_path)

if __name__ == '__main__':
    main()
