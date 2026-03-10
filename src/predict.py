import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import logging
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR           = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH         = os.path.join(BASE_DIR, "models", "mobilenetv2_model.pth")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "models", "class_indices.json")
LOGS_DIR           = os.path.join(BASE_DIR, "logs")

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logger():
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"inference_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("predict")

def load_model_and_classes():
    """Loads the trained model and inverts class_indices.json to {index: class_name}"""
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)    # { "Tomato___Bacterial_spot": 0, ... }
    
    index_to_class = {v: k for k, v in class_indices.items()}
    num_classes = len(class_indices)
    
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
    
    return model, index_to_class

def preprocess_image(image_path):
    """
    Loads a single image, resizes to 224x224, normalizes
    and adds batch dimension.
    """
    img = Image.open(image_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, 224, 224)
    return img_tensor.to(DEVICE)

def predict(image_path):
    """
    Given an image path, returns the predicted disease class and confidence %.
    """
    model, index_to_class = load_model_and_classes()
    img_tensor = preprocess_image(image_path)
    
    # Run Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    confidence_val, pred_idx = torch.max(probabilities, 0)
    
    pred_idx = pred_idx.item()
    confidence = confidence_val.item() * 100
    pred_class = index_to_class[pred_idx]
    
    return {
        "predicted_class": pred_class,
        "confidence": f"{confidence:.2f}%",
        "all_probabilities": {index_to_class[i]: f"{prob.item()*100:.2f}%" for i, prob in enumerate(probabilities)}
    }

if __name__ == "__main__":
    import sys
    
    logger = setup_logger()
    
    if len(sys.argv) < 2:
        logger.error("Usage: python predict.py <path_to_image>")
        logger.error("Example: python predict.py ../data/test/Tomato___healthy/img001.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        logger.error(f"Error: Image file not found at '{image_path}'")
        sys.exit(1)
    
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Running inference on: {image_path}")
    
    result = predict(image_path)
    
    logger.info(f"\n--- Prediction Result ---")
    logger.info(f"Predicted Class : {result['predicted_class']}")
    logger.info(f"Confidence      : {result['confidence']}")
    
    logger.info(f"\nAll Class Probabilities:")
    for cls, prob in result["all_probabilities"].items():
        logger.info(f"  {cls:<35} {prob}")
