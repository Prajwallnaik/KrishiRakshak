import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io

import sys
import os

# Add project root to path so we can import config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
    
from config.settings import settings

MODEL_PATH = str(settings.MODEL_PATH)
CLASS_INDICES_PATH = str(settings.CLASS_INDICES_PATH)

IMG_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables to hold model in memory
_model = None
_index_to_class = None
_transform = None

def get_model():
    """Returns the loaded model, loading it if necessary."""
    global _model, _index_to_class, _transform
    
    if _model is None:
        print(f"Loading PyTorch Model onto {DEVICE}...")
        
        # Load class indices
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)
        _index_to_class = {v: k for k, v in class_indices.items()}
        num_classes = len(class_indices)
        
        # Load Architecture
        _model = models.mobilenet_v2()
        _model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(_model.last_channel, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Load Weights
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        _model = _model.to(DEVICE)
        _model.eval()
        
        # Define standard transformation
        _transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Model loaded successfully.")
        
    return _model, _index_to_class, _transform

def predict_image(image_bytes: bytes):
    """
    Takes raw image bytes from an upload, preprocesses it, and runs inference.
    Returns: (predicted_class, confidence_percentage, all_probabilities)
    """
    model, index_to_class, transform = get_model()
    
    # 1. Read bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 2. Transform into tensor batch
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # 3. Model Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    confidence_val, pred_idx = torch.max(probabilities, 0)
    
    pred_idx = pred_idx.item()
    confidence = round(confidence_val.item() * 100, 2)
    pred_class = index_to_class[pred_idx]
    
    all_probs = {index_to_class[i]: round(prob.item() * 100, 2) 
                 for i, prob in enumerate(probabilities)}
    
    return pred_class, confidence, all_probs
