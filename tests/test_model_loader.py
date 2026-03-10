import sys
import os
import io
import pytest
from PIL import Image

# Ensure we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'api')))

import model_loader

def test_get_model_loads_successfully():
    """
    Ensure the PyTorch model, transforms, and class indices load
    without raising exceptions.
    """
    model_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'models', 'mobilenetv2_model.pth')
    if not os.path.exists(model_path):
        pytest.skip(f"Weights missing at {model_path}. Skipping test.")
        
    model, index_to_class, transform = model_loader.get_model()
    
    assert model is not None
    assert isinstance(index_to_class, dict)
    assert len(index_to_class) > 0
    assert transform is not None

def test_predict_image_integration():
    """
    Optional test for actual prediction if the model file is present.
    If tomato_disease_model.pth is missing, we skip.
    """
    model_path = os.path.join(os.path.dirname(__file__), '..', 'api', 'tomato_disease_model.pth')
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found at {model_path}. Skipping actual inference test.")
    
    # Create a dummy image just to ensure inference runs without crashing
    img = Image.new('RGB', (224, 224), color='green')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    # Run prediction
    predicted_class, confidence, all_probs = model_loader.predict_image(img_bytes)
    
    # Assertions on return types and values
    assert isinstance(predicted_class, str)
    assert "Tomato___" in predicted_class
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 100.0
    assert isinstance(all_probs, dict)
    assert len(all_probs) == 10 # 10 tomato classes
