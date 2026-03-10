import sys
import os
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Ensure we can import app from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'api')))

from app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Tomato Disease API is running" in response.json().get("message", "")

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@patch("app.predict_image")
def test_predict_disease_success(mock_predict):
    # Setup mock to return dummy prediction
    mock_predict.return_value = ("Tomato___Early_blight", "98.5", {"Tomato___Early_blight": 98.5})
    
    # Create a dummy image file
    fake_image = io.BytesIO(b"dummy image data")
    
    response = client.post(
        "/predict", 
        files={"file": ("test.jpg", fake_image, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.jpg"
    assert data["predicted_class"] == "Tomato___Early_blight"
    assert data["confidence"] == "98.5%"
    assert "probabilities" in data

def test_predict_disease_invalid_file_type():
    fake_text = io.BytesIO(b"this is text, not an image")
    
    response = client.post(
        "/predict", 
        files={"file": ("test.txt", fake_text, "text/plain")}
    )
    
    assert response.status_code == 400
    assert "Invalid file type" in response.json().get("detail", "")

def test_recommend_treatment():
    mock_data = {
        "symptoms": "Test symptoms",
        "causes": "Test causes",
        "organic": "Test organic",
        "chemical": "Test chemical"
    }
    
    # We must mock it at the actual module it comes from because app.py imports it locally.
    with patch("llm_service.get_treatment_recommendations", return_value=mock_data):
        response = client.get("/recommend/Tomato___Early_blight")
    
    assert response.status_code == 200
    assert response.json() == mock_data
