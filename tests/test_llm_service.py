import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Ensure we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'api')))

import llm_service

def test_get_fallback_recommendations_known_disease():
    result = llm_service.get_treatment_recommendations("Tomato___Late_blight")
    # Verify it returns a dictionary with the required keys
    assert "symptoms" in result
    assert "causes" in result
    assert "organic" in result
    assert "chemical" in result

def test_get_fallback_recommendations_unknown_disease():
    result = llm_service.get_treatment_recommendations("NonExistent_Disease")
    assert "symptoms" in result
    assert type(result["symptoms"]) is str

@patch("llm_service.requests.post")
def test_get_treatment_recommendations_openai_success(mock_post):
    # Mock successful response from OpenAI
    mock_response = MagicMock()
    mock_response.status_code = 200
    
    # Format the fake response exactly how OpenAI returns JSON inside a markdown block
    fake_json_string = '''```json
    {
        "symptoms": "LLM Test Symptoms",
        "causes": "LLM Test Causes",
        "organic": "LLM Test Organic",
        "chemical": "LLM Test Chemical"
    }
    ```'''
    
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": fake_json_string
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    # Force the key check to pass momentarily by assuming openAI is configured
    original_key = llm_service.settings.OPENAI_API_KEY
    llm_service.settings.OPENAI_API_KEY = "dummy_key"
    
    result = llm_service.get_treatment_recommendations("Tomato___Late_blight")
    
    # Restore original key
    llm_service.settings.OPENAI_API_KEY = original_key
    
    assert result["symptoms"] == "LLM Test Symptoms"
    assert result["causes"] == "LLM Test Causes"

@patch("llm_service.requests.post")
def test_get_treatment_recommendations_api_quota_exceeded(mock_post):
    # Mock quota exceeded error (429)
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_post.return_value = mock_response
    
    # Ensure a key is set so it uses the OpenAI path
    original_key = llm_service.settings.OPENAI_API_KEY
    llm_service.settings.OPENAI_API_KEY = "dummy_key"
    
    # It should automatically fall back to hardcoded dictionary
    result = llm_service.get_treatment_recommendations("Tomato___Late_blight")
    
    # Restore key
    llm_service.settings.OPENAI_API_KEY = original_key
    # Check that it fell back by ensuring it returns a dict with symptoms
    assert "symptoms" in result
