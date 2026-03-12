import os
import json
import requests

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
    
from config.settings import settings

FALLBACK_RECOMMENDATIONS = {
    "bacterial spot": {
        "symptoms": "Water-soaked spots on leaves that turn brown or black. Fruit may develop raised, scabby spots.",
        "causes": "Bacterium Xanthomonas campestris, favored by high humidity and warm weather.",
        "organic": "Prune infected leaves and stems. Use copper sprays and ensure good air circulation.",
        "chemical": "Apply streptomycin or copper-based bactericides early."
    },
    "early blight": {
        "symptoms": "Dark, concentric rings (bullseye shape) on older leaves. Leaves may yellow and drop.",
        "causes": "Fungus Alternaria solani, thrives in warm temperatures with high humidity.",
        "organic": "Remove infected lower leaves. Use organic copper fungicides and space plants well.",
        "chemical": "Use fungicides with chlorothalonil, mancozeb, or copper."
    },
    "late blight": {
        "symptoms": "Large, dark brown or grey blotches on leaves and stems. Rapid rotting of fruit.",
        "causes": "Oomycete Phytophthora infestans, requires cool, wet weather.",
        "organic": "Destroy infected plants immediately. Avoid overhead watering.",
        "chemical": "Apply protectant fungicides like chlorothalonil or mancozeb before symptoms appear."
    },
    "leaf mold": {
        "symptoms": "Pale green or yellow spots on the upper leaf surface with velvety olive-green mold underneath.",
        "causes": "Fungus Passalora fulva, common in high humidity and poor ventilation.",
        "organic": "Improve ventilation to reduce humidity. Avoid wetting foliage.",
        "chemical": "Use fungicides containing chlorothalonil or mancozeb as preventative sprays."
    },
    "target spot": {
        "symptoms": "Small brown spots forming target-like concentric rings. Fruit may have dark sunken lesions.",
        "causes": "Fungus Corynespora cassiicola, favored by high humidity and moderate temperatures.",
        "organic": "Ensure good air circulation. Remove severely infected lower leaves.",
        "chemical": "Apply fungicides such as chlorothalonil or azoxystrobin."
    },
    "healthy": {
        "symptoms": "No visible symptoms.",
        "causes": "Plant is thriving.",
        "organic": "Maintain current care routines (proper watering, sunlight).",
        "chemical": "None required."
    }
}

def get_fallback_recommendation(disease_name: str) -> dict:
    disease_lower = disease_name.lower().strip()
    return FALLBACK_RECOMMENDATIONS.get(disease_lower, {
        "symptoms": "Unable to fetch live symptoms at this time.",
        "causes": "Recommendation service unavailable.",
        "organic": "Please consult a local agricultural extension.",
        "chemical": "Ensure proper diagnosis before applying chemicals."
    })

def get_treatment_recommendations(disease_name: str) -> dict:
    """
    Requests a structured JSON response from OpenAI containing 
    treatment advice for a specific tomato plant disease via REST API.
    """
    if not disease_name or disease_name.lower() == "healthy":
        return get_fallback_recommendation("healthy")

    prompt = f"""
    You are an expert agricultural botanist specializing in tomato plant diseases.
    Provide treatment advice for the tomato plant condition: "{disease_name}".
    
    You MUST return the output strictly as a valid JSON object with the following exactly four keys (do not use Markdown code blocks or any other formatting around the JSON):
    {{
        "symptoms": "A brief 2-sentence description of the visual symptoms.",
        "causes": "A brief 1-sentence description of what causes this.",
        "organic": "A brief 2-sentence description of organic/natural treatments.",
        "chemical": "A brief 2-sentence description of chemical/synthetic treatments if necessary."
    }}
    """

    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {settings.OPENAI_API_KEY}'
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(
            url, 
            headers=headers, 
            json=payload
        )
        
        if response.status_code == 429:
            print("OpenAI API Rate Limit Exceeded (429). Using fallback dictionary.")
            return get_fallback_recommendation(disease_name)
            
        response.raise_for_status()
        
        data = response.json()
        text = data['choices'][0]['message']['content'].strip()
        
        # Clean up in case the LLM wrapped it in markdown json block
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        text = text.strip()
        
        result_data = json.loads(text)
        
        # Ensure all keys exist
        return {
            "symptoms": result_data.get("symptoms", "Data unavailable."),
            "causes": result_data.get("causes", "Data unavailable."),
            "organic": result_data.get("organic", "Data unavailable."),
            "chemical": result_data.get("chemical", "Data unavailable.")
        }
    except Exception as e:
        print(f"Error fetching LLM recommendations: {str(e)}. Using fallback dictionary.")
        # Fallback if the API fails or JSON parsing fails
        return get_fallback_recommendation(disease_name)
