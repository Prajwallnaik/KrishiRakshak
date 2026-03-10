import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_loader import predict_image

# Import config via absolute path since sys.path is already updated above
from config.settings import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A PyTorch MobileNetV2 model serving tomato leaf disease predictions.",
    version=settings.VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["System"])
def root():
    """Root - points to docs."""
    return {"message": "Tomato Disease API is running. Visit /docs for Swagger UI."}

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/predict", tags=["Inference"])
async def predict_disease(file: UploadFile = File(...)):
    """
    Upload a tomato leaf image (JPEG/PNG) to get the disease prediction.
    Returns the predicted class, confidence percentage, and all class probabilities.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG and PNG images are supported."
        )

    try:
        image_bytes = await file.read()
        predicted_class, confidence, all_probs = predict_image(image_bytes)

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": f"{confidence}%",
            "probabilities": all_probs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/{disease}", tags=["LLM"])
async def recommend_treatment(disease: str):
    """
    Get live treatment recommendations for a given tomato leaf disease 
    using the Gemini API.
    """
    try:
        from llm_service import get_treatment_recommendations
        # Assuming the LLM call is synchronous, run it directly
        recommendations = get_treatment_recommendations(disease)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")
