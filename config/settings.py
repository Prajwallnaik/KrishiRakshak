import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")

class Settings:
    # API Settings
    PROJECT_NAME: str = "Tomato Disease Classification API"
    VERSION: str = "1.0.0"
    
    # Allowed CORS Origins
    CORS_ORIGINS: list[str] = [
        "http://localhost",
        "http://localhost:5173", # Vite default
        "http://localhost:3000",
        "*" # Temporarily allow all for local dev
    ]

    # Model Paths
    MODEL_DIR: Path = BASE_DIR / "models"
    MODEL_PATH: Path = MODEL_DIR / "mobilenetv2_model.pth"
    CLASS_INDICES_PATH: Path = MODEL_DIR / "class_indices.json"

    # External APIs
    # OpenRouter key loaded from .env
    OPENROUTER_API_KEY: str = os.getenv(
        "OPENROUTER_API_KEY",
        ""
    )

settings = Settings()
