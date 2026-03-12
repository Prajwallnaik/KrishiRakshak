# Systems Architecture: KrishiRakshak AI

This document provides an in-depth technical overview of the KrishiRakshak AI system, detailing the data flow, model architecture, and backend infrastructure.

## 1. High-Level Architecture

The system is designed as a decoupled, microservice-inspired monorepo.

*   **Frontend**: A React 18 single-page application (SPA) built with Vite, handling image acquisition and displaying cinematic diagnostic dashboards.
*   **Backend**: A FastAPI asynchronous service that acts as the primary gateway for both image inference and LLM orchestration.
*   **Inference Engine**: A PyTorch runtime environments loaded with a specially fine-tuned MobileNetV2 architecture.
*   **Intelligence Layer**: Seamless integration with the OpenAI/Gemini APIs for localized treatment generation.

## 2. Neural Architecture (MobileNetV2)

The core vision component utilizes **MobileNetV2**, chosen for its optimal balance between parameter count and classification accuracy.

### 2.1 Why MobileNetV2?
*   **Inverted Residual Blocks**: Allows for efficient feature extraction with minimal memory overhead.
*   **Edge-Ready**: The lightweight nature of the model (<15MB) ensures that future deployments to edge devices (e.g., Raspberry Pi or mobile apps) are feasible without architectural rewrites.

### 2.2 Model Pipeline
1.  **Input**: RGB Image ($224 \times 224$ pixels).
2.  **Preprocessing**:
    *   Resize and Center Crop.
    *   ToTensor conversion.
    *   ImageNet Normalization: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
3.  **Forward Pass**: Forced `model.eval()` and `torch.no_grad()` to ensure deterministic output and prevent gradient tracking memory leaks.
4.  **Output**: Softmax probability distribution across 10 classes.

## 3. Data Flow & State Management

### 3.1 Inference Workflow
1.  The user uploads an image via the React frontend.
2.  The image is sent as `multipart/form-data` to the FastAPI `/predict` endpoint.
3.  FastAPI passes the byte stream to `model_loader.py`.
4.  The PyTorch engine processes the image and returns the predicted class and confidence score.
5.  FastAPI responds to the frontend with the diagnosis and inference metadata (e.g., latency).

### 3.2 Generative AI Workflow
1.  Upon receiving a diagnosis, the frontend triggers a `GET` request to `/recommend/{disease_name}`.
2.  The backend's `llm_service.py` constructs a highly structured prompt incorporating the disease name.
3.  An asynchronous call is made to the LLM provider (OpenAI/Gemini).
4.  **Resilience**: If the API call fails (timeout or quota exceeded), the system automatically falls back to a local, deterministic JSON registry of treatments to ensure the user always receives actionable data.

## 4. Training Methodology (Summary)
*   **Dataset**: PlantVillage Dataset (Tomato specific subsets).
*   **Classes**: 10 (9 disease states, 1 healthy state).
*   **Transfer Learning**: The backbone weights were frozen initially, and a custom linear classifier was trained. Subsequently, the top layers were unfrozen for fine-tuning with a lower learning rate.
*   **Metrics**: The final model achieves >92% accuracy on the validation split.

## 5. Security and Environment
*   All secrets (API keys) are managed via `python-dotenv` and completely isolated from the source code.
*   Cross-Origin Resource Sharing (CORS) is strictly configured in FastAPI to prevent unauthorized domain access.
