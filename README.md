# KrishiRakshak AI: Neural Plant Pathology Classification and Treatment System

## Technical Overview

KrishiRakshak AI is a production-grade inference engine designed for the identification of pathological conditions in tomato foliage. The architecture integrates deep convolutional neural networks (CNNs) for vision-based diagnostics with large language model (LLM) heuristics for agricultural treatment protocols.

## Machine Learning Architecture

### Vision Transformer and CNN Optimization
The system utilizes a **MobileNetV2** architecture, optimized for latency-sensitive environments. The model was trained using transfer learning on a specialized dataset of tomato leaf images, achieving high precision across ten distinct classes of plant stress.

* **Inference Pipeline**: Raw image data is normalized using standard ImageNet statistics ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$) and resized to a $224 \times 224$ tensorspace.
* **Optimization**: The model leverages PyTorch's `eval()` mode and `torch.no_grad()` context managers to minimize memory overhead and maximize throughput during real-time inference.

### LLM-Augmented Decision Support
Upon classification of a disease state, the system triggers an asynchronous request to an LLM provider (GPT-4o / Gemini). The prompt engineering is designed for high-fidelity JSON extraction, ensuring that recommendations are structured for programmatic consumption by the React interface.

* **Fail-Safe Mechanism**: A deterministic fallback registry is implemented to handle API latency or rate-limiting (429 errors), ensuring system availability even in degraded network conditions.

## Software Engineering Standards

### Backend Infrastructure
The backend is built on **FastAPI**, leveraging asynchronous request handling for Non-blocking I/O during model loading and API orchestration. 

### Frontend Aesthetics and UX
The interface follows a minimalist, data-driven design philosophy. It utilizes a custom CSS variable system for consistent thematic tokens, glassmorphism for visual depth, and optimized asset delivery for performance.

### DevOps and CI/CD
- **Containerization**: The environment is encapsulated via Docker, providing parity between development and production stages.
- **Automated Validation**: A 10-node test suite covers API endpoints, model preprocessing logic, and LLM fallback paths using `pytest` and `unittest.mock`.
- **Dependency Management**: Integrated with GitHub Dependabot and standardized requirements files for both root and API sub-directories.

## Repository Structure

```text
.
├── api/                # FastAPI application and inference services
├── config/             # Centralized configuration and environment orchestration
├── frontend/           # React 18 / Vite source code
├── models/             # PyTorch weights (.pth) and metadata
├── tests/              # Automated Python validation suite
├── .github/            # GitHub Actions (CI) and repository management
├── Dockerfile          # OCI-compliant backend container definition
└── docker-compose.yml  # Multi-service orchestration
```

## Setup and Deployment

### Development Environment
The system requires Python 3.10+ and Node.js 20+. Detailed setup instructions for virtual environments and node package installation are provided in the documentation.

### Orchestration
To deploy the full-stack system:
```bash
docker-compose up --build
```
Access points:
- API Documentation (Swagger): `http://localhost:8000/docs`
- Production Frontend: `http://localhost:5173`

## Validation Suite
To execute the automated testing pipeline:
```bash
python -m pytest tests/ -v
```

## License
This project is distributed under the MIT License. See [LICENSE](LICENSE) for the full legal text.
