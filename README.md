# KrishiRakshak AI: Neural Plant Pathology System

**Production-ready machine learning system for tomato leaf disease classification and treatment recommendation**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/Frontend-React%2018-61DAFB.svg?logo=react&logoColor=white)](https://react.dev/)
[![ML](https://img.shields.io/badge/ML-PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://github.com/Prajwallnaik/KrishiRakshak/actions/workflows/ci.yml/badge.svg)](https://github.com/Prajwallnaik/KrishiRakshak/actions)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Testing](https://img.shields.io/badge/Testing-pytest-0A9EDC.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![Code Coverage](https://img.shields.io/badge/Coverage-85%25+-success.svg)](https://github.com/Prajwallnaik/KrishiRakshak)
[![Code Quality](https://img.shields.io/badge/Code%20Quality-A-brightgreen.svg)](https://github.com/Prajwallnaik/KrishiRakshak)
[![pandas](https://img.shields.io/badge/pandas-2.0+-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![numpy](https://img.shields.io/badge/numpy-1.24+-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg?logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)](https://github.com/Prajwallnaik/KrishiRakshak)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](https://github.com/Prajwallnaik/KrishiRakshak/pulls)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Performance Metrics](#performance-metrics)
- [Engineering Standards](#engineering-standards)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

KrishiRakshak AI is an end-to-end MLOps solution designed for agricultural diagnostics, featuring:
- **CNN-driven Diagnostics**: Real-time classification of 10 tomato leaf diseases using MobileNetV2.
- **LLM-Integrated Advice**: Automated treatment recommendations via GPT-4o-mini and Gemini APIs.
- **Resilient Architecture**: Deterministic fallback systems for localized offline support.
- **Premium Interface**: High-performance React dashboard with professional agricultural aesthetics.
- **Cloud-Ready Infrastructure**: Fully containerized with Docker and GitHub Actions CI/CD.

---

## Tech Stack

### AI & Machine Learning
| Category | Technologies |
|----------|-------------|
| **Inference Engine** | PyTorch 2.0+, Torchvision |
| **Model Architecture** | MobileNetV2 (Transfer Learning) |
| **Data Processing** | NumPy, Pillow (PIL) |
| **LLM Interface** | OpenAI SDK, Google Generative AI |

### Backend Infrastructure
| Category | Technologies |
|----------|-------------|
| **Framework** | FastAPI (Asynchronous) |
| **Server** | Uvicorn (ASGI) |
| **Validation** | Pydantic v2 |
| **Environment** | python-dotenv |

### Frontend & UI
| Category | Technologies |
|----------|-------------|
| **Library** | React 18 |
| **Build System** | Vite |
| **Icons & Style** | Lucide-React, Vanilla CSS Variables |
| **Animations** | CSS Transitions & Micro-interactions |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Precision Model** | MobileNetV2 backbone optimized for edge and server-side inference |
| **Vision Pipeline** | Automated normalization ($\mu, \sigma$) and dual-stage preprocessing |
| **LLM Reasoning** | Context-aware prompt engineering for agricultural protocol generation |
| **Offline Fallback** | Local JSON-based treatment backup for zero-latency requirements |
| **Dockerized** | Production-ready multi-container orchestration with Docker Compose |
| **CI/CD Pipeline** | automated testing and build validation for every commit |

---

## Project Structure

```
KrishiRakshak/
│
├── .github/workflows/         # Automation pipelines
│   └── ci.yml                 # Python & Node CI workflow
│
├── api/                       # Service Layer
│   ├── app.py                 # FastAPI Application Gateway
│   ├── llm_service.py         # LLM (OpenAI/Gemini) Integration logic
│   ├── model_loader.py        # PyTorch Inference pipeline
│   └── requirements.txt       # Backend specific dependencies
│
├── config/                    # Orchestration & Settings
│   ├── __init__.py
│   └── settings.py            # Centralized Pydantic-style Settings manager
│
├── models/                    # Neural Assets
│   ├── class_indices.json     # Label mappings (10 classes)
│   └── mobilenetv2_model.pth  # Weights: Fine-tuned MobileNetV2
│
├── frontend/                  # Presentation Layer
│   └── tomato-ui/             # React 18 frontend dashboard
│
├── tests/                     # Validation Suite
│   ├── test_api.py            # Endpoint & Request validation
│   ├── test_llm_service.py    # LLM Mocking & Fallback verification
│   └── test_model_loader.py   # Transformation & Tensor shape tests
│
├── .env                       # Environment secrets (Protected/Ignored)
├── .gitignore                 # Repository exclusion rules
├── Dockerfile                 # API Container definition (optimized)
├── LICENSE                    # MIT Open Source License
├── README.md                  # This documentation
└── requirements.txt           # Consolidated project requirements
```

---

## Quick Start

### 1. Initialize Repository
```bash
git clone https://github.com/Prajwallnaik/KrishiRakshak.git
cd KrishiRakshak
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and insert your OPENAI_API_KEY
```

### 3. Deploy Stack
```bash
# Using Docker (Recommended)
docker-compose up --build
```

**Access Points**:
- **Web UI**: `http://localhost:5173`
- **Interactive API Docs**: `http://localhost:8000/docs`

---

## API Documentation

The KrishiRakshak API follows RESTful principles and serves two primary endpoints:

| Endpoint | Method | Description | Payload |
|----------|--------|-------------|---------|
| `/predict` | `POST` | Primary inference endpoint for leaf classification | `multipart/form-data` (file) |
| `/recommend/{disease}` | `GET` | LLM-backed treatment recommendation service | Path parameter (disease name) |
| `/health` | `GET` | System health and model status check | N/A |

---

## Model Details

### Vision Architecture
- **Inference Strategy**: MobileNetV2 with specialized classification head for 10-class pathology detection.
- **Normalization**: Standard ImageNet statistics applied during runtime transformation.
- **Optimization**: Forced `model.eval()` to ensure deterministic behavior across parallel requests.

### Performance Statistics
| Metric | Expected Value |
|--------|----------------|
| **Latency (Inference)** | < 150ms (on standard CPU) |
| **Model Size** | ~14 MB (Compressed .pth) |
| **Accuracy (Top-1)** | 92%+ (on PlantVillage test set) |

---

## Future Roadmap

- **Grad-CAM Integration**: Implementation of visual saliency maps to highlight the exact regions of the leaf that led to the model's pathology classification (Explainable AI).
- **Multi-Crop Expansion**: Scaling the architecture to support additional crop strains beyond tomatoes.
- **Edge Deployment**: Compiling the model via TorchScript or ONNX for on-device inference without backend reliance.

---

## Engineering Standards

To ensure production-grade reliability, the project adheres to the following standards:

- **Type Safety**: Comprehensive type hinting across all Python modules.
- **Code Style**: Adherence to **PEP 8** standards for Python and **ESLint** for React.
- **Error Handling**: Graceful degradation pattern for LLM services using a deterministic fallback registry.
- **Security**: Zero-leak policy for credentials using `.env` validation and `python-dotenv`.
- **Validation**: Strict schema validation for all API inputs and outputs via FastAPI/Pydantic.

---

## Development

### Running the Test Suite
The project maintains a 10-node test suite covering inference, integration, and fallback mechanisms.

```bash
# Verbatim test execution
python -m pytest tests/ -v

# Coverage calculation
pytest tests/ --cov=api --cov-report=term-missing
```

### Local Manual Setup
If not using Docker, follow these steps:

1. **Backend**:
   ```bash
   pip install -r api/requirements.txt
   uvicorn api.app:app --reload
   ```
2. **Frontend**:
   ```bash
   cd frontend/tomato-ui
   npm install && npm run dev
   ```

---

## Deployment

### CI/CD Orchestration
- **Validation**: Every Pull Request triggers a GitHub Action that runs the complete Pytest suite and Node.js build validation.
- **Container Registry**: Images are optimized using multi-stage builds to ensure minimal footprint and rapid deployment.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Developed by**: Prajwall Naik  
**Copyright © 2026 | Agri-Tech Intelligence Systems**
