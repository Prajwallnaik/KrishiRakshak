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
[![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)](https://github.com/Prajwallnaik/KrishiRakshak)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](https://github.com/Prajwallnaik/KrishiRakshak/pulls)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Development](#development)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

KrishiRakshak AI is an end-to-end MLOps solution designed for agricultural diagnostics, featuring:
- **CNN-driven Diagnostics**: Real-time classification of 10 tomato leaf diseases.
- **LLM-Integrated Advice**: Automated treatment recommendations via GPT-4o-mini and Gemini APIs.
- **Resilient Architecture**: Deterministic fallback systems for localized offline support.
- **Premium Interface**: High-performance React dashboard with professional agricultural aesthetics.
- **Cloud-Ready Infrastructure**: Fully containerized with Docker and GitHub Actions CI/CD.

---

## Tech Stack

### Core Technologies
| Category | Technologies |
|----------|-------------|
| **Inference Engine** | PyTorch, Torchvision |
| **Backend API** | FastAPI, Uvicorn |
| **Frontend UI** | React 18, Vite, Vanilla CSS |
| **LLM Interface** | OpenAI API, Google Gemini |
| **Data Processing**| NumPy, Pillow |

### Development & MLOps
| Category | Technologies |
|----------|-------------|
| **Testing** | pytest, httpx, unittest.mock |
| **DevOps** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Configuration** | Pydantic-style Settings, python-dotenv |
| **Version Control**| Git, GitHub |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **MobileNetV2** | High-efficiency CNN backbone optimized for server-side inference |
| **Vision Pipeline**| Automated normalization ($\mu, \sigma$) and dual-stage preprocessing |
| **LLM Heuristics**| Context-aware prompt engineering for agricultural protocol generation |
| **Fallback Registry**| Local JSON-based treatment backup (Zero-API-latency mode) |
| **Dockerized** | Production-ready multi-container orchestration |
| **Automated CI** | Continuous integration with parallel Python/Node validation |

---

## Project Structure

```
KrishiRakshak/
│
├── .github/workflows/         # Automation pipelines
│   └── ci.yml                 # Python & Node CI workflow
│
├── api/                       # Service Layer
│   ├── app.py                 # FastAPI Gateway
│   ├── llm_service.py         # LLM Integration logic
│   ├── model_loader.py        # PyTorch Inference pipeline
│   └── requirements.txt       # Backend dependencies
│
├── config/                    # Orchestration
│   ├── __init__.py
│   └── settings.py            # Centralized environment manager
│
├── models/                    # Neural Assets
│   ├── class_indices.json     # Label mappings
│   └── mobilenetv2_model.pth  # Trained weights
│
├── frontend/                  # Presentation Layer
│   └── tomato-ui/             # React application source
│
├── tests/                     # Validation Suite
│   ├── test_api.py            # Endpoint unit tests
│   ├── test_llm_service.py    # LLM Mocking tests
│   └── test_model_loader.py   # Transformation tests
│
├── .env                       # Environment secrets (gitignored)
├── .gitignore                 # Exclusion rules
├── Dockerfile                 # API Container specification
├── LICENSE                    # MIT License
├── README.md                  # This file
└── requirements.txt           # Global dependencies
```

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/Prajwallnaik/KrishiRakshak.git
cd KrishiRakshak

# Setup environment
cp .env.example .env
# Edit .env and insert your OPENAI_API_KEY

# Deploy with Docker (Recommended)
docker-compose up --build
```

**Access Points**:
- **Dashboard**: `http://localhost:5173`
- **API Swagger**: `http://localhost:8000/docs`

---

## Model Details

### Vision Architecture
- **Backbone**: MobileNetV2 (Pre-trained on ImageNet, fine-tuned for Plant Pathology).
- **Classification Head**: Dense Linear layer optimized for 10-class multiclass identification.
- **Input Dimension**: $224 \times 224$ pixels, 3 channels (RGB).

### Performance Logic
- **Precision Optimization**: Implements `torch.no_grad()` to suppress gradient calculation during inference.
- **Inference Mode**: Forced `model.eval()` to stabilize batch normalization and dropout layers.

---

## Development

### Running the Test Suite
The project maintains high coverage through automated mocking of external APIs and neural model weights.

```bash
# Run all tests with verbatim output
python -m pytest tests/ -v

# Run specific integration tests
python -m pytest tests/test_api.py
```

### Manual Service Execution

**1. Backend**:
```bash
pip install -r api/requirements.txt
uvicorn api.app:app --reload
```

**2. Frontend**:
```bash
cd frontend/tomato-ui
npm install
npm run dev
```

---

## Deployment

### CI/CD Pipeline
- **Continuous Integration**: Every push to `main` triggers automated testing for both the Python core and the React build process.
- **Docker Automation**: The `Dockerfile` uses a `python:3.10-slim` base to minimize image surface area and improve security.

### GitHub Secrets
To fully utilize the CI pipeline, configure the following in GitHub:
```
OPENAI_API_KEY      # Required for integration testing (optional if mocked)
```

---

## Configuration

The system uses a centralized management pattern in `config/settings.py`.

### Environment Variables (.env)
```env
# Credentials
OPENAI_API_KEY=your_key_here

# App settings
PROJECT_NAME="KrishiRakshak AI"
VERSION="1.0.0"
```

---

## Contributing

1. **Fork** the Repository.
2. **Create** a Feature Branch (`git checkout -b feature/pathology-update`).
3. **Commit** your changes following professional standards.
4. **Push** to the Branch and **Open** a Pull Request.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full legal text.

---

**Developed by**: Prajwall Naik

**Copyright © 2026 | Agri-Tech Intelligence Systems**
