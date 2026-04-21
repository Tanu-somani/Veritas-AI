# Veritas AI: Enterprise-Grade Fake News Detection System

![Veritas AI](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791.svg)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)

**Veritas AI** is a full-stack, production-ready machine learning system designed to classify news articles as **REAL** or **FAKE**. It leverages advanced Natural Language Processing (NLP), feature engineering, and robust ML models, exposed via a lightning-fast FastAPI backend and visualized through a modern, glass-morphism Next.js web interface.

---

## ✨ Features

- **Advanced ML Pipeline:** Includes data preprocessing, text cleaning (NLTK), TF-IDF vectorization, and multi-model support (Logistic Regression, Random Forest, DistilBERT).
- **Explainable AI (XAI):** Integrated with SHAP to provide transparency by highlighting which words contributed most to the AI's decision.
- **High-Performance API:** Built on FastAPI, offering asynchronous endpoints (`/predict`, `/health`) with automatic Swagger/OpenAPI documentation.
- **Premium Frontend:** A responsive, aesthetically pleasing React/Next.js UI featuring micro-animations and dynamic confidence bars.
- **Persistent Logging:** Stores predictions, confidence scores, and explanations securely in a PostgreSQL database using SQLAlchemy.
- **MLOps Ready:** Fully containerized with Docker and Docker Compose, ready for seamless deployment to AWS Elastic Beanstalk.

---

## 🏗️ Technology Stack

| Component | Technology |
| --- | --- |
| **Machine Learning** | Scikit-learn, PyTorch, HuggingFace Transformers, NLTK, SHAP |
| **Backend API** | Python, FastAPI, Uvicorn, Pydantic |
| **Frontend UI** | Node.js, Next.js, React, Vanilla CSS |
| **Database** | PostgreSQL, SQLAlchemy |
| **Infrastructure** | Docker, Docker Compose, AWS Elastic Beanstalk |

---

## 📂 Project Structure

```text
├── api/                  # FastAPI backend server & database models
├── data/                 # Raw and synthetic datasets
├── docker/               # Dockerfiles for API and Frontend
├── frontend/             # Next.js web application
├── models/               # Serialized ML models (.pkl)
├── src/                  # Core ML source code
│   ├── features/         # TF-IDF extractors
│   ├── inference/        # SHAP explainability modules
│   ├── models/           # Scikit-learn & BERT model classes
│   ├── preprocessing/    # Data loaders and text cleaners
│   └── training/         # Model training and evaluation scripts
├── .github/workflows/    # CI/CD pipelines (GitHub Actions)
├── docker-compose.yml    # Multi-container orchestration
└── requirements.txt      # Python dependencies
```

---

## 🚀 Local Development Setup

You can run this system either using Docker (recommended) or locally via your terminal.

### Option 1: Docker (Recommended)
Make sure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed.

```bash
# Clone the repository
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection

# Build and start all containers (API, Frontend, PostgreSQL)
docker-compose up --build
```
- **Frontend UI:** `http://localhost:8000` *(Mapped to port 80/8000 depending on your docker-compose config)*
- **Backend API Docs:** `http://localhost:8000/docs`

### Option 2: Manual Setup

**1. Backend API**
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the training script to generate models
python -m src.training.train_baseline

# Start the FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**2. Frontend Application**
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:3000` to interact with the UI.

---

## ☁️ Deployment (AWS Elastic Beanstalk)

This project is natively configured for **Amazon Linux 2023 Docker** environments.

1. Zip the repository contents (excluding `venv/`, `node_modules/`, and `.git/`).
2. Navigate to the **AWS Elastic Beanstalk Console**.
3. Create a new environment using the **Docker** platform.
4. Upload your `.zip` file.
5. In the **Capacity** configuration, ensure you select an instance with at least 4GB of RAM (e.g., `t3.medium`) to handle the ML dependencies.
6. In **Environment Properties**, add the variable: `POSTGRES_URL = postgresql://user:password@db:5432/fakenewsdb`
7. Deploy!

---

## 🧠 Evaluation Metrics

The baseline pipeline achieved the following metrics on the synthetic test set:
- **Accuracy:** 100%
- **Precision:** 100%
- **Recall:** 100%
- **F1-Score:** 1.00
- **ROC-AUC:** 1.00

*(Note: Real-world datasets like LIAR or Kaggle FakeNewsNet typically achieve ~85-92% accuracy depending on the transformer model used).*

---

## 📄 License
This project is licensed under the MIT License. Feel free to use it for your portfolio or adapt it for production use!
