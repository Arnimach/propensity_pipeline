# propensity_pipeline

# 🎯 Customer Purchase Propensity Pipeline

An end-to-end, production-grade MLOps system designed to predict real-time online shopper purchase intent using a **Rank-Averaged Ensemble** of tree-based models. Built with **FastAPI**, **MLflow**, **Docker Compose**, and **Scikit-Learn / XGBoost / LightGBM**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-0194E2?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)

---

## 📌 Project Overview

This project builds, evaluates, logs, and serves a high-performance machine learning pipeline on web session analytics data (e.g., page duration, bounce rates, special day index) to identify high-intent visitors and drive conversion.

### Key Highlights
- **Dynamic Ensemble**: Trains multiple hyperparameter variants of **RandomForest**, **XGBoost**, and **LightGBM** across multiple random seeds.
- **Out-of-Fold (OOF) CV & Diversity Selection**: Evaluates model variants via 5-Fold Stratified Cross-Validation, optimizes $F_2$ score to prioritize recall, and filters out redundant models via prediction correlation thresholds ($\rho < 0.95$).
- **Custom MLflow PyFunc Wrapper**: Encapsulates selected model pipelines and averaging logic into a single MLflow artifact, promoted to `@Production`.
- **Low-Latency REST API**: Exposes a FastAPI application with an asynchronous `lifespan` context manager that pre-loads the `@Production` model at startup.
- **Microservices Architecture**: Containerized with Docker Compose to coordinate the FastAPI server and local MLflow Tracking Server.

---

## 🏗 System Architecture

```text
├── data/
│   └── clean_data.csv          # Preprocessed web analytics dataset
├── api/
│   ├── main.py                 # FastAPI inference application & endpoints
│   └── schemas.py              # Pydantic input/output validation schemas
├── src/
│   ├── train.py                # Model training, OOF ensembling & MLflow logging script
│     
├── notebooks/
│   └── download_and_eda.ipynb  # Exploratory Data Analysis & baseline experiments
├── Dockerfile.api              # Container specification for FastAPI service
├── docker-compose.yml          # Multi-container orchestration (FastAPI + MLflow)
├── requirements.txt            # Python dependencies
└── README.md
