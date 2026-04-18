# Demand Forecasting Backend System (Wildberries Hackathon Project)

## Overview

This project is a backend system for demand forecasting across logistics routes, built during a 3-day hackathon. The system predicts future values of `target_2h` across multiple time horizons using machine learning models trained on engineered time-series features.

The solution is built around a modular backend architecture with clear separation between data processing, feature engineering, model training, inference, and evaluation. It uses an ensemble of CatBoost and LightGBM models for robust predictions.

---

## Key Features

- Multi-horizon forecasting (1–10 steps ahead)
- Ensemble of CatBoost + LightGBM models
- Modular feature engineering pipeline (step-based system)
- Custom validation and backtesting logic
- Time-series lag features and statistical aggregations
- Support for categorical and missing data handling
- Full training + inference + submission pipeline

---

## System Architecture

### 1. Data Layer
- Loads training and prediction datasets from Parquet files
- Supports time-based splitting for train/validation scenarios
- Implements sliding window training logic

---

### 2. Feature Engineering Pipeline

A modular pipeline system where each transformation is an independent step.

Feature types:
- Timestamp decomposition (hour, day of week, day of month)
- Cyclical encoding (sin/cos transformations)
- Route-level aggregations (mean, historical patterns)
- Lag features (1–12 steps)
- Flow speed estimation
- Anomaly detection flags
- Target-based statistical features per route

Each feature is implemented as a class with a unified interface:
FeatureStep → apply(table, context)

Pipeline execution is registry-based and fully configurable.

---

### 3. Model Layer

CatBoost Model Manager:
- One model per horizon
- Native categorical feature support
- Early stopping with validation sets
- Strong performance on mixed-type data

LightGBM Model Manager:
- One model per horizon
- Optimized for large-scale numerical features
- Categorical preprocessing via dtype conversion

---

### 4. Ensemble Strategy

Final prediction:

final_prediction = 0.5 * CatBoost + 0.5 * LightGBM

This improves robustness and reduces variance across horizons.

---

### 5. Training Strategies

Test Strategy:
- Train/validation split (80/20)
- Used for local evaluation and debugging
- Computes metrics per horizon

Submission Strategy:
- Full dataset training
- No validation split
- Generates final predictions for submission

---

## Evaluation Metrics

- WAPE (Weighted Absolute Percentage Error)
- Bias
- Score:

Score = WAPE + Bias

---

## Tech Stack

- Python 3.10+
- Pandas / NumPy
- CatBoost
- LightGBM
- Parquet data format
- Object-oriented architecture
- Design patterns (Strategy, Factory, Registry, Pipeline)

---

## Design Patterns Used

- Strategy Pattern → training/inference modes
- Factory Pattern → model creation
- Pipeline Pattern → feature engineering
- Registry Pattern → feature step execution
- Modular service-based backend design

---

## Engineering Decisions

- Horizon-based independent models (no shared weights)
- Strict separation of concerns (data / features / models)
- Fully modular feature system for extensibility
- Explicit handling of missing/edge cases
- Lightweight ensemble instead of complex stacking
- Designed for fast iteration under hackathon constraints

---

## What This Project Demonstrates

- Backend architecture design for ML systems
- Time-series forecasting pipelines
- Feature engineering at scale
- Multi-model orchestration (ensemble systems)
- Production-style engineering under tight deadlines
- Ability to build end-to-end ML backend systems

---

## Notes

The project was implemented in 3 days under hackathon constraints, prioritizing:
- System stability
- Modularity
- Reproducibility
- Extensibility

Despite time limitations, the architecture was designed to be production-like and scalable.
