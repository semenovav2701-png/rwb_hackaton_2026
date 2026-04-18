# Project Overview

This project is an end-to-end machine learning and backend system designed for time-series demand prediction and decision-making based on aggregated model outputs. It combines feature engineering pipelines, multi-horizon forecasting models, ensemble prediction logic, and a FastAPI-based service layer for serving predictions and decisions.

---

# System Architecture

The system is split into two main parts:

## 1. Machine Learning Pipeline (`lm_model`)

Responsible for data processing, feature engineering, model training, inference, and submission generation.

### Key Components:

### Data Layer (`tables.py`)

* `FilledTable`: used during training, contains full historical data.
* `EmptyTable`: used during inference, relies on parental historical context.
* Supports feature creation such as:

  * Time-based features (hour, day of week, cyclic encoding)
  * Lag features
  * Rolling and aggregated statistics
  * Route-based historical statistics

### Feature Engineering (`features_adder.py`)

A modular pipeline that applies transformations:

* Timestamp parsing
* Cyclic encodings (hour/day-of-week)
* Flow speed estimation
* Lag features creation
* Route-level statistical features
* Anomaly detection flags

### Models (`model.py`)

Two main model families are used:

* CatBoost Regressor
* LightGBM Regressor

Both are trained independently per forecasting horizon (multi-horizon learning strategy).

### Training Strategy (`model_strategies.py`)

Supports interchangeable training strategies:

* `TestStrategy`: includes train/validation split and evaluation
* `SubmissionStrategy`: full-data training for final inference

Each horizon is trained as a separate model.

### Configuration (`config.py`)

Centralized configuration for:

* Model hyperparameters (CatBoost, LightGBM)
* Feature lists
* Application settings (data paths, time windows, split dates)
* Forecast horizons

### Data Handler

Responsible for:

* Loading parquet datasets
* Time-window based splitting
* Preparing training and prediction datasets
* Writing final submission CSV

### Ensemble Logic

Final predictions are generated using a simple weighted ensemble:

* 50% CatBoost
* 50% LightGBM

Metrics used:

* WAPE (Weighted Absolute Percentage Error)
* Bias
* Combined score

---

# Backend Service (`backend`)

A FastAPI-based microservice responsible for serving predictions, aggregations, and decision-making logic.

## Core Design

The backend follows a strategy-based architecture:

* Aggregation Strategy
* Decision Strategy
* Source abstraction via factories

---

## API Endpoints

### Health Check

```
GET /health
```

Returns service status.

---

### Predictions API

```
GET /predictions
POST /predictions
```

* Stores and retrieves prediction records
* Uses in-memory storage (demo implementation)

---

### Planning API

```
POST /plan
```

Main endpoint that:

1. Aggregates predictions by route and timestamp
2. Applies decision logic to compute required resources (e.g., trucks)
3. Returns final planning output

---

## Aggregation Layer (`aggregation.py`)

Responsible for grouping raw predictions.

### StrategyCounter

* Groups predictions by:

  * `office_from_id`
  * `timestamp`
* Sums predicted values into total volume

---

## Decision Engine (`decision.py`)

Transforms aggregated volumes into actionable decisions.

### VolumeBasedTruckCounter

* Converts total volume into number of trucks
* Applies:

  * buffer coefficient
  * min/max constraints

---

## Application Layer (`application.py`)

Orchestrates the full pipeline:

1. Aggregation
2. Decision making
3. Output generation

---

## Strategy Pattern System

The backend uses factories to dynamically select implementations:

* `AggregationStrategyFactory`
* `DecisionStrategyFactory`
* `DataSourceFactory`

This allows easy extension without modifying core logic.

---

## Data Sources

Supports pluggable prediction sources.

### Demo Source

Provides static mock predictions for testing and development.

---

## Schema (`schemas.py`)

Defines core data model:

```python
Prediction:
- route_id
- office_from_id
- timestamp
- predicted_target_2h
```

---

# Key Features

* Multi-horizon forecasting (1–10 steps ahead)
* Dual-model ensemble (CatBoost + LightGBM)
* Modular feature engineering pipeline
* Strategy-based backend architecture
* Extensible data source system
* REST API for prediction and planning
* Time-series aware training and inference

---

# Typical Workflow

1. Load historical data
2. Build features
3. Train models per horizon
4. Evaluate (optional validation mode)
5. Generate predictions
6. Aggregate predictions in backend
7. Apply decision logic
8. Output final operational plan

---

# Output Example

Final system output typically includes:

* Route-level or office-level aggregated demand
* Required resource allocation (e.g., number of trucks)
* Timestamped planning decisions

---

# Notes

* Designed for extensibility and experimentation
* Supports both training and production inference modes
* Uses modular design for ML + backend separation

