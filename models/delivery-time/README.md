# Delivery-Time Regression

## Overview

Classical ML regression project for predicting product delivery time. The
business framing is delivery-time estimation for delivery operations, using
the Zomato Delivery Operations Analytics Dataset.

## Dataset

- **Dataset:** Zomato Delivery Operations Analytics Dataset
- **Source:** <https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset>
- **Raw data location:** `data/raw/`
- **Fetch command:** `uv run --env-file .env fetch-delivery-data`
- **Credentials:** `KAGGLE_USERNAME`, `KAGGLE_KEY`

The exact target column must be confirmed during EDA before training begins.

## Target and metrics

- **Target:** product delivery time.
- **Metrics:** RMSE, MAE, and R².

## Planned workflow

1. Fetch the dataset with `httpx` and store it under `data/raw/`.
2. Explore the data in `notebooks/01_eda.py` using Polars and plotnine.
3. Clean and preprocess the data in `data.py` and `features.py`.
4. Train an XGBoost model wrapped in a scikit-learn `Pipeline` in `train.py`.
5. Evaluate the model in `evaluate.py` and `notebooks/02_modeling.py`.
6. Export the trained pipeline and metadata under `artifacts/` for the API.

## Layout

- `src/delivery_time/`: importable package and CLI entry points.
- `notebooks/`: marimo notebooks for EDA and modeling.
- `data/raw/`, `data/interim/`, `data/processed/`: local dataset stages.
- `artifacts/`: trained model and metadata; not committed.
- `tests/`: package tests.

## Commands

```bash
uv run marimo edit models/delivery-time/notebooks/01_eda.py
uv run --env-file .env fetch-delivery-data
uv run train-delivery-model
uv run pytest models/delivery-time/tests
```

## Artifact contract

- `artifacts/model.joblib`: full preprocessing + XGBoost pipeline.
- `artifacts/metrics.json`: evaluation metrics.
- `artifacts/feature_schema.json`: expected input feature names and types.

The API reads the model through `DELIVERY_MODEL_PATH`, defaulting to
`models/delivery-time/artifacts/model.joblib`.

## Status

Scaffold only. No implementation yet.
