# Delivery-Time Regression

## Overview

Classical ML regression project for predicting product delivery time. The
business framing is delivery-time estimation for delivery operations, using
the Zomato Delivery Operations Analytics Dataset.

## Dataset

- **Dataset:** Zomato Delivery Operations Analytics Dataset
- **Source:** <https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset>
- **Raw data location:** `data/raw/zomato/`
- **Fetch command:** `uv run fetch-delivery-data`
- **Data dictionary:** [DATA_DICTIONARY.md](DATA_DICTIONARY.md)
- **Feature engineering checklist:** [feature_engineering_checklist.md](feature_engineering_checklist.md)
- **Credentials:** optional. The dataset currently downloads anonymously. Set
  `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env` to use authenticated requests.

The fetched dataset contains 45,584 data rows and 20 columns. The confirmed
target column is `Time_taken (min)`.

## Target and metrics

- **Target:** product delivery time.
- **Metrics:** RMSE, MAE, and R².

## Fetched output

The fetch command stores the archive, extracted CSV, and metadata manifest
together:

```text
data/raw/zomato/
├── zomato-delivery-operations-analytics-dataset.zip
├── Zomato Dataset.csv
└── fetch-manifest.json
```

The manifest records the source URL, download timestamp, archive size and
SHA-256, CSV row count and columns, and target column.

## Planned workflow

1. Fetch the dataset with `httpx` and store it under `data/raw/zomato/`.
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
# Anonymous public download
uv run fetch-delivery-data

# Optional authenticated download
uv run --env-file .env fetch-delivery-data

# Re-download and re-extract
uv run fetch-delivery-data --force

# Custom destination and verbose logging
uv run fetch-delivery-data --output-dir /tmp/zomato --verbose

uv run marimo edit models/delivery-time/notebooks/01_eda.py
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

Data fetching is implemented. EDA, preprocessing, training, evaluation, and
API integration remain placeholders.
