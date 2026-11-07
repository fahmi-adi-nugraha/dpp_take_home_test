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

## Workflow

1. Fetch the dataset with `httpx` and store it under `data/raw/zomato/`.
2. Explore the data in `notebooks/01_eda.py` using Polars and plotnine.
3. Clean and standardize the data in `notebooks/02_data_cleaning.py` using the
   functions in `src/delivery_time/data.py`.
4. Build the semantic feature frame in
   `notebooks/03_feature_engineering.py` using `src/delivery_time/features.py`.
5. Train an XGBoost model wrapped in a scikit-learn `Pipeline` in
   `notebooks/04_model_training.py` and `train.py`.
6. Evaluate the model in `evaluate.py` and the training notebook.
7. Export the trained pipeline and metadata under `artifacts/` for the API.

## Processed outputs

```text
data/processed/
├── zomato_clean.parquet
├── zomato_clean.csv
├── zomato_clean_manifest.json
├── zomato_features.parquet
├── zomato_features.csv
└── zomato_features_manifest.json
```

Parquet is the canonical typed artifact. CSV copies are provided for
inspection. Processed data and manifests are generated locally and are ignored
by Git.

## Layout

- `src/delivery_time/`: importable package, cleaning/feature logic, and CLI entry points.
- `notebooks/`: marimo notebooks for EDA, data cleaning, feature engineering, and model training.
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
uv run marimo edit models/delivery-time/notebooks/02_data_cleaning.py
uv run marimo edit models/delivery-time/notebooks/03_feature_engineering.py
uv run marimo edit models/delivery-time/notebooks/04_model_training.py
uv run train-delivery-model
uv run pytest models/delivery-time/tests
```

## Training and evaluation

The training command runs an 80/10/10 random split, a Ridge linear baseline, a
feature ablation over the baseline feature set, and randomized hyperparameter tuning for
XGBoost. The selected feature set currently includes grouped weather
(`weather_group`) based on validation RMSE.

Current held-out test results:

```text
RMSE: 4.0126
MAE:  3.1646
R²:   0.8153
```

Artifacts:

- `artifacts/model.joblib`: full preprocessing + tuned XGBoost pipeline.
- `artifacts/metrics.json`: baseline, tuned, final, and test metrics.
- `artifacts/feature_schema.json`: expected model inputs and categorical levels.
- `artifacts/model_metadata.json`: feature set, hyperparameters, split, hashes, versions.
- `artifacts/ablation_results.parquet`: feature-ablation results.
- `artifacts/test_predictions.parquet`: held-out predictions and residuals.
- `artifacts/permutation_importance.csv`: test-set permutation importance.
- `artifacts/model_card.md`: concise model summary.

The API reads the model through `DELIVERY_MODEL_PATH`, defaulting to
`models/delivery-time/artifacts/model.joblib`.

## Status

Data fetching, EDA, data cleaning, feature engineering, model training, and
evaluation are implemented. API integration remains a placeholder.
