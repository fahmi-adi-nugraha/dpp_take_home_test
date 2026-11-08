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
- **Credentials:** optional. The dataset downloads anonymously. Set
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

## Setup and usage

```bash
uv sync --all-packages

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

The training command runs an 80/10/10 random split, a Ridge linear baseline, a feature ablation over the baseline feature set, and randomized hyperparameter tuning for XGBoost. Ridge alpha and XGBoost were tuned with five-fold `RandomizedSearchCV` (20 and 30 draws, seed 42, negative RMSE). The selected feature set includes grouped weather (`weather_group`) based on validation RMSE. The final XGBoost model was refit on train + validation and evaluated once on test.

### Final metrics and best hyperparameters

| Stage | RMSE | MAE | R² |
|---|---:|---:|---:|
| 5-fold CV, tuned XGBoost | 3.92 | — | — |
| Validation, tuned XGBoost | 3.87 | 3.07 | 0.83 |
| Test, final refit | 4.01 | 3.16 | 0.82 |

The XGBoost search space covered `n_estimators` 200–799, `max_depth` 2–8, `learning_rate` 0.01–0.20, `subsample` 0.60–1.00, `colsample_bytree` 0.40–1.00, `min_child_weight` 1–19, `reg_alpha` and `reg_lambda` 0.001–10.00, and `gamma` 0.001–5.00.

The final model uses the following values. They are saved in `artifacts/metrics.json` and `artifacts/model_metadata.json`, and we verified that the serialized pipeline uses the same values.

| Parameter | Value | Controls | Why we selected it |
|---|---:|---|---|
| `n_estimators` | 453 | Number of boosting rounds. | More rounds let the low learning rate converge; cross-validation balanced fit and runtime. |
| `max_depth` | 7 | Maximum depth of each tree. | Deeper trees capture traffic, weather, distance, and city interactions while the sampling and penalty settings control variance. |
| `learning_rate` | 0.0162 | Shrinkage applied to each boosting round. | A small step size with more rounds generalized better than faster, shallower fits. |
| `subsample` | 0.8654 | Fraction of training rows sampled for each tree. | Mild row subsampling reduced variance without discarding too much signal. |
| `colsample_bytree` | 0.7852 | Fraction of features sampled for each tree. | Feature subsampling decorrelated the trees and improved validation RMSE. |
| `min_child_weight` | 8 | Minimum hessian weight required in a leaf. | Stronger leaves prevented small, noisy splits. |
| `reg_alpha` | 0.00109 | L1 regularization on leaf weights. | A small L1 penalty kept the model sparse without suppressing useful splits. |
| `reg_lambda` | 0.00255 | L2 regularization on leaf weights. | A small L2 penalty stabilized the fit by shrinking leaf weights. |
| `gamma` | 0.00205 | Minimum loss reduction required for a split. | A near-zero threshold allowed modest-gain splits while the other penalties controlled complexity. |

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

## API integration

The delivery-time model is served by `POST /predict/delivery-time` in the local
FastAPI app. The endpoint accepts raw order, location, and courier inputs,
constructs the 18 selected features in `src/delivery_time/inference.py`, and
returns the predicted minutes plus any input-quality warnings. See
[`../../apps/api/README.md`](../../apps/api/README.md) for the request contract
and run command.
