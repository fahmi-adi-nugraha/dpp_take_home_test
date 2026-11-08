# DPP DSML Take-Home Test

A `uv` workspace containing two independent machine-learning projects and a
local FastAPI inference service.

| Project | Package | Path | Task | API endpoint |
|---|---|---|---|---|
| Delivery time | `dpp-delivery-time` | `models/delivery-time` | Zomato delivery-time regression | `POST /predict/delivery-time` |
| Yelp sentiment | `dpp-yelp-sentiment` | `models/yelp-sentiment` | Five-class Yelp review sentiment | `POST /predict/sentiment` |
| Inference service | `dpp-inference-api` | `apps/api` | Local FastAPI inference | — |

## Final results

### Delivery time

The tuned XGBoost regressor uses the `plus_weather_group` feature set.

| Stage | RMSE | MAE | R² |
|---|---:|---:|---:|
| 5-fold CV | 3.92 | — | — |
| Validation | 3.87 | 3.07 | 0.83 |
| Held-out test | 4.01 | 3.16 | 0.82 |

### Yelp sentiment

The BERT model is compared with a TF-IDF logistic regression baseline on the
full 50,000-review test split.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| BERT | 0.68 | 0.67 | 0.67 |
| TF-IDF logistic regression | 0.58 | 0.58 | 0.58 |

## Repository layout

```text
.
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── .env.example
├── README.md
├── models/
│   ├── delivery-time/
│   └── yelp-sentiment/
└── apps/
    └── api/
```

`models/*` and `apps/*` are workspace globs. New packages are picked up
automatically when they contain a `pyproject.toml`.

Each model package contains:

- `src/<package>/` for importable code and CLI entry points
- `notebooks/` for marimo analysis and reporting notebooks
- `data/{raw,interim,processed}/` for local datasets
- `artifacts/` for trained model outputs
- `tests/` for package tests

## Prerequisites

- `uv` 0.13 or newer
- Git
- Python 3.12, managed by `uv` from `.python-version`

## Setup

Install the workspace and verify the packages import:

```bash
uv sync --all-packages
uv run python -c "import delivery_time, yelp_sentiment, inference_api"
```

Copy the environment template when authenticated Kaggle requests or Hugging
Face rate limits are needed:

```bash
cp .env.example .env
```

### Fetch the datasets

```bash
# Zomato: anonymous download works for the public dataset
uv run fetch-delivery-data

# Zomato: optional authenticated download
uv run --env-file .env fetch-delivery-data

# Yelp: anonymous download works for the public dataset
uv run fetch-sentiment-data

# Yelp: optional authenticated download if HF_TOKEN is configured
uv run --env-file .env fetch-sentiment-data
```

The field-level Zomato dataset dictionary is at
[`models/delivery-time/DATA_DICTIONARY.md`](models/delivery-time/DATA_DICTIONARY.md).

## Model artifacts

The trained artifacts are published as a GitHub release:

```text
https://github.com/fahmi-adi-nugraha/dpp_take_home_test/releases/tag/models-v1.0.0
```

Download and verify the archive:

```bash
curl -L -o /tmp/dpp-inference-models-v1.zip \
  https://github.com/fahmi-adi-nugraha/dpp_take_home_test/releases/download/models-v1.0.0/dpp-inference-models-v1.zip
curl -L -o /tmp/dpp-inference-models-v1.zip.sha256 \
  https://github.com/fahmi-adi-nugraha/dpp_take_home_test/releases/download/models-v1.0.0/dpp-inference-models-v1.zip.sha256
(cd /tmp && sha256sum -c dpp-inference-models-v1.zip.sha256)
```

Extract it from the repository root so the API can find the artifacts:

```bash
unzip -o /tmp/dpp-inference-models-v1.zip -d .
```

The archive contains both the delivery-time pipeline and the Yelp sentiment
bundle, plus their metadata, evaluation outputs, and checksums.

## Model commands

### Delivery time

```bash
uv run marimo edit models/delivery-time/notebooks/01_eda.py
uv run marimo edit models/delivery-time/notebooks/02_data_cleaning.py
uv run marimo edit models/delivery-time/notebooks/03_feature_engineering.py
uv run marimo edit models/delivery-time/notebooks/04_model_training.py
uv run train-delivery-model
uv run pytest models/delivery-time/tests
```

### Yelp sentiment

```bash
uv run marimo edit models/yelp-sentiment/notebooks/01_eda.py
uv run marimo edit models/yelp-sentiment/notebooks/02_data_cleaning.py
uv run marimo edit models/yelp-sentiment/notebooks/03_feature_engineering.py
uv run marimo edit models/yelp-sentiment/notebooks/04_finetune.py
uv run marimo edit models/yelp-sentiment/notebooks/05_evaluate.py
uv run train-sentiment-model
uv run evaluate-sentiment-models
uv run export-sentiment-model
uv run pytest models/yelp-sentiment/tests
```

## API usage

Start the local FastAPI service:

```bash
uv run uvicorn inference_api.main:app --reload
```

The service is available at `http://127.0.0.1:8000`, with interactive
documentation at `http://127.0.0.1:8000/docs`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Model and service readiness |
| `GET` | `/models` | Per-model readiness details |
| `POST` | `/predict/delivery-time` | Delivery-time regression |
| `POST` | `/predict/sentiment` | Five-class sentiment classification |

Example delivery-time request:

```bash
curl -X POST http://127.0.0.1:8000/predict/delivery-time \
  -H "Content-Type: application/json" \
  -d '{
    "restaurant": {"latitude": 12.9716, "longitude": 77.5946},
    "delivery": {"latitude": 12.9352, "longitude": 77.6245},
    "order_time": "12:30",
    "weather": "Sunny",
    "traffic": "Medium",
    "city": "Urban",
    "festival": "No",
    "multiple_deliveries": 1,
    "vehicle_condition": 1,
    "courier_age": 30,
    "courier_rating": 4.7
  }'
```

Example sentiment request:

```bash
curl -X POST http://127.0.0.1:8000/predict/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "The food was great and the delivery was fast."}'
```

See [`apps/api/README.md`](apps/api/README.md) for request fields, response
schemas, error behavior, and configuration.

## Configuration

- One root `.venv` and one root `uv.lock` serve the workspace.
- All commands run through `uv run` from the repository root.
- `DELIVERY_MODEL_PATH` overrides the delivery-time pipeline artifact.
- `SENTIMENT_MODEL_PATH` overrides the exported sentiment `.pt` bundle.
- `SENTIMENT_TOKENIZER_PATH` overrides the sentiment tokenizer directory.
- `SENTIMENT_DEVICE` selects `auto`, `cpu`, or `cuda` sentiment inference.
- `LOG_LEVEL` controls API logging, defaulting to `INFO`.

## Data and artifact policy

- Raw, interim, and processed datasets are local and ignored by Git.
- Trained model artifacts and deployment bundles are distributed through the
  `models-v1.0.0` GitHub release and ignored by Git.
- `.gitkeep` files preserve the directory structure in Git.
- `uv.lock` is committed for reproducible dependency resolution.
- `.venv/` is generated locally and ignored.

## Extending the workspace

To add a model, create `models/<slug>/` with a `pyproject.toml`, `README.md`,
`src/`, `notebooks/`, `data/`, `artifacts/`, and `tests/`. Add it as a dependency
of the API and add a root `[tool.uv.sources]` entry when the service serves it.

To add an app, create `apps/<slug>/` with a `pyproject.toml` and package source.

## Deployment scope

The service is a local demonstration. It intentionally omits a database,
authentication, rate limiting, Docker configuration, and production CI.
