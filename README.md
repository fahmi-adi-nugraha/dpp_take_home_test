# DPP DSML Take-Home Test

Monorepo for the DPP DSML take-home test. It contains two independent ML
projects and a local FastAPI inference service, managed as a single `uv`
workspace.

> **Status:** classical ML data fetching, EDA, data cleaning, feature
> engineering, model training, and evaluation are implemented. NLP data
> fetching, API behavior, and the remaining tests are not implemented yet. The
> commands below describe the intended workflow.

## Projects

| Exam project | Package | Path | Task | API endpoint |
|---|---|---|---|---|
| Project 1 | `dpp-delivery-time` | `models/delivery-time` | Zomato delivery-time regression | `POST /predict/delivery-time` |
| Project 2 | `dpp-yelp-sentiment` | `models/yelp-sentiment` | Yelp review sentiment | `POST /predict/sentiment` |
| Service | `dpp-inference-api` | `apps/api` | Local FastAPI inference | — |

## Guideline alignment

The take-home test requires two independent ML projects and a deployed REST
API, with an optional Streamlit UI and a final report and presentation.

| Guideline requirement | Where it is addressed |
|---|---|
| Project 1: simple ML model (regression, classification, or clustering) | `models/delivery-time` scopes a regression model for Zomato delivery time. |
| Project 2: NLP, CV, or RAG model | `models/yelp-sentiment` scopes an NLP sentiment model using a BERT-family transformer. |
| REST API deployment | `apps/api` contains the FastAPI service with planned endpoints for both models. |
| Streamlit UI (optional) | Future slot: `apps/streamlit`. |
| Final report and presentation | Not yet added. |

The required workflow for each project is:

- define a specific, relevant, and measurable business problem;
- choose a dataset and document the methodology, assumptions, and target
  definition;
- handle missing values, duplicate rows, anomalies, type conversion,
  categorical encoding, and numeric scaling or standardization;
- perform EDA covering distributions, feature relationships, and documented
  insights;
- train and evaluate the model using task-appropriate metrics;
- serve the best model through the REST API with a prediction endpoint,
  example request and response, and local run instructions;
- explain the performance-monitoring plan and when the model should be
  retrained.

Suggested evaluation metrics:

- Project 1 regression: RMSE, MAE, and R².
- Project 2 NLP classification: accuracy, precision, recall, and F1; macro F1
  is useful for class balance.
- If Project 2 changes to generation, computer vision, or RAG, use the
  guideline metrics such as BLEU, perplexity, IoU, Hit@K, or MRR as
  appropriate.

Datasets may come from the provided references or another open-source or real
dataset. The current choices are the Zomato Delivery Operations Analytics
Dataset and the Yelp review dataset.

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

`models/*` and `apps/*` are workspace globs, so new model packages and new
apps are picked up automatically when they contain a `pyproject.toml`.

Each model package contains:

- `src/<package>/` for importable code
- `notebooks/` for marimo notebooks
- `data/{raw,interim,processed}/` for local datasets
- `artifacts/` for trained model outputs
- `tests/` for package tests

## Prerequisites

- `uv` 0.13 or newer
- Git
- Python 3.12, managed/installed by `uv` from `.python-version`

## Quickstart

```bash
uv sync --all-packages
uv run python -c "import delivery_time, yelp_sentiment, inference_api"
```

Intended commands once implementation is complete:

```bash
uv run uvicorn inference_api.main:app --reload
uv run pytest
```

The API will be available at `http://127.0.0.1:8000`, with interactive docs at
`http://127.0.0.1:8000/docs`.

## Data setup

Kaggle credentials are optional for the public Zomato dataset. Copy the
environment template if you want authenticated Kaggle requests or need a Hugging
Face token for the Yelp dataset:

```bash
cp .env.example .env
```

Fetch commands:

```bash
# Zomato: anonymous download works for the public dataset
uv run fetch-delivery-data

# Zomato: optional authenticated download
uv run --env-file .env fetch-delivery-data

# Yelp: planned, not implemented yet
uv run --env-file .env fetch-sentiment-data
```

The field-level Zomato dataset dictionary is at
[`models/delivery-time/DATA_DICTIONARY.md`](models/delivery-time/DATA_DICTIONARY.md).

## Data and artifact policy

- Raw, interim, and processed datasets are not committed.
- Trained model artifacts are not committed.
- `.gitkeep` files preserve the directory structure in Git.
- `uv.lock` is committed for reproducible dependency resolution.
- `.venv/` is generated locally and ignored.

## Configuration

- There is one root `.venv` and one root `uv.lock` for the whole workspace.
- All commands are run through `uv run` from the repository root.
- API model artifact paths can be overridden with `DELIVERY_MODEL_PATH` and
  `SENTIMENT_MODEL_PATH`.

## Adding a new model

1. Create `models/<slug>/` with a `pyproject.toml`, `README.md`, `src/`,
   `notebooks/`, `data/`, `artifacts/`, and `tests/`.
2. The `models/*` workspace glob picks up the new member automatically.
3. If the API needs the model, add it as an API dependency and add a root
   `[tool.uv.sources]` entry.

## Adding a new app

Create `apps/<slug>/` with a `pyproject.toml` and package source. The
`apps/*` workspace glob picks it up automatically. `apps/streamlit` is the
intended future slot for the optional UI.

## Scope and status

Classical ML data fetching, EDA, data cleaning, feature engineering, model
training, and evaluation are implemented. Still pending:

- NLP data fetching, training, and evaluation
- API logic
- optional Streamlit UI
- final report and presentation

The repository does not include a database, authentication, Docker, or CI.
