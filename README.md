# DPP DSML Take-Home Test

Monorepo for the DPP DSML take-home test. It contains two independent ML
projects and a local FastAPI inference service, managed as a single `uv`
workspace.

> **Status:** scaffold only. Training, EDA, data fetching, API behavior, and
> tests are not implemented yet. The commands below describe the intended
> workflow.

## Projects

| Exam project | Package | Path | Task | API endpoint |
|---|---|---|---|---|
| Project 1 | `dpp-delivery-time` | `models/delivery-time` | Zomato delivery-time regression | `POST /predict/delivery-time` |
| Project 2 | `dpp-yelp-sentiment` | `models/yelp-sentiment` | Yelp review sentiment | `POST /predict/sentiment` |
| Service | `dpp-inference-api` | `apps/api` | Local FastAPI inference | — |

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

Copy the environment template and fill in the credentials needed for the
chosen data sources:

```bash
cp .env.example .env
```

Then run the planned fetch commands:

```bash
uv run --env-file .env fetch-delivery-data
uv run --env-file .env fetch-sentiment-data
```

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

This repository is currently a scaffold. It does not include training code,
EDA, data fetching, API logic, a database, authentication, Docker, or CI.
