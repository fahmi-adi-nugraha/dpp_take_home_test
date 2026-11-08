# Yelp Sentiment Classification

## Overview

NLP sentiment classification project. We fine-tune a BERT-family model with
Hugging Face `transformers` and compare it with a TF-IDF logistic regression
baseline.

## Dataset

- **Preferred source:** <https://huggingface.co/datasets/Yelp/yelp_review_full>
- **Alternative source:** <https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data>
- **Raw data location:** `data/raw/yelp_review_full/`
- **Fetch command:** `uv run fetch-sentiment-data`
- **Credentials:** not required for the public dataset; set `HF_TOKEN` when
  rate limits or access restrictions require it.

The fetcher downloads the two parquet files from the Hugging Face dataset,
pinned to commit `c1f9ee939b7d05667af864ee1cb066393154bf85`, and validates their
checksums, schema, row counts, and labels.

| Split | Rows | Classes |
|---|---:|---:|
| train | 650,000 | 5 balanced classes |
| test | 50,000 | 5 balanced classes |

The raw labels are `0–4`, corresponding to 1–5 stars. We keep the five-class
target. A `fetch-manifest.json` records file sizes, SHA-256 hashes, row counts,
label counts, and the dataset revision.

## Target and metrics

- **Target:** five-class review sentiment derived from star ratings.
- **Metrics:** accuracy, macro F1, weighted F1, and per-class precision/recall.
- **Primary metric:** macro F1 because each star rating is weighted equally.

## Workflow

1. Fetch the dataset with `httpx` and store it under `data/raw/`.
2. Explore the data in `notebooks/01_eda.py` using Polars and plotnine.
3. Clean and decode the text in `notebooks/02_data_cleaning.py`.
4. Build the reduced modeling split in `notebooks/03_feature_engineering.py`.
5. Tune and fine-tune the models in `train.py` and `notebooks/04_finetune.py`.
6. Evaluate the final models in `evaluate.py` and `notebooks/05_evaluate.py`.
7. Export the model and tokenizer under `artifacts/` for the API.

## Layout

- `src/yelp_sentiment/`: importable package and CLI entry points.
- `notebooks/`: marimo notebooks for EDA, cleaning, features, fine-tuning, and evaluation.
- `data/raw/`, `data/interim/`, `data/processed/`: local dataset stages.
- `artifacts/`: trained model, tokenizer, and metadata; not committed.
- `tests/`: package tests.

## Setup and usage

```bash
uv sync --all-packages
uv run fetch-sentiment-data
uv run --env-file .env fetch-sentiment-data
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

Fetch options:

```bash
uv run fetch-sentiment-data --force
uv run fetch-sentiment-data --revision main
uv run fetch-sentiment-data --output-dir /tmp/yelp-raw
```

## Training and evaluation

We use a reduced stratified hold-out split: **200,000** training rows,
**20,000** validation rows, and the full **50,000**-row test split. Ray Tune
`ASHAScheduler` tunes the models with validation macro F1 as the objective:
20 trials for the logistic baseline and 8 trials for BERT. Final models are
refit on the reduced train + validation frame; the test split is untouched
until evaluation.

### Final test results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF logistic regression | 0.58 | 0.58 | 0.58 |
| BERT | 0.68 | 0.67 | 0.67 |

BERT is the stronger model, especially on the 1-star and 5-star classes. The
middle classes remain the hardest.

The selected BERT configuration uses `learning_rate=4.6878e-05`,
`per_device_train_batch_size=16`, `gradient_accumulation_steps=1`,
`weight_decay=0.1`, `warmup_ratio=0.06`, a linear scheduler, label smoothing
`0.1`, dropout `0.2/0.1`, `max_length=256`, and 3 epochs. Exact parameter sets
and trial histories are stored in the artifact directories.

## Deployment export

The training pipeline keeps the Hugging Face checkpoint and exports a
deployment-friendly state-dict bundle:

```text
artifacts/bert/model.pt
artifacts/bert/tokenizer/
artifacts/bert/export_manifest.json
```

The `.pt` bundle stores the model config, state dict, maximum sequence length,
label map, training parameters, and final test metrics. The export command
verifies that the reloaded bundle reproduces the source model probabilities
before it succeeds. `train-sentiment-model` also exports the bundle at the end
of every BERT refit.

## Inference device

`SentimentPredictor` resolves the inference device from `auto`, `cpu`, or
`cuda`:

- `auto` uses CUDA when a usable CUDA-enabled PyTorch runtime and GPU are
  available, otherwise CPU.
- `cpu` forces CPU.
- `cuda` requires a usable CUDA device and raises a clear error instead of
  silently falling back.

The REST API exposes this through `SENTIMENT_DEVICE`, defaulting to `auto`.

## Artifact contract

- `artifacts/logistic/`: TF-IDF logistic best parameters, fitted pipeline,
  tuning trials, and validation metrics.
- `artifacts/bert/model/`: Hugging Face `save_pretrained` source model.
- `artifacts/bert/model.pt`: exported deployment state-dict bundle.
- `artifacts/bert/tokenizer/`: tokenizer files used by the exported model.
- `artifacts/bert/export_manifest.json`: export metadata and test metrics.
- `artifacts/bert/`: best parameters, training history, tuning trials, and
  validation metrics.
- `artifacts/evaluation/`: final test metrics, per-class metrics, confusion
  matrices, hyperparameter table, test predictions, and comparison report.

The API reads the exported bundle through `SENTIMENT_MODEL_PATH`, defaulting to
`models/yelp-sentiment/artifacts/bert/model.pt`, and the tokenizer through
`SENTIMENT_TOKENIZER_PATH`, defaulting to
`models/yelp-sentiment/artifacts/bert/tokenizer`.
