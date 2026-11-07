# Yelp Sentiment Classification

## Overview

NLP sentiment classification project. The planned model is a BERT-family model
fine-tuned with Hugging Face `transformers`.

## Dataset

- **Preferred source:** <https://huggingface.co/datasets/Yelp/yelp_review_full>
- **Alternative source:** <https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data>
- **Raw data location:** `data/raw/yelp_review_full/`
- **Fetch command:** `uv run fetch-sentiment-data`
- **Credentials:** not required for the public dataset; set `HF_TOKEN` only if
  rate limits or future access restrictions require it.

The fetcher downloads the two parquet files from the Hugging Face dataset,
pinned to commit `c1f9ee939b7d05667af864ee1cb066393154bf85`, and validates their
checksums, schema, row counts, and labels.

| Split | Rows | Classes |
|---|---:|---:|
| train | 650,000 | 5 balanced classes |
| test | 50,000 | 5 balanced classes |

The raw labels are `0–4`, corresponding to 1–5 stars. The final sentiment label
mapping (five-class or collapsed binary) must be confirmed during EDA. A
`fetch-manifest.json` records file sizes, SHA-256 hashes, row counts, label
counts, and the dataset revision.

## Target and metrics

- **Target:** review sentiment derived from star ratings.
- **Planned metrics:** accuracy, macro F1, and per-class precision/recall.

## Planned workflow

1. Fetch the dataset with `httpx` and store it under `data/raw/`.
2. Explore the data in `notebooks/01_eda.py` using Polars and plotnine.
3. Clean and preprocess the text in `preprocess.py`.
4. Tokenize and fine-tune the model in `train.py` and
   `notebooks/02_finetune.py`.
5. Evaluate the model in `evaluate.py` and `notebooks/03_evaluate.py`.
6. Export the model and tokenizer under `artifacts/` for the API.

## Layout

- `src/yelp_sentiment/`: importable package and CLI entry points.
- `notebooks/`: marimo notebooks for EDA, fine-tuning, and evaluation.
- `data/raw/`, `data/interim/`, `data/processed/`: local dataset stages.
- `artifacts/`: trained model, tokenizer, and metadata; not committed.
- `tests/`: package tests.

## Commands

```bash
uv run fetch-sentiment-data
uv run --env-file .env fetch-sentiment-data
uv run marimo edit models/yelp-sentiment/notebooks/01_eda.py
uv run train-sentiment-model
uv run pytest models/yelp-sentiment/tests
```

Fetch options:

```bash
uv run fetch-sentiment-data --force
uv run fetch-sentiment-data --revision main
uv run fetch-sentiment-data --output-dir /tmp/yelp-raw
```

## Artifact contract

- `artifacts/model/`: Hugging Face `save_pretrained` output plus tokenizer
  files.
- `artifacts/metrics.json`: evaluation metrics.
- `artifacts/label_map.json`: class index to label mapping.

The API reads the model through `SENTIMENT_MODEL_PATH`, defaulting to
`models/yelp-sentiment/artifacts/model`.

## Status

Data fetching and validation are implemented. EDA, preprocessing, fine-tuning,
evaluation, and API integration remain to be implemented.
