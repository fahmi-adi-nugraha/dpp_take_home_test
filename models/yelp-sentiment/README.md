# Yelp Sentiment Classification

## Overview

NLP sentiment classification project. The planned model is a BERT-family model
fine-tuned with Hugging Face `transformers`.

## Dataset

- **Preferred source:** <https://huggingface.co/datasets/Yelp/yelp_review_full>
- **Alternative source:** <https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data>
- **Raw data location:** `data/raw/`
- **Fetch command:** `uv run --env-file .env fetch-sentiment-data`
- **Credentials:** `HF_TOKEN` only if required by the chosen source.

The final source and sentiment label mapping must be confirmed during EDA.

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
uv run marimo edit models/yelp-sentiment/notebooks/01_eda.py
uv run --env-file .env fetch-sentiment-data
uv run train-sentiment-model
uv run pytest models/yelp-sentiment/tests
```

## Artifact contract

- `artifacts/model/`: Hugging Face `save_pretrained` output plus tokenizer
  files.
- `artifacts/metrics.json`: evaluation metrics.
- `artifacts/label_map.json`: class index to label mapping.

The API reads the model through `SENTIMENT_MODEL_PATH`, defaulting to
`models/yelp-sentiment/artifacts/model`.

## Status

Scaffold only. No implementation yet.
