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
- **Data dictionary:** [DATA_DICTIONARY.md](DATA_DICTIONARY.md)
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
target.

## Target and metrics

- **Target:** five-class review sentiment derived from star ratings.
- **Metrics:** accuracy, macro F1, weighted F1, and per-class precision/recall.
- **Primary metric:** macro F1 because each star rating is weighted equally.

## Input variables

The API accepts one review per request.

| Input | Format | Valid range | Notes |
|---|---|---|---|
| `text` | UTF-8 string | 1–10,000 characters | Empty strings are rejected; text is tokenized and truncated to 256 tokens |

The tokenizer converts each review into:

| Model input | Format | Range | Notes |
|---|---|---|---|
| `input_ids` | integer tensor | 1–256 tokens | Includes `[CLS]` and `[SEP]`; longer reviews are truncated |
| `attention_mask` | 0/1 tensor | Same length as `input_ids` | Marks real tokens versus padding |

The model returns five-class probabilities for labels `0–4`, corresponding to
1–5 stars.

## Fetched output

The fetch command stores the two parquet files and a metadata manifest:

```text
data/raw/yelp_review_full/
├── train-00000-of-00001.parquet
├── test-00000-of-00001.parquet
└── fetch-manifest.json
```

The manifest records file sizes, SHA-256 hashes, row counts, label counts, and
the dataset revision.

## Processed outputs

```text
data/processed/
├── yelp_clean.parquet
├── yelp_clean_manifest.json
├── yelp_features.parquet
├── yelp_features_manifest.json
├── yelp_features_model.parquet
└── yelp_features_model_manifest.json
```

Parquet is the canonical typed artifact. The cleaned dataset stores decoded
review text, labels, review IDs, and split assignments. The reduced feature
frame keeps the same text-level data for the 200,000-row train split,
20,000-row validation split, and full 50,000-row test split.

Unlike the delivery-time project, there is no persisted numeric or tokenized
feature matrix. BERT tokenization happens during training and inference, and
the TF-IDF vectorizer is learned inside the saved logistic pipeline. The
processed data artifacts and manifests are generated locally and are ignored
by Git.

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

## Training environment

The training CLI is designed for local use. It reads data from `data/`, writes
artifacts under `artifacts/`, and does not provision or manage remote compute.
Training time depends heavily on the user's GPU; BERT fine-tuning and the ASHA
search can take many hours on a laptop GPU and is generally impractical on CPU.

For practical training, use a cloud GPU rental service; a cloud A100 instead of
a laptop GPU drastically reduces wall-clock time. One option is
[JarvisLabs](https://jarvislabs.ai). On the A100 instance used for this project,
the complete Yelp training, evaluation, and export run took about 1 hour 50
minutes.

### Example JarvisLabs workflow

Install and authenticate the `jl` CLI, then add the SSH key that will be used
for the instance:

```bash
jl status --json
jl ssh-key add ~/.ssh/id_ed25519.pub --name yelp-training
```

Create an instance. Copy the `machine_id` from the JSON output:

```bash
jl create --gpu A100 --storage 50 --name yelp-bert --yes --json
```

Upload the repository to the instance. Use `jl upload` for a quick transfer, or
use `rsync` with excludes for `.git`, `.venv`, local artifacts, and local
datasets. The data fetch command downloads the dataset again on the instance.

```bash
jl upload <machine_id> . /home/dpp_take_home_test
```

Start the full training pipeline as a managed run:

```bash
jl run --on <machine_id> --json --yes -- sh -lc '
  cd /home/dpp_take_home_test &&
  uv sync --all-packages &&
  uv run fetch-sentiment-data &&
  uv run train-sentiment-model &&
  uv run evaluate-sentiment-models &&
  uv run export-sentiment-model
'
```

Monitor the run with the returned `run_id`:

```bash
jl run logs <run_id> --tail 50
jl run status <run_id> --json
```

After the run succeeds, download the artifacts and tokenizer:

```bash
jl download <machine_id> \
  /home/dpp_take_home_test/models/yelp-sentiment/artifacts \
  ./models/yelp-sentiment/artifacts -r
```

Finally, destroy the instance so GPU and disk billing stop:

```bash
jl destroy <machine_id> --yes
```

## Training and evaluation

We use a reduced stratified hold-out split: **200,000** training rows,
**20,000** validation rows, and the full **50,000**-row test split. Ray Tune
`ASHAScheduler` tunes the models with validation macro F1 as the objective:
20 trials for the logistic baseline and 8 trials for BERT. Final models are
refit on the reduced train + validation frame; the test split is untouched
until evaluation.

### Final test results and selected hyperparameters

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF logistic regression | 0.58 | 0.58 | 0.58 |
| BERT | 0.68 | 0.67 | 0.67 |

BERT is the stronger model, especially on the 1-star and 5-star classes. The
middle classes remain the hardest.

The logistic search covered `tfidf_max_features` 10,000–100,000,
`tfidf_ngram_range` unigram and unigram+bigram, `tfidf_min_df` 2–10,
`tfidf_max_df` 0.50–0.90, sublinear term frequency on/off, `clf_C` 0.001–100,
and intercept on/off. The BERT search covered `learning_rate` 1e-5–5e-5,
batch size 8 or 16, gradient accumulation 1, 2, or 4, weight decay 0, 0.01, or
0.10, warmup ratio 0, 0.06, or 0.10, linear or cosine schedules, label
smoothing 0, 0.05, or 0.10, and hidden/attention dropout 0.10 or 0.20. The
sequence length was fixed at 256 and ASHA selected the number of epochs from
1–3.

The final models use the following values. Exact parameter sets and trial
histories are stored in the artifact directories.

| Model | Parameter | Value | Controls | Why we selected it |
|---|---|---:|---|---|
| Logistic | `tfidf_max_features` | 10,000 | Maximum TF-IDF vocabulary size. | A compact vocabulary reduced vectorizer size and overfitting while retaining enough signal. |
| Logistic | `tfidf_ngram_range` | `unigram_bigram` | Word n-grams included in the TF-IDF features. | Bigrams captured negation and short phrases that unigrams missed. |
| Logistic | `tfidf_min_df` | 10 | Minimum document frequency for a term. | Removing rare terms reduced noise and model size. |
| Logistic | `tfidf_max_df` | 0.75 | Maximum document frequency for a term. | Dropping overly common terms reduced uninformative tokens. |
| Logistic | `tfidf_sublinear_tf` | `true` | Sublinear term-frequency scaling. | Dampened the effect of repeated words and improved validation macro F1. |
| Logistic | `clf_C` | 0.2982 | Inverse regularization strength. | Moderate regularization generalized better than weaker or stronger penalties. |
| Logistic | `clf_fit_intercept` | `false` | Whether to fit an intercept. | Validation macro F1 was better without an extra intercept. |
| BERT | `learning_rate` | 4.6878e-05 | Shrinkage applied to each optimizer step. | A low learning rate with three epochs gave the best validation macro F1. |
| BERT | `per_device_train_batch_size` | 16 | Training examples per device batch. | Batch 16 fit A100 memory and produced stable gradients. |
| BERT | `gradient_accumulation_steps` | 1 | Number of batches accumulated per optimizer step. | The selected batch size did not need extra accumulation. |
| BERT | `weight_decay` | 0.1 | AdamW weight regularization. | Regularization reduced overfitting during fine-tuning. |
| BERT | `warmup_ratio` | 0.06 | Fraction of steps used for learning-rate warmup. | A short warmup stabilized early training. |
| BERT | `lr_scheduler_type` | `linear` | Learning-rate decay schedule. | The linear schedule outperformed cosine for this run. |
| BERT | `label_smoothing_factor` | 0.1 | Softens the one-hot training targets. | Smoothing helped with noisy and ambiguous adjacent-star reviews. |
| BERT | `hidden_dropout_prob` | 0.2 | Dropout on hidden representations. | Stronger hidden dropout regularized the classifier. |
| BERT | `attention_probs_dropout_prob` | 0.1 | Dropout on attention probabilities. | Mild attention dropout regularized without disrupting learning. |
| BERT | `max_length` | 256 | Maximum tokens per review. | A 256-token budget balanced coverage, memory, and training speed. |
| BERT | `num_train_epochs` | 3 | Number of fine-tuning epochs. | The third epoch gave the best validation macro F1 and ASHA selected it. |

## Deployment artifacts

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

Artifacts:

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

Prebuilt artifacts, including both project models and evaluation outputs, are
available from the
[`models-v1.0.0` release](https://github.com/fahmi-adi-nugraha/dpp_take_home_test/releases/tag/models-v1.0.0).
Download `dpp-inference-models-v1.zip`, verify it with the published SHA-256
file, and extract it from the repository root.

## API integration

The Yelp model is served by `POST /predict/sentiment` in the local FastAPI app.
The endpoint accepts raw review text, tokenizes it with the exported tokenizer,
runs BERT inference, and returns the predicted label, confidence, and
five-class probabilities. See [`../../apps/api/README.md`](../../apps/api/README.md)
for the request contract and run command.

The API reads the exported bundle through `SENTIMENT_MODEL_PATH`, defaulting to
`models/yelp-sentiment/artifacts/bert/model.pt`, and the tokenizer through
`SENTIMENT_TOKENIZER_PATH`, defaulting to
`models/yelp-sentiment/artifacts/bert/tokenizer`.

### Inference device

`SentimentPredictor` resolves the inference device from `auto`, `cpu`, or
`cuda`:

- `auto` uses CUDA when a usable CUDA-enabled PyTorch runtime and GPU are
  available, otherwise CPU.
- `cpu` forces CPU.
- `cuda` requires a usable CUDA device and raises a clear error instead of
  silently falling back.

The REST API exposes this through `SENTIMENT_DEVICE`, defaulting to `auto`.
