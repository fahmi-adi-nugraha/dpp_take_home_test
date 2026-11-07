"""Configuration constants for the Yelp sentiment project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "yelp_review_full"
RAW_TRAIN_PATH = RAW_DATA_DIR / "train-00000-of-00001.parquet"
RAW_TEST_PATH = RAW_DATA_DIR / "test-00000-of-00001.parquet"
MANIFEST_FILENAME = "fetch-manifest.json"
FETCH_MANIFEST_PATH = RAW_DATA_DIR / MANIFEST_FILENAME

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_DATA_PARQUET = PROCESSED_DATA_DIR / "yelp_clean.parquet"
CLEAN_MANIFEST_PATH = PROCESSED_DATA_DIR / "yelp_clean_manifest.json"
FEATURES_DATA_PARQUET = PROCESSED_DATA_DIR / "yelp_features.parquet"
FEATURES_MANIFEST_PATH = PROCESSED_DATA_DIR / "yelp_features_manifest.json"
MODEL_FEATURES_PARQUET = PROCESSED_DATA_DIR / "yelp_features_model.parquet"
MODEL_FEATURES_MANIFEST_PATH = PROCESSED_DATA_DIR / "yelp_features_model_manifest.json"

CLEANING_VERSION = "1"
FEATURE_VERSION = "1"
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.1

TOKENIZER_NAME = "bert-base-uncased"
MAX_LENGTH_FAST = 256
MAX_LENGTH_FINAL = 512
TRUNCATION = "head"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGISTIC_DIR = ARTIFACTS_DIR / "logistic"
LOGISTIC_MODEL_PATH = LOGISTIC_DIR / "model.joblib"
LOGISTIC_BEST_PARAMS_PATH = LOGISTIC_DIR / "best_params.json"
LOGISTIC_VALIDATION_METRICS_PATH = LOGISTIC_DIR / "validation_metrics.json"
LOGISTIC_TUNING_TRIALS_PATH = LOGISTIC_DIR / "tuning_trials.parquet"

BERT_DIR = ARTIFACTS_DIR / "bert"
BERT_MODEL_DIR = BERT_DIR / "model"
BERT_BEST_PARAMS_PATH = BERT_DIR / "best_params.json"
BERT_VALIDATION_METRICS_PATH = BERT_DIR / "validation_metrics.json"
BERT_TUNING_TRIALS_PATH = BERT_DIR / "tuning_trials.parquet"
BERT_TRAINING_HISTORY_PATH = BERT_DIR / "training_history.json"

EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"
EVALUATION_METRICS_PATH = EVALUATION_DIR / "metrics.json"
EVALUATION_METRICS_TABLE_PATH = EVALUATION_DIR / "metrics_table.parquet"
EVALUATION_HYPERPARAMETERS_PATH = EVALUATION_DIR / "hyperparameters.parquet"
EVALUATION_PER_CLASS_PATH = EVALUATION_DIR / "per_class_metrics.parquet"
EVALUATION_CONFUSION_PATH = EVALUATION_DIR / "confusion_matrices.parquet"
EVALUATION_PREDICTIONS_PATH = EVALUATION_DIR / "test_predictions.parquet"
EVALUATION_COMPARISON_PATH = EVALUATION_DIR / "model_comparison.md"

ASHA_SEED = RANDOM_SEED
ASHA_MAX_T = 3
ASHA_GRACE_PERIOD = 1
ASHA_REDUCTION_FACTOR = 3
LOGISTIC_TRIALS = 20
BERT_TRIALS = 8
LOGISTIC_SEARCH_TRAIN_ROWS = 100_000
BERT_SEARCH_TRAIN_ROWS = 30_000
BERT_SEARCH_VALIDATION_ROWS = 3_000
BERT_SEARCH_MODEL_NAME = "bert-base-uncased"
BERT_TRAIN_BATCH_SIZES = (8, 16)
BERT_EVAL_BATCH_SIZE = 16
BERT_REDUCED_MAX_LENGTH = 256
BERT_REDUCED_EPOCHS = 3

MODEL_TRAIN_ROWS = 200_000
MODEL_VALIDATION_ROWS = 20_000
MODEL_TEST_ROWS = 50_000

DATASET_ID = "Yelp/yelp_review_full"
DATASET_URL = "https://huggingface.co/datasets/Yelp/yelp_review_full"
DATASET_REVISION = "c1f9ee939b7d05667af864ee1cb066393154bf85"
DATASET_LICENSE = "other (Yelp license)"

TARGET_COLUMN = "label"
TEXT_COLUMN = "text"
SPLIT_COLUMN = "split"
REVIEW_ID_COLUMN = "review_id"
EXPECTED_COLUMNS = ("label", "text")

LABEL_MAP = {
    0: "1 star",
    1: "2 star",
    2: "3 stars",
    3: "4 stars",
    4: "5 stars",
}

DEFAULT_TIMEOUT_SECONDS = 300.0
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "dpp-yelp-sentiment/0.1.0"
