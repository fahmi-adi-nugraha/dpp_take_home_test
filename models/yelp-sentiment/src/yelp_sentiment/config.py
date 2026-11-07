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

CLEANING_VERSION = "1"
FEATURE_VERSION = "1"
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.1

TOKENIZER_NAME = "bert-base-uncased"
MAX_LENGTH_FAST = 256
MAX_LENGTH_FINAL = 512
TRUNCATION = "head"

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
