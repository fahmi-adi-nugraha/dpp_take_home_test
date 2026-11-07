"""Configuration constants for the Yelp sentiment project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "yelp_review_full"
MANIFEST_FILENAME = "fetch-manifest.json"

DATASET_ID = "Yelp/yelp_review_full"
DATASET_URL = "https://huggingface.co/datasets/Yelp/yelp_review_full"
DATASET_REVISION = "c1f9ee939b7d05667af864ee1cb066393154bf85"
DATASET_LICENSE = "other (Yelp license)"

TARGET_COLUMN = "label"
TEXT_COLUMN = "text"
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
