"""Feature and split preparation for Yelp sentiment training."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from yelp_sentiment import config
from yelp_sentiment.data import sha256_file


class FeatureError(RuntimeError):
    """Raised when features cannot be built or validated."""


@dataclass(frozen=True)
class FeatureArtifacts:
    """Paths and manifest produced by feature preparation."""

    parquet_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def validate_clean_frame(frame: pl.DataFrame) -> None:
    """Validate the cleaned frame before assigning model splits."""
    required = (
        config.REVIEW_ID_COLUMN,
        config.SPLIT_COLUMN,
        config.TARGET_COLUMN,
        config.TEXT_COLUMN,
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FeatureError(f"Missing required columns: {', '.join(missing)}")
    if frame[config.REVIEW_ID_COLUMN].n_unique() != frame.height:
        raise FeatureError("review_id must be unique.")
    if frame[config.TEXT_COLUMN].null_count():
        raise FeatureError("text contains null values.")
    if frame[config.TARGET_COLUMN].null_count():
        raise FeatureError("label contains null values.")
    labels = set(frame[config.TARGET_COLUMN].unique().to_list())
    unexpected = sorted(label for label in labels if label not in config.LABEL_MAP)
    if unexpected:
        raise FeatureError(f"Unexpected labels: {unexpected}")
    splits = set(frame[config.SPLIT_COLUMN].unique().to_list())
    if not splits <= {"train", "test"}:
        raise FeatureError(f"Unexpected input splits: {sorted(splits)}")


def stratified_sample(
    frame: pl.DataFrame,
    n_rows: int,
    *,
    seed: int = config.RANDOM_SEED,
) -> pl.DataFrame:
    """Take a class-stratified sample, or return the frame when small enough."""
    if n_rows >= frame.height:
        return frame
    labels = frame[config.TARGET_COLUMN].to_numpy()
    indices = np.arange(frame.height)
    sampled_indices, _ = train_test_split(
        indices,
        train_size=n_rows,
        random_state=seed,
        stratify=labels,
    )
    return frame[np.sort(sampled_indices)]


def build_reduced_feature_frame(
    full_frame: pl.DataFrame,
    *,
    train_rows: int = config.MODEL_TRAIN_ROWS,
    validation_rows: int = config.MODEL_VALIDATION_ROWS,
    seed: int = config.RANDOM_SEED,
) -> pl.DataFrame:
    """Reduce train and validation while preserving the full test split."""
    train = stratified_sample(
        full_frame.filter(pl.col(config.SPLIT_COLUMN) == "train"),
        train_rows,
        seed=seed,
    )
    validation = stratified_sample(
        full_frame.filter(pl.col(config.SPLIT_COLUMN) == "validation"),
        validation_rows,
        seed=seed + 1,
    )
    test = full_frame.filter(pl.col(config.SPLIT_COLUMN) == "test")
    return pl.concat([train, validation, test])


def assign_splits(
    clean_frame: pl.DataFrame,
    *,
    validation_fraction: float = config.VALIDATION_FRACTION,
    seed: int = config.RANDOM_SEED,
) -> pl.DataFrame:
    """Assign a class-stratified validation split without touching test."""
    validate_clean_frame(clean_frame)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    train = clean_frame.filter(pl.col(config.SPLIT_COLUMN) == "train")
    test = clean_frame.filter(pl.col(config.SPLIT_COLUMN) == "test")
    if train.height == 0 or test.height == 0:
        raise FeatureError("Both train and test rows are required.")

    labels = train[config.TARGET_COLUMN].to_numpy()
    indices = np.arange(train.height)
    train_indices, _ = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=labels,
    )
    split_values = np.full(train.height, "validation", dtype=object)
    split_values[train_indices] = "train"
    train_with_validation = train.with_columns(pl.Series(config.SPLIT_COLUMN, split_values))
    return pl.concat([train_with_validation, test])


def split_counts(frame: pl.DataFrame) -> dict[str, int]:
    """Return row counts by split."""
    counts = frame.group_by(config.SPLIT_COLUMN).len().sort(config.SPLIT_COLUMN)
    return {str(row[config.SPLIT_COLUMN]): int(row["len"]) for row in counts.iter_rows(named=True)}


def label_counts_by_split(frame: pl.DataFrame) -> dict[str, dict[int, int]]:
    """Return label counts for each split."""
    counts = (
        frame.group_by(config.SPLIT_COLUMN, config.TARGET_COLUMN)
        .len()
        .sort(config.SPLIT_COLUMN, config.TARGET_COLUMN)
    )
    result: dict[str, dict[int, int]] = {}
    for row in counts.iter_rows(named=True):
        split = str(row[config.SPLIT_COLUMN])
        result.setdefault(split, {})[int(row[config.TARGET_COLUMN])] = int(row["len"])
    return result


def leakage_summary(frame: pl.DataFrame) -> dict[str, int]:
    """Check exact text overlap between held-out test and non-test rows."""
    test_texts = (
        frame.filter(pl.col(config.SPLIT_COLUMN) == "test").select(config.TEXT_COLUMN).unique()
    )
    non_test_texts = (
        frame.filter(pl.col(config.SPLIT_COLUMN) != "test").select(config.TEXT_COLUMN).unique()
    )
    overlap = non_test_texts.join(test_texts, on=config.TEXT_COLUMN, how="inner")
    return {"test_overlap_texts": overlap.height}


def tokenizer_config() -> dict[str, Any]:
    """Return the tokenizer and sequence-length contract."""
    return {
        "name": config.TOKENIZER_NAME,
        "max_length_fast": config.MAX_LENGTH_FAST,
        "max_length_final": config.MAX_LENGTH_FINAL,
        "truncation": config.TRUNCATION,
    }


def token_lengths(texts: Sequence[str], tokenizer) -> list[int]:
    """Return tokenizer lengths for a text sample."""
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]


def token_length_stats_from_lengths(lengths: Sequence[int]) -> dict[str, float | int]:
    """Summarize token lengths."""
    lengths_array = np.asarray(lengths, dtype=int)
    if lengths_array.size == 0:
        raise FeatureError("Token length statistics require at least one review.")
    return {
        "rows": int(lengths_array.size),
        "min": int(lengths_array.min()),
        "p25": float(np.percentile(lengths_array, 25)),
        "median": float(np.percentile(lengths_array, 50)),
        "mean": float(lengths_array.mean()),
        "p75": float(np.percentile(lengths_array, 75)),
        "p90": float(np.percentile(lengths_array, 90)),
        "p95": float(np.percentile(lengths_array, 95)),
        "p99": float(np.percentile(lengths_array, 99)),
        "max": int(lengths_array.max()),
        "over_128": int((lengths_array > 128).sum()),
        "over_256": int((lengths_array > 256).sum()),
        "over_512": int((lengths_array > 512).sum()),
        "share_over_128_pct": float((lengths_array > 128).mean() * 100),
        "share_over_256_pct": float((lengths_array > 256).mean() * 100),
        "share_over_512_pct": float((lengths_array > 512).mean() * 100),
    }


def token_length_stats(
    texts: Sequence[str],
    tokenizer,
) -> dict[str, float | int]:
    """Measure token lengths for a text sample."""
    return token_length_stats_from_lengths(token_lengths(texts, tokenizer))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(f"{path.name}.part")
    part_path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(part_path, path)


def write_feature_artifacts(
    frame: pl.DataFrame,
    *,
    token_stats: dict[str, Any] | None = None,
    validation_fraction: float = config.VALIDATION_FRACTION,
    seed: int = config.RANDOM_SEED,
    parquet_path: Path = config.FEATURES_DATA_PARQUET,
    manifest_path: Path = config.FEATURES_MANIFEST_PATH,
    cleaning_manifest_path: Path = config.CLEAN_MANIFEST_PATH,
    source_feature_manifest_path: Path | None = None,
) -> FeatureArtifacts:
    """Write the model-ready feature frame and manifest."""
    expected_splits = {"train", "validation", "test"}
    actual_splits = set(frame[config.SPLIT_COLUMN].unique().to_list())
    if actual_splits != expected_splits:
        raise FeatureError(
            f"Expected splits {sorted(expected_splits)}, got {sorted(actual_splits)}."
        )
    if frame.select(config.TEXT_COLUMN).unique().height != frame.height:
        raise FeatureError("Feature frame contains duplicate review texts.")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = parquet_path.with_name(f"{parquet_path.name}.part")
    frame.write_parquet(part_path)
    os.replace(part_path, parquet_path)

    cleaning_manifest_sha = (
        sha256_file(cleaning_manifest_path) if cleaning_manifest_path.exists() else None
    )
    source_feature_manifest_sha = (
        sha256_file(source_feature_manifest_path)
        if source_feature_manifest_path is not None and source_feature_manifest_path.exists()
        else None
    )
    manifest: dict[str, Any] = {
        "dataset": config.DATASET_ID,
        "source_revision": config.DATASET_REVISION,
        "feature_version": config.FEATURE_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "label_map": {str(key): value for key, value in sorted(config.LABEL_MAP.items())},
        "cleaning_manifest_sha256": cleaning_manifest_sha,
        "source_feature_manifest_sha256": source_feature_manifest_sha,
        "output": {
            "path": _project_relative(parquet_path),
            "rows": frame.height,
            "columns": frame.columns,
        },
        "split": {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "counts": split_counts(frame),
            "label_counts": label_counts_by_split(frame),
        },
        "tokenizer": tokenizer_config(),
        "token_length_stats": token_stats,
        "leakage": leakage_summary(frame),
    }
    _write_json(manifest_path, manifest)
    return FeatureArtifacts(
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def load_features(path: Path = config.FEATURES_DATA_PARQUET) -> pl.DataFrame:
    """Load the full model-ready feature frame."""
    if not path.is_file():
        raise FeatureError(f"Feature dataset not found: {path}. Run the feature notebook first.")
    return pl.read_parquet(path)


def load_model_features(path: Path = config.MODEL_FEATURES_PARQUET) -> pl.DataFrame:
    """Load the reduced feature frame used for model training."""
    if not path.is_file():
        raise FeatureError(
            f"Reduced feature dataset not found: {path}. Run the feature notebook first."
        )
    return pl.read_parquet(path)
