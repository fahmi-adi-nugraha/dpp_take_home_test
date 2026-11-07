"""Data loading, cleaning, and artifact helpers for Yelp reviews."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from yelp_sentiment import config
from yelp_sentiment.preprocess import clean_text_series, normalize_for_matching

logger = logging.getLogger(__name__)

_INTEGER_DTYPES = (pl.Int8, pl.Int16, pl.Int32, pl.Int64)


class DataError(RuntimeError):
    """Raised when the dataset cannot be loaded, cleaned, or validated."""


@dataclass(frozen=True)
class CleaningResult:
    """Cleaned train/test frames and cleaning statistics."""

    train: pl.DataFrame
    test: pl.DataFrame
    stats: dict[str, Any]


@dataclass(frozen=True)
class CleanArtifacts:
    """Paths and manifest produced by the cleaning step."""

    parquet_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(config.CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def load_raw_split(split: str) -> pl.DataFrame:
    """Load one raw parquet split."""
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split!r}")
    path = config.RAW_TRAIN_PATH if split == "train" else config.RAW_TEST_PATH
    if not path.is_file():
        raise DataError(f"Raw {split} file not found: {path}")
    return pl.read_parquet(path)


def load_raw_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the raw train and test splits."""
    return load_raw_split("train"), load_raw_split("test")


def validate_raw_frame(frame: pl.DataFrame, name: str) -> None:
    """Validate the raw schema, target domain, and required values."""
    missing = [column for column in config.EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataError(f"{name} is missing required columns: {', '.join(missing)}")

    extra = [column for column in frame.columns if column not in config.EXPECTED_COLUMNS]
    if extra:
        logger.warning("%s has unexpected columns: %s", name, ", ".join(extra))

    target_dtype = frame.schema[config.TARGET_COLUMN]
    if target_dtype not in _INTEGER_DTYPES:
        raise DataError(f"{name} target must be an integer, found {target_dtype}.")
    if frame.schema[config.TEXT_COLUMN] != pl.String:
        raise DataError(f"{name} text must be a string.")

    if frame[config.TARGET_COLUMN].null_count():
        raise DataError(f"{name} target contains null values.")
    if frame[config.TEXT_COLUMN].null_count():
        raise DataError(f"{name} text contains null values.")

    labels = set(frame[config.TARGET_COLUMN].unique().to_list())
    unexpected = sorted(label for label in labels if label not in config.LABEL_MAP)
    if unexpected:
        raise DataError(f"{name} contains unexpected labels: {unexpected}")


def add_review_ids(frame: pl.DataFrame, split: str) -> pl.DataFrame:
    """Add deterministic review identifiers and a split column."""
    review_ids = pl.Series(
        config.REVIEW_ID_COLUMN,
        [f"{split}-{index:06d}" for index in range(frame.height)],
    )
    return frame.with_columns(
        review_ids,
        pl.lit(split).alias(config.SPLIT_COLUMN),
    )


def clean_frame(frame: pl.DataFrame, split: str) -> pl.DataFrame:
    """Decode, normalize, and identify one raw split."""
    validate_raw_frame(frame, split)
    identified = add_review_ids(frame, split)
    return identified.select(
        config.REVIEW_ID_COLUMN,
        config.SPLIT_COLUMN,
        config.TARGET_COLUMN,
        clean_text_series(identified[config.TEXT_COLUMN]).alias(config.TEXT_COLUMN),
    )


def _label_counts(frame: pl.DataFrame) -> dict[int, int]:
    counts = frame.group_by(config.TARGET_COLUMN).len().sort(config.TARGET_COLUMN)
    return {int(row[config.TARGET_COLUMN]): int(row["len"]) for row in counts.iter_rows(named=True)}


def build_clean_dataset(
    raw_train: pl.DataFrame | None = None,
    raw_test: pl.DataFrame | None = None,
) -> CleaningResult:
    """Decode, normalize, deduplicate, and remove train/test overlap."""
    if raw_train is None or raw_test is None:
        raw_train, raw_test = load_raw_data()

    train = clean_frame(raw_train, "train")
    test = clean_frame(raw_test, "test")
    labels_before = _label_counts(train)

    duplicate_stats = train.group_by(config.TEXT_COLUMN).agg(
        pl.len().alias("n"),
        pl.col(config.TARGET_COLUMN).n_unique().alias("n_labels"),
    )
    conflicting = duplicate_stats.filter((pl.col("n") > 1) & (pl.col("n_labels") > 1))
    non_conflicting = duplicate_stats.filter((pl.col("n") > 1) & (pl.col("n_labels") == 1))
    conflicting_texts = conflicting.select(config.TEXT_COLUMN)
    train_without_conflicts = train.join(
        conflicting_texts,
        on=config.TEXT_COLUMN,
        how="anti",
    )
    conflict_rows_removed = train.height - train_without_conflicts.height

    train_deduplicated = train_without_conflicts.unique(
        subset=[config.TEXT_COLUMN],
        keep="first",
        maintain_order=True,
    )
    duplicate_rows_removed = train_without_conflicts.height - train_deduplicated.height

    test_texts = test.select(config.TEXT_COLUMN).unique()
    overlap_texts = (
        train_deduplicated.select(config.TEXT_COLUMN)
        .unique()
        .join(
            test_texts,
            on=config.TEXT_COLUMN,
            how="inner",
        )
    )
    train_final = train_deduplicated.join(
        overlap_texts,
        on=config.TEXT_COLUMN,
        how="anti",
    )
    overlap_rows_removed = train_deduplicated.height - train_final.height

    stats: dict[str, Any] = {
        "raw_train_rows": raw_train.height,
        "raw_test_rows": raw_test.height,
        "clean_train_rows": train_final.height,
        "clean_test_rows": test.height,
        "duplicate_groups": non_conflicting.height,
        "duplicate_rows_removed": duplicate_rows_removed,
        "conflicting_groups": conflicting.height,
        "conflict_rows_removed": conflict_rows_removed,
        "overlap_texts": overlap_texts.height,
        "overlap_rows_removed": overlap_rows_removed,
        "empty_texts_train": int((train_final[config.TEXT_COLUMN].str.strip_chars() == "").sum()),
        "empty_texts_test": int((test[config.TEXT_COLUMN].str.strip_chars() == "").sum()),
        "label_counts_before_train": labels_before,
        "label_counts_after_train": _label_counts(train_final),
        "label_counts_test": _label_counts(test),
    }

    return CleaningResult(train=train_final, test=test, stats=stats)


def near_duplicate_groups(frame: pl.DataFrame) -> pl.DataFrame:
    """Return near-duplicate groups after case and punctuation normalization."""
    normalized = frame.with_columns(
        pl.col(config.TEXT_COLUMN)
        .map_elements(normalize_for_matching, return_dtype=pl.String)
        .alias("_normalized")
    )
    return (
        normalized.group_by("_normalized")
        .agg(
            pl.len().alias("n"),
            pl.col(config.TARGET_COLUMN).n_unique().alias("n_labels"),
        )
        .filter(pl.col("n") > 1)
        .sort("n", descending=True)
    )


def near_duplicate_stats(frame: pl.DataFrame) -> dict[str, int]:
    """Summarize near-duplicate normalized texts without changing the data."""
    groups = near_duplicate_groups(frame)
    return {
        "groups": groups.height,
        "rows": int(groups["n"].sum()) if groups.height else 0,
        "max_group_size": int(groups["n"].max()) if groups.height else 0,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(f"{path.name}.part")
    part_path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(part_path, path)


def write_clean_artifacts(
    result: CleaningResult,
    *,
    parquet_path: Path = config.CLEAN_DATA_PARQUET,
    manifest_path: Path = config.CLEAN_MANIFEST_PATH,
) -> CleanArtifacts:
    """Write the cleaned Parquet dataset and its manifest."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined = pl.concat([result.train, result.test])
    part_path = parquet_path.with_name(f"{parquet_path.name}.part")
    combined.write_parquet(part_path)
    os.replace(part_path, parquet_path)

    manifest: dict[str, Any] = {
        "dataset": config.DATASET_ID,
        "source_revision": config.DATASET_REVISION,
        "cleaning_version": config.CLEANING_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "label_map": {str(key): value for key, value in sorted(config.LABEL_MAP.items())},
        "raw": {
            "train": {
                "path": _project_relative(config.RAW_TRAIN_PATH),
                "sha256": (
                    sha256_file(config.RAW_TRAIN_PATH) if config.RAW_TRAIN_PATH.exists() else None
                ),
                "rows": result.stats["raw_train_rows"],
            },
            "test": {
                "path": _project_relative(config.RAW_TEST_PATH),
                "sha256": (
                    sha256_file(config.RAW_TEST_PATH) if config.RAW_TEST_PATH.exists() else None
                ),
                "rows": result.stats["raw_test_rows"],
            },
        },
        "output": {
            "path": _project_relative(parquet_path),
            "rows": combined.height,
            "columns": combined.columns,
        },
        "stats": result.stats,
    }
    _write_json(manifest_path, manifest)
    return CleanArtifacts(
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def load_clean_data(path: Path = config.CLEAN_DATA_PARQUET) -> pl.DataFrame:
    """Load the cleaned dataset."""
    if not path.is_file():
        raise DataError(f"Cleaned dataset not found: {path}. Run the cleaning notebook first.")
    return pl.read_parquet(path)
