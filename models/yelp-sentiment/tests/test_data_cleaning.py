"""Tests for Yelp data cleaning and artifact writing."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from yelp_sentiment import data


def make_raw() -> tuple[pl.DataFrame, pl.DataFrame]:
    train = pl.DataFrame(
        {
            "label": [0, 0, 1, 2, 3, 4],
            "text": [
                r"dup\nreview",
                r"dup\nreview",
                r"conflict\nreview",
                r"conflict\nreview",
                r"overlap\nreview",
                "unique review",
            ],
        }
    )
    test = pl.DataFrame({"label": [3], "text": [r"overlap\nreview"]})
    return train, test


def test_build_clean_dataset_applies_agreed_rules() -> None:
    raw_train, raw_test = make_raw()

    result = data.build_clean_dataset(raw_train, raw_test)

    assert result.train.height == 2
    assert result.test.height == 1
    assert result.train.columns == ["review_id", "split", "label", "text"]
    assert result.train["text"].to_list() == ["dup review", "unique review"]
    assert result.train["review_id"].to_list() == ["train-000000", "train-000005"]
    assert result.test["review_id"].to_list() == ["test-000000"]
    assert result.stats["duplicate_groups"] == 1
    assert result.stats["duplicate_rows_removed"] == 1
    assert result.stats["conflicting_groups"] == 1
    assert result.stats["conflict_rows_removed"] == 2
    assert result.stats["overlap_texts"] == 1
    assert result.stats["overlap_rows_removed"] == 1
    assert result.stats["clean_train_rows"] == 2
    assert result.stats["clean_test_rows"] == 1


def test_build_clean_dataset_rejects_unknown_labels() -> None:
    train = pl.DataFrame({"label": [0, 5], "text": ["a", "b"]})
    test = pl.DataFrame({"label": [0], "text": ["c"]})

    with pytest.raises(data.DataError, match="unexpected labels"):
        data.build_clean_dataset(train, test)


def test_near_duplicate_stats() -> None:
    frame = pl.DataFrame(
        {
            "review_id": ["train-1", "train-2", "train-3"],
            "split": ["train", "train", "train"],
            "label": [4, 4, 0],
            "text": ["Hello, world!", "hello world", "different"],
        }
    )

    stats = data.near_duplicate_stats(frame)

    assert stats == {"groups": 1, "rows": 2, "max_group_size": 2}


def test_write_clean_artifacts(tmp_path: Path) -> None:
    raw_train, raw_test = make_raw()
    result = data.build_clean_dataset(raw_train, raw_test)
    parquet_path = tmp_path / "clean.parquet"
    manifest_path = tmp_path / "clean.json"

    artifacts = data.write_clean_artifacts(
        result,
        parquet_path=parquet_path,
        manifest_path=manifest_path,
    )

    assert artifacts.parquet_path == parquet_path
    assert artifacts.manifest_path == manifest_path
    assert parquet_path.is_file()
    assert manifest_path.is_file()
    frame = pl.read_parquet(parquet_path)
    assert frame.height == 3
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["cleaning_version"] == "1"
    assert manifest["stats"]["clean_train_rows"] == 2
    assert manifest["output"]["rows"] == 3


def test_load_clean_data_missing(tmp_path: Path) -> None:
    with pytest.raises(data.DataError, match="not found"):
        data.load_clean_data(tmp_path / "missing.parquet")
