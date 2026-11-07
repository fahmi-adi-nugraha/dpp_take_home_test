"""Tests for Yelp split and feature preparation."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from yelp_sentiment import features


def make_clean_frame() -> pl.DataFrame:
    train = pl.DataFrame(
        {
            "review_id": [f"train-{index:06d}" for index in range(100)],
            "split": ["train"] * 100,
            "label": [index % 5 for index in range(100)],
            "text": [f"train review {index}" for index in range(100)],
        }
    )
    test = pl.DataFrame(
        {
            "review_id": [f"test-{index:06d}" for index in range(20)],
            "split": ["test"] * 20,
            "label": [index % 5 for index in range(20)],
            "text": [f"test review {index}" for index in range(20)],
        }
    )
    return pl.concat([train, test])


def test_assign_splits_is_stratified_and_keeps_test() -> None:
    clean = make_clean_frame()

    prepared = features.assign_splits(clean, validation_fraction=0.1, seed=42)

    counts = features.split_counts(prepared)
    assert counts == {"test": 20, "train": 90, "validation": 10}
    test_ids = set(prepared.filter(pl.col("split") == "test")["review_id"].to_list())
    assert test_ids == set(clean.filter(pl.col("split") == "test")["review_id"].to_list())
    label_counts = features.label_counts_by_split(prepared)
    assert set(label_counts["validation"]) == {0, 1, 2, 3, 4}
    assert sum(label_counts["validation"].values()) == 10


def test_build_reduced_feature_frame_keeps_test() -> None:
    prepared = features.assign_splits(make_clean_frame(), validation_fraction=0.1, seed=42)

    reduced = features.build_reduced_feature_frame(
        prepared,
        train_rows=40,
        validation_rows=10,
        seed=42,
    )

    counts = features.split_counts(reduced)
    assert counts == {"test": 20, "train": 40, "validation": 10}
    assert (
        reduced.filter(pl.col("split") == "test")["review_id"].to_list()
        == prepared.filter(pl.col("split") == "test")["review_id"].to_list()
    )


def test_leakage_summary_detects_overlap() -> None:
    frame = pl.DataFrame(
        {
            "review_id": ["train-1", "test-1"],
            "split": ["train", "test"],
            "label": [1, 1],
            "text": ["same text", "same text"],
        }
    )

    assert features.leakage_summary(frame) == {"test_overlap_texts": 1}


def test_token_length_stats_with_fake_tokenizer() -> None:
    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            lengths = [3, 300, 600]
            return {"input_ids": [[0] * length for length in lengths]}

    stats = features.token_length_stats(["a", "b", "c"], FakeTokenizer())

    assert stats["rows"] == 3
    assert stats["min"] == 3
    assert stats["max"] == 600
    assert stats["over_256"] == 2
    assert stats["over_512"] == 1
    assert stats["share_over_512_pct"] == pytest.approx(100 / 3)


def test_write_feature_artifacts(tmp_path: Path) -> None:
    prepared = features.assign_splits(make_clean_frame(), validation_fraction=0.1, seed=42)
    parquet_path = tmp_path / "features.parquet"
    manifest_path = tmp_path / "features.json"

    artifacts = features.write_feature_artifacts(
        prepared,
        token_stats={"rows": 3, "share_over_512_pct": 1.23},
        parquet_path=parquet_path,
        manifest_path=manifest_path,
    )

    assert artifacts.parquet_path.is_file()
    assert artifacts.manifest_path.is_file()
    assert pl.read_parquet(parquet_path).height == 120
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["feature_version"] == "1"
    assert manifest["split"]["counts"] == {
        "test": 20,
        "train": 90,
        "validation": 10,
    }
    assert manifest["tokenizer"]["max_length_fast"] == 256
    assert manifest["tokenizer"]["max_length_final"] == 512
    assert manifest["leakage"]["test_overlap_texts"] == 0


def test_write_feature_artifacts_rejects_duplicate_text(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "review_id": ["train-1", "train-2", "test-1"],
            "split": ["train", "validation", "test"],
            "label": [0, 1, 2],
            "text": ["same", "same", "other"],
        }
    )

    with pytest.raises(features.FeatureError, match="duplicate review texts"):
        features.write_feature_artifacts(
            frame,
            parquet_path=tmp_path / "features.parquet",
            manifest_path=tmp_path / "features.json",
        )
