"""Tests for Yelp training utilities."""

from __future__ import annotations

import numpy as np
import polars as pl
from yelp_sentiment import features, train


def test_stratified_sample_preserves_classes() -> None:
    frame = pl.DataFrame(
        {
            "label": [index % 5 for index in range(100)],
            "text": [f"review {index}" for index in range(100)],
        }
    )

    sampled = features.stratified_sample(frame, 30, seed=42)

    assert sampled.height == 30
    counts = sampled.group_by("label").len().sort("label")
    assert counts["len"].to_list() == [6, 6, 6, 6, 6]


def test_logistic_pipeline_fits_and_predicts() -> None:
    params = {
        "tfidf_max_features": 100,
        "tfidf_ngram_range": "unigram",
        "tfidf_min_df": 1,
        "tfidf_max_df": 1.0,
        "tfidf_sublinear_tf": True,
        "clf_C": 1.0,
        "clf_fit_intercept": True,
    }
    texts = ["great", "terrible", "wonderful", "awful", "good", "bad"]
    labels = [4, 0, 4, 0, 3, 1]

    pipeline = train.make_logistic_pipeline(params).fit(texts, labels)

    assert pipeline.predict(["great", "awful"]).shape == (2,)


def test_review_dataset_tokenizes_on_demand() -> None:
    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    dataset = train.ReviewDataset(
        ["first review", "second review"],
        [1, 4],
        FakeTokenizer(),
        max_length=16,
    )

    item = dataset[0]

    assert len(dataset) == 2
    assert item["input_ids"] == [1, 2, 3]
    assert item["labels"] == 1


def test_bert_params_from_config_fixes_sequence_contract() -> None:
    params = train._bert_params_from_config(
        {
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "weight_decay": 0.01,
            "warmup_ratio": 0.06,
            "lr_scheduler_type": "linear",
            "label_smoothing_factor": 0.0,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
        }
    )

    assert params["max_length"] == 256
    assert params["num_train_epochs"] == 3
    assert params["learning_rate"] == 2e-5
    assert np.isclose(params["weight_decay"], 0.01)
