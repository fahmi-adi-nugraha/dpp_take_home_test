"""Tests for Yelp evaluation helpers."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from yelp_sentiment import evaluate


def test_classification_metrics_perfect_predictions() -> None:
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]

    metrics = evaluate.classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0


def test_per_class_metrics() -> None:
    table = evaluate.per_class_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
    )

    assert table.height == 5
    assert table.filter(pl.col("label") == 0).row(0, named=True)["recall"] == 0.5
    assert table.filter(pl.col("label") == 1).row(0, named=True)[
        "precision"
    ] == pytest.approx(2 / 3)


def test_confusion_matrix_table_is_long_form() -> None:
    table = evaluate.confusion_matrix_table(
        [0, 1, 1],
        [0, 2, 1],
    )

    assert table.height == 25
    assert (
        table.filter((pl.col("true_label") == 1) & (pl.col("predicted_label") == 1)).row(
            0, named=True
        )["count"]
        == 1
    )
    assert (
        table.filter((pl.col("true_label") == 1) & (pl.col("predicted_label") == 2)).row(
            0, named=True
        )["count"]
        == 1
    )


def test_hyperparameters_table_serializes_nested_values() -> None:
    table = evaluate._hyperparameters_table({"bert": {"learning_rate": 2e-5, "choices": [1, 2]}})

    assert table.height == 2
    choices = table.filter(pl.col("parameter") == "choices").row(0, named=True)
    assert json.loads(choices["value"]) == [1, 2]


def test_predict_logistic_uses_probabilities() -> None:
    class FakeModel:
        def predict_proba(self, texts):
            return np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])

    predictions, probabilities = evaluate.predict_logistic(FakeModel(), ["a", "b"])

    assert predictions.tolist() == [2, 0]
    assert probabilities.shape == (2, 3)
