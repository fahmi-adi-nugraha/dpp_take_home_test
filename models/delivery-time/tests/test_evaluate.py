"""Tests for model evaluation helpers."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest
from delivery_time.evaluate import (
    error_by_category,
    prediction_frame,
    regression_metrics,
    residual_summary,
)


def test_regression_metrics() -> None:
    metrics = regression_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 4.0]),
    )

    assert metrics["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert metrics["mae"] == pytest.approx(1 / 3)
    assert metrics["r2"] == pytest.approx(0.5)


def test_prediction_frame_adds_residuals() -> None:
    frame = pl.DataFrame(
        {
            "id": ["a", "b"],
            "order_date": [date(2022, 1, 1), date(2022, 1, 2)],
            "target": [10, 20],
            "distance_km": [1.0, 2.0],
            "coordinates_invalid": [False, True],
            "coord_status": ["valid", "invalid"],
            "traffic_ordinal": [0, 3],
            "weather": ["Sunny", "Fog"],
            "city": ["Urban", "Metropolitan"],
            "festival_binary": [0, 1],
            "multiple_deliveries": [0, 2],
            "order_hour": [9, 20],
            "distance_decile": [0, 9],
        }
    )

    predictions = prediction_frame(
        frame,
        np.array([12.0, 17.0]),
        split_name="validation",
    )

    assert predictions["residual"].to_list() == [2.0, -3.0]
    assert predictions["abs_error"].to_list() == [2.0, 3.0]
    assert predictions["split"].to_list() == ["validation", "validation"]


def test_error_by_category() -> None:
    predictions = pl.DataFrame(
        {
            "city": ["Urban", "Urban", "Metropolitan"],
            "target": [10.0, 20.0, 30.0],
            "prediction": [12.0, 18.0, 27.0],
            "residual": [2.0, -2.0, -3.0],
            "abs_error": [2.0, 2.0, 3.0],
        }
    )

    table = error_by_category(predictions, "city")

    urban = table.filter(pl.col("city") == "Urban").row(0, named=True)
    assert urban["n"] == 2
    assert urban["mae"] == pytest.approx(2.0)
    assert urban["rmse"] == pytest.approx(2.0)


def test_residual_summary() -> None:
    predictions = pl.DataFrame(
        {
            "residual": [1.0, -1.0],
            "abs_error": [1.0, 1.0],
        }
    )

    summary = residual_summary(predictions).row(0, named=True)

    assert summary["n"] == 2
    assert summary["mae"] == pytest.approx(1.0)
    assert summary["rmse"] == pytest.approx(1.0)
