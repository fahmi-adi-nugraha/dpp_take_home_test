"""Tests for data splitting, pipelines, and ablation selection."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest
from delivery_time import features
from delivery_time.train import (
    AblationSpec,
    build_xy,
    make_linear_pipeline,
    make_xgboost_pipeline,
    select_best_ablation,
    split_dataset,
    validate_feature_names,
)


def make_frame(n_rows: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    order_hour = rng.integers(8, 23, n_rows)
    multiple = rng.integers(0, 4, n_rows)
    return pl.DataFrame(
        {
            "id": [f"id-{index}" for index in range(n_rows)],
            "courier_id": [f"courier-{index % 12}" for index in range(n_rows)],
            "order_date": [date(2022, 1, 1)] * n_rows,
            "distance_km": rng.uniform(1.0, 20.0, n_rows),
            "coordinates_invalid": rng.integers(0, 2, n_rows).astype(bool),
            "traffic_ordinal": rng.integers(0, 4, n_rows),
            "weather": rng.choice(["Sunny", "Fog", "Cloudy", "Stormy"], n_rows).tolist(),
            "city": rng.choice(["Metropolitan", "Urban", "Semi-Urban"], n_rows).tolist(),
            "festival_binary": rng.integers(0, 2, n_rows),
            "multiple_deliveries": multiple,
            "has_multiple_deliveries": multiple > 0,
            "vehicle_condition": rng.integers(0, 4, n_rows),
            "age": rng.uniform(18.0, 50.0, n_rows),
            "rating": rng.uniform(1.0, 5.0, n_rows),
            "order_hour": order_hour,
            "is_rush_hour": (order_hour >= 17) & (order_hour < 21),
            "is_meal_period": order_hour < 22,
            "time_order_missing": rng.integers(0, 2, n_rows).astype(bool),
            "city_missing": rng.integers(0, 2, n_rows).astype(bool),
            "multiple_missing": rng.integers(0, 2, n_rows).astype(bool),
            "target": rng.uniform(10.0, 54.0, n_rows),
        }
    )


def test_split_dataset_sizes_and_disjointness() -> None:
    frame = make_frame(100)
    split = split_dataset(frame)

    assert split.train.height == 80
    assert split.validation.height == 10
    assert split.test.height == 10
    assert split.train["id"].n_unique() == 80
    assert split.validation["id"].n_unique() == 10
    assert split.test["id"].n_unique() == 10
    assert set(split.train["id"]).isdisjoint(split.validation["id"])
    assert set(split.train["id"]).isdisjoint(split.test["id"])
    assert set(split.validation["id"]).isdisjoint(split.test["id"])


def test_split_dataset_is_reproducible() -> None:
    frame = make_frame(20)
    first = split_dataset(frame, seed=123)
    second = split_dataset(frame, seed=123)

    assert first.train["id"].to_list() == second.train["id"].to_list()
    assert first.validation["id"].to_list() == second.validation["id"].to_list()
    assert first.test["id"].to_list() == second.test["id"].to_list()


def test_build_xy_and_pipelines() -> None:
    frame = make_frame(120)
    feature_names = features.BASELINE_FEATURES
    x_train, y_train = build_xy(frame, feature_names)

    assert x_train.shape == (120, len(feature_names))
    assert y_train.shape == (120,)

    ridge = make_linear_pipeline(feature_names).fit(x_train, y_train)
    assert ridge.predict(x_train[:5]).shape == (5,)

    xgboost = make_xgboost_pipeline(
        feature_names,
        {"n_estimators": 5, "n_jobs": 1},
    ).fit(x_train, y_train)
    assert xgboost.predict(x_train[:5]).shape == (5,)


def test_validate_feature_names_rejects_target_and_leakage() -> None:
    with pytest.raises(ValueError, match="Target"):
        validate_feature_names(("target",))

    with pytest.raises(ValueError, match="Excluded"):
        validate_feature_names(("prep_minutes",))


def test_select_best_ablation_prefers_baseline_for_tiny_gain() -> None:
    specs = (
        AblationSpec("baseline", ("distance_km",), "baseline"),
        AblationSpec("plus", ("distance_km", "weather"), "plus"),
    )
    table = pl.DataFrame(
        {
            "name": ["baseline", "plus"],
            "rmse": [10.0, 9.995],
            "eligible_for_final": [True, True],
        }
    )

    assert select_best_ablation(table, specs).name == "baseline"


def test_select_best_ablation_selects_real_improvement() -> None:
    specs = (
        AblationSpec("baseline", ("distance_km",), "baseline"),
        AblationSpec("plus", ("distance_km", "weather"), "plus"),
    )
    table = pl.DataFrame(
        {
            "name": ["baseline", "plus"],
            "rmse": [10.0, 9.5],
            "eligible_for_final": [True, True],
        }
    )

    assert select_best_ablation(table, specs).name == "plus"
