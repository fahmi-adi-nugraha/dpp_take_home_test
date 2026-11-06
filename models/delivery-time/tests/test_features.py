"""Tests for feature construction and feature-set definitions."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from delivery_time.data import clean_data
from delivery_time.features import (
    ALL_FEATURES,
    BASELINE_FEATURES,
    EXCLUDED_FEATURES,
    FEATURE_DESCRIPTIONS,
    LEAKAGE_FIELDS,
    LINEAR_EXTRA_FEATURES,
    OPTIONAL_FEATURES,
    build_feature_frame,
    feature_set_table,
    haversine_km,
    write_feature_artifacts,
)

RAW_DEFAULTS = {
    "ID": "0x1",
    "Delivery_person_ID": "DEHRES17DEL01",
    "Delivery_person_Age": "30",
    "Delivery_person_Ratings": "4.7",
    "Restaurant_latitude": "12.9",
    "Restaurant_longitude": "77.5",
    "Delivery_location_latitude": "13.0",
    "Delivery_location_longitude": "77.6",
    "Order_Date": "12-02-2022",
    "Time_Orderd": "12:00",
    "Time_Order_picked": "12:10",
    "Weather_conditions": "Cloudy",
    "Road_traffic_density": "Jam",
    "Vehicle_condition": "0",
    "Type_of_order": "Meal",
    "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": "2",
    "Festival": "Yes",
    "City": "Metropolitian",
    "Time_taken (min)": "30",
}


def make_raw(*records: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame([{**RAW_DEFAULTS, **record} for record in records])


def test_haversine_km() -> None:
    assert haversine_km(0, 0, 0, 0) == pytest.approx(0.0)
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.19, rel=1e-3)


def test_feature_sets_are_disjoint_and_cover_all_features() -> None:
    baseline = set(BASELINE_FEATURES)
    linear = set(LINEAR_EXTRA_FEATURES)
    optional = set(OPTIONAL_FEATURES)

    assert baseline.isdisjoint(linear)
    assert baseline.isdisjoint(optional)
    assert linear.isdisjoint(optional)
    assert baseline | linear | optional == set(ALL_FEATURES)
    assert set(ALL_FEATURES).isdisjoint(EXCLUDED_FEATURES)
    assert set(ALL_FEATURES).isdisjoint(LEAKAGE_FIELDS)
    assert set(FEATURE_DESCRIPTIONS) == set(ALL_FEATURES)


def test_feature_set_table_has_no_unassigned_features() -> None:
    table = feature_set_table()
    assert table.filter(pl.col("sets") == "unassigned").height == 0


def test_build_feature_frame_constructs_expected_features() -> None:
    raw = make_raw(
        {
            "ID": "0x1",
            "Time_taken (min)": "30",
        },
        {
            "ID": "0x2",
            "Delivery_person_Age": "16",
            "Delivery_person_Ratings": "6.0",
            "Restaurant_latitude": "0",
            "Restaurant_longitude": "0",
            "Delivery_location_latitude": "0.05",
            "Delivery_location_longitude": "0.05",
            "Time_Orderd": "NaN",
            "Time_Order_picked": "10:10",
            "Weather_conditions": "NaN",
            "Road_traffic_density": "NaN",
            "multiple_deliveries": "NaN",
            "Festival": "NaN",
            "City": "NaN",
            "Time_taken (min)": "11",
        },
    )
    features = build_feature_frame(clean_data(raw))

    assert features.columns[:4] == ["id", "courier_id", "order_date", "target"]
    assert features.height == 2
    assert features["distance_km"][0] is not None
    assert features["distance_km"][1] is None
    assert features["distance_missing"][1] is True
    assert features["coordinates_invalid"][1] is True
    assert features["traffic_ordinal"][0] == 3
    assert features["traffic_ordinal"][1] is None
    assert features["festival_binary"][0] == 1
    assert features["has_multiple_deliveries"][0] is True
    assert features["vehicle_condition_is_zero"][0] is True
    assert features["order_hour"][0] == 12
    assert features["is_rush_hour"][0] is True
    assert features["is_meal_period"][0] is True
    assert features["age_bin"][0] == 30
    assert features["age_bin"][1] == 15
    assert features["rating_bin"][0] == "high"
    assert features["rating_bin"][1] == "out_of_range"
    assert features["restaurant_grid"][0] == "12.9,77.5"
    assert features["restaurant_grid"][1] is None
    assert "pickup_minutes" not in features.columns
    assert "prep_minutes" not in features.columns
    assert "Time_Order_picked" not in features.columns


def test_write_feature_artifacts(tmp_path: Path) -> None:
    raw = make_raw({"ID": "0x1"}, {"ID": "0x2", "Time_taken (min)": "11"})
    cleaned = clean_data(raw)
    features = build_feature_frame(cleaned)
    clean_path = tmp_path / "clean.parquet"
    cleaned.write_parquet(clean_path)

    manifest = write_feature_artifacts(
        features,
        clean_path=clean_path,
        parquet_path=tmp_path / "features.parquet",
        csv_path=tmp_path / "features.csv",
        manifest_path=tmp_path / "features_manifest.json",
    )

    assert (tmp_path / "features.parquet").is_file()
    assert (tmp_path / "features.csv").is_file()
    assert manifest["baseline_features"] == list(BASELINE_FEATURES)
    assert json.loads((tmp_path / "features_manifest.json").read_text())["rows"] == 2
