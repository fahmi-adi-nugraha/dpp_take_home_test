"""Tests for data loading and cleaning."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from delivery_time.data import (
    classify_time_format,
    clean_data,
    parse_order_date,
    parse_time_minutes,
    write_clean_artifacts,
)

RAW_DEFAULTS = {
    "ID": "0x1",
    "Delivery_person_ID": "DEHRES17DEL01",
    "Delivery_person_Age": "30",
    "Delivery_person_Ratings": "4.5",
    "Restaurant_latitude": "12.9",
    "Restaurant_longitude": "77.5",
    "Delivery_location_latitude": "13.0",
    "Delivery_location_longitude": "77.6",
    "Order_Date": "12-02-2022",
    "Time_Orderd": "10:00",
    "Time_Order_picked": "10:10",
    "Weather_conditions": "Sunny",
    "Road_traffic_density": "Low",
    "Vehicle_condition": "2",
    "Type_of_order": "Meal",
    "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": "1",
    "Festival": "No",
    "City": "Metropolitian",
    "Time_taken (min)": "20",
}


def make_raw(*records: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame([{**RAW_DEFAULTS, **record} for record in records])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("21:55", 1315),
        ("0.833333333", 1200),
        ("1", 1440),
        ("24:05:00", 1445),
        ("NaN", None),
        ("", None),
    ],
)
def test_parse_time_minutes(raw: str, expected: int | None) -> None:
    assert parse_time_minutes(raw) == expected


def test_parse_order_date() -> None:
    assert parse_order_date("12-02-2022") == date(2022, 2, 12)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("21:55", "HH:MM"),
        ("24:05:00", "HH:MM:SS"),
        ("0.833333333", "fractional day"),
        ("1", "1 (24:00)"),
        ("NaN", "missing"),
    ],
)
def test_classify_time_format(raw: str, expected: str) -> None:
    assert classify_time_format(raw) == expected


def test_clean_data_normalizes_city_and_flags_values() -> None:
    raw = make_raw(
        {
            "Delivery_person_Age": "15",
            "Delivery_person_Ratings": "6.0",
            "multiple_deliveries": "NaN",
            "City": "Metropolitian",
            "Weather_conditions": "NaN",
            "Road_traffic_density": "NaN",
            "Time_Orderd": "NaN",
            "Restaurant_latitude": "0",
            "Restaurant_longitude": "0",
            "Delivery_location_latitude": "0.05",
            "Delivery_location_longitude": "0.05",
        }
    )

    cleaned = clean_data(raw).row(0, named=True)

    assert cleaned["city"] == "Metropolitan"
    assert cleaned["age_clipped"] == 18.0
    assert cleaned["age_under_18"] is True
    assert cleaned["rating_clipped"] == 5.0
    assert cleaned["rating_out_of_range"] is True
    assert cleaned["multiple_missing"] is True
    assert cleaned["time_order_missing"] is True
    assert cleaned["coordinates_invalid"] is True
    assert cleaned["coord_status"] == "invalid"
    assert cleaned["weather_missing"] is True
    assert cleaned["traffic_missing"] is True


def test_clean_data_repairs_sign_flipped_restaurant_coordinates() -> None:
    raw = make_raw(
        {
            "Restaurant_latitude": "-12.9",
            "Restaurant_longitude": "-77.5",
            "Delivery_location_latitude": "13.0",
            "Delivery_location_longitude": "77.6",
        }
    )

    cleaned = clean_data(raw).row(0, named=True)

    assert cleaned["restaurant_coords_sign_flipped"] is True
    assert cleaned["coordinates_invalid"] is False
    assert cleaned["coord_status"] == "sign_flipped"
    assert cleaned["restaurant_latitude_repaired"] == pytest.approx(12.9)
    assert cleaned["restaurant_longitude_repaired"] == pytest.approx(77.5)


def test_clean_data_rejects_missing_target() -> None:
    raw = make_raw({"Time_taken (min)": "NaN"})

    with pytest.raises(ValueError, match="Target column"):
        clean_data(raw)


def test_write_clean_artifacts(tmp_path: Path) -> None:
    raw = make_raw({})
    raw_path = tmp_path / "raw.csv"
    raw.write_csv(raw_path)
    cleaned = clean_data(raw)

    manifest = write_clean_artifacts(
        cleaned,
        raw_path=raw_path,
        parquet_path=tmp_path / "clean.parquet",
        csv_path=tmp_path / "clean.csv",
        manifest_path=tmp_path / "manifest.json",
    )

    assert (tmp_path / "clean.parquet").is_file()
    assert (tmp_path / "clean.csv").is_file()
    assert manifest["rows"] == 1
    assert manifest["target_column"] == "Time_taken (min)"
    assert json.loads((tmp_path / "manifest.json").read_text())["rows"] == 1
