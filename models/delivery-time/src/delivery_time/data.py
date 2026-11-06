"""Data loading, cleaning, and artifact writing for the delivery-time project."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from delivery_time import config

MISSING_TOKENS = {"NaN", "nan", "", "None", None}


def is_missing(value: object) -> bool:
    """Return whether a raw value is one of the dataset's missing tokens."""
    return value is None or str(value).strip() in MISSING_TOKENS


def as_float(value: object) -> float | None:
    """Parse a raw numeric value, returning None for missing tokens."""
    return None if is_missing(value) else float(str(value).strip())


def as_int(value: object) -> int | None:
    """Parse a raw integer value, returning None for missing tokens."""
    return None if is_missing(value) else int(float(str(value).strip()))


def parse_time_minutes(value: object) -> int | None:
    """Normalize a mixed-format clock time to minutes since midnight."""
    if is_missing(value):
        return None

    text = str(value).strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = int(match.group(3) or 0)
        return round(hour * 60 + minute + second / 60)

    try:
        return round(float(text) * 24 * 60)
    except ValueError:
        return None


def classify_time_format(value: object) -> str:
    """Classify the raw representation used by a time value."""
    if is_missing(value):
        return "missing"

    text = str(value).strip()
    if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", text):
        return "HH:MM:SS"
    if re.fullmatch(r"\d{1,2}:\d{2}", text):
        return "HH:MM"
    if text == "1":
        return "1 (24:00)"
    try:
        float(text)
        return "fractional day"
    except ValueError:
        return "other"


def parse_order_date(value: object) -> date:
    """Parse the dataset's day-first order date."""
    if is_missing(value):
        raise ValueError("Order_Date cannot be missing")
    return datetime.strptime(str(value).strip(), "%d-%m-%Y").date()


def load_raw_data(path: Path = config.RAW_CSV_PATH) -> pl.DataFrame:
    """Load the raw CSV with every column as text."""
    return pl.read_csv(path, infer_schema_length=0)


def load_clean_data(path: Path = config.CLEAN_DATA_PARQUET) -> pl.DataFrame:
    """Load the typed cleaned dataset."""
    return pl.read_parquet(path)


def clean_data(raw: pl.DataFrame) -> pl.DataFrame:
    """Parse, normalize, and flag the raw delivery dataset."""
    rows = []
    for record in raw.iter_rows(named=True):
        rlat = as_float(record["Restaurant_latitude"])
        rlon = as_float(record["Restaurant_longitude"])
        dlat = as_float(record["Delivery_location_latitude"])
        dlon = as_float(record["Delivery_location_longitude"])
        age = as_float(record["Delivery_person_Age"])
        rating = as_float(record["Delivery_person_Ratings"])
        multiple = as_int(record["multiple_deliveries"])
        target = as_int(record["Time_taken (min)"])
        if target is None:
            raise ValueError("Target column contains a missing value")

        order_minutes = parse_time_minutes(record["Time_Orderd"])
        pickup_minutes = parse_time_minutes(record["Time_Order_picked"])
        prep_minutes = None
        if order_minutes is not None and pickup_minutes is not None:
            prep_minutes = pickup_minutes - order_minutes
            if prep_minutes < 0:
                prep_minutes += 1440

        coordinates_invalid = (
            None in (rlat, rlon, dlat, dlon)
            or (rlat == 0 and rlon == 0)
            or (dlat == dlon and 0 < dlat < 1)
            or not (-90 <= rlat <= 90)
            or not (-180 <= rlon <= 180)
            or not (-90 <= dlat <= 90)
            or not (-180 <= dlon <= 180)
        )
        sign_flipped = (rlat is not None and dlat is not None and rlat < 0 and dlat > 0) or (
            rlon is not None and dlon is not None and rlon < 0 and dlon > 0
        )
        repaired_lat = (
            abs(rlat) if rlat is not None and dlat is not None and rlat < 0 and dlat > 0 else rlat
        )
        repaired_lon = (
            abs(rlon) if rlon is not None and dlon is not None and rlon < 0 and dlon > 0 else rlon
        )

        city = None if is_missing(record["City"]) else record["City"].strip()
        if city == "Metropolitian":
            city = "Metropolitan"

        rows.append(
            {
                "id": record["ID"],
                "courier_id": record["Delivery_person_ID"],
                "order_date": parse_order_date(record["Order_Date"]),
                "age_raw": age,
                "age_clipped": (None if age is None else min(max(age, 18.0), 50.0)),
                "age_under_18": age is not None and age < 18,
                "age_missing": age is None,
                "rating_raw": rating,
                "rating_clipped": (None if rating is None else min(max(rating, 1.0), 5.0)),
                "rating_out_of_range": (rating is not None and (rating < 1.0 or rating > 5.0)),
                "rating_missing": rating is None,
                "restaurant_latitude": rlat,
                "restaurant_longitude": rlon,
                "restaurant_latitude_repaired": repaired_lat,
                "restaurant_longitude_repaired": repaired_lon,
                "delivery_latitude": dlat,
                "delivery_longitude": dlon,
                "coordinates_invalid": coordinates_invalid,
                "restaurant_coords_sign_flipped": sign_flipped,
                "coord_status": (
                    "invalid"
                    if coordinates_invalid
                    else "sign_flipped"
                    if sign_flipped
                    else "valid"
                ),
                "order_minutes": order_minutes,
                "pickup_minutes": pickup_minutes,
                "prep_minutes": prep_minutes,
                "time_order_missing": order_minutes is None,
                "weather": (
                    None
                    if is_missing(record["Weather_conditions"])
                    else record["Weather_conditions"].strip()
                ),
                "weather_missing": is_missing(record["Weather_conditions"]),
                "traffic": (
                    None
                    if is_missing(record["Road_traffic_density"])
                    else record["Road_traffic_density"].strip()
                ),
                "traffic_missing": is_missing(record["Road_traffic_density"]),
                "vehicle_condition": as_int(record["Vehicle_condition"]),
                "order_type": (
                    None if is_missing(record["Type_of_order"]) else record["Type_of_order"].strip()
                ),
                "vehicle_type": (
                    None
                    if is_missing(record["Type_of_vehicle"])
                    else record["Type_of_vehicle"].strip()
                ),
                "multiple_deliveries": multiple,
                "multiple_missing": multiple is None,
                "festival": (
                    None if is_missing(record["Festival"]) else record["Festival"].strip()
                ),
                "festival_missing": is_missing(record["Festival"]),
                "city": city,
                "city_missing": city is None,
                "target": target,
            }
        )

    return pl.DataFrame(rows)


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(config.CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset_artifacts(
    frame: pl.DataFrame,
    *,
    parquet_path: Path,
    csv_path: Path,
    manifest_path: Path,
    dataset_name: str,
    source_files: list[Path],
    version: str,
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write Parquet, CSV, and a reproducible manifest."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(parquet_path)
    frame.write_csv(csv_path)

    manifest: dict[str, Any] = {
        "dataset": dataset_name,
        "version": version,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "rows": frame.height,
        "columns": frame.width,
        "source_files": [
            {
                "path": _project_relative(path),
                "sha256": sha256_file(path),
            }
            for path in source_files
        ],
        "outputs": {
            "parquet": {
                "path": _project_relative(parquet_path),
                "sha256": sha256_file(parquet_path),
            },
            "csv": {
                "path": _project_relative(csv_path),
                "sha256": sha256_file(csv_path),
            },
        },
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    part_path = manifest_path.with_name(f"{manifest_path.name}.part")
    part_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(part_path, manifest_path)
    return manifest


def write_clean_artifacts(
    cleaned: pl.DataFrame,
    *,
    raw_path: Path = config.RAW_CSV_PATH,
    parquet_path: Path = config.CLEAN_DATA_PARQUET,
    csv_path: Path = config.CLEAN_DATA_CSV,
    manifest_path: Path = config.CLEAN_MANIFEST_PATH,
) -> dict[str, Any]:
    """Write the cleaned dataset and its manifest."""
    return write_dataset_artifacts(
        cleaned,
        parquet_path=parquet_path,
        csv_path=csv_path,
        manifest_path=manifest_path,
        dataset_name="zomato_clean",
        source_files=[raw_path],
        version=config.CLEANING_VERSION,
        extra_manifest={
            "target_column": config.TARGET_COLUMN,
            "target_feature": config.TARGET_FEATURE_NAME,
        },
    )
