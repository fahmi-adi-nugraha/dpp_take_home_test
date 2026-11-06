"""Feature construction and feature-set definitions for delivery time."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import polars as pl

from delivery_time import config
from delivery_time.data import write_dataset_artifacts

METADATA_COLUMNS = ("id", "courier_id", "order_date", "target")

BASELINE_FEATURES = (
    "distance_km",
    "coordinates_invalid",
    "traffic_ordinal",
    "weather",
    "city",
    "festival_binary",
    "multiple_deliveries",
    "has_multiple_deliveries",
    "vehicle_condition",
    "age",
    "rating",
    "order_hour",
    "is_rush_hour",
    "is_meal_period",
    "time_order_missing",
    "city_missing",
    "multiple_missing",
)

LINEAR_EXTRA_FEATURES = (
    "order_hour_sin",
    "order_hour_cos",
    "age_bin",
    "rating_bin",
)

OPTIONAL_FEATURES = (
    "distance_missing",
    "restaurant_coords_sign_flipped",
    "coord_status",
    "weather_group",
    "vehicle_type",
    "order_type",
    "day_of_week",
    "is_weekend",
    "festival_missing",
    "age_missing",
    "rating_missing",
    "weather_missing",
    "traffic_missing",
    "age_clipped",
    "rating_clipped",
    "age_under_18",
    "rating_out_of_range",
    "courier_delivery_count",
    "distance_decile",
    "vehicle_condition_is_zero",
    "restaurant_grid",
    "delivery_grid",
)

ALL_FEATURES = BASELINE_FEATURES + LINEAR_EXTRA_FEATURES + OPTIONAL_FEATURES

EXCLUDED_FEATURES = (
    "id",
    "courier_id",
    "order_date",
    "target",
    "order_minutes",
    "pickup_minutes",
    "prep_minutes",
    "Time_Order_picked",
)

LEAKAGE_FIELDS = (
    "pickup_minutes",
    "prep_minutes",
    "Time_Order_picked",
)

CATEGORICAL_FEATURES = (
    "weather",
    "city",
    "weather_group",
    "vehicle_type",
    "coord_status",
    "rating_bin",
    "restaurant_grid",
    "delivery_grid",
)

ORDINAL_FEATURES = (
    "traffic_ordinal",
    "vehicle_condition",
)

NUMERIC_FEATURES = (
    "distance_km",
    "multiple_deliveries",
    "age",
    "age_clipped",
    "rating",
    "rating_clipped",
    "order_hour",
    "order_hour_sin",
    "order_hour_cos",
    "day_of_week",
    "age_bin",
    "distance_decile",
    "courier_delivery_count",
)

BINARY_FEATURES = (
    "coordinates_invalid",
    "distance_missing",
    "restaurant_coords_sign_flipped",
    "festival_binary",
    "has_multiple_deliveries",
    "vehicle_condition_is_zero",
    "is_weekend",
    "is_rush_hour",
    "is_meal_period",
    "time_order_missing",
    "city_missing",
    "multiple_missing",
    "festival_missing",
    "age_missing",
    "rating_missing",
    "weather_missing",
    "traffic_missing",
    "age_under_18",
    "rating_out_of_range",
)

FEATURE_DESCRIPTIONS = {
    "distance_km": "Haversine distance from restaurant to delivery location.",
    "coordinates_invalid": "Coordinate pair failed the validity checks.",
    "traffic_ordinal": "Road traffic density encoded Low=0, Medium=1, High=2, Jam=3.",
    "weather": "Raw weather category at delivery time.",
    "city": "Normalized city or area type.",
    "festival_binary": "Festival indicator: Yes=1, No=0, missing=null.",
    "multiple_deliveries": "Number of deliveries in the same trip.",
    "has_multiple_deliveries": "Binary flag for multiple_deliveries > 0.",
    "vehicle_condition": "Ordinal vehicle condition score 0–3.",
    "age": "Courier age in years.",
    "rating": "Courier rating, including out-of-range values.",
    "order_hour": "Hour of order placement.",
    "is_rush_hour": "Order hour falls in a rush-hour window.",
    "is_meal_period": "Order hour falls in the broad meal window.",
    "time_order_missing": "Order time is missing.",
    "city_missing": "City is missing.",
    "multiple_missing": "Multiple-delivery count is missing.",
    "order_hour_sin": "Cyclical sine encoding of order hour.",
    "order_hour_cos": "Cyclical cosine encoding of order hour.",
    "age_bin": "Five-year courier age bin.",
    "rating_bin": "Coarse courier rating bin.",
    "distance_decile": "Distance decile over valid distances.",
    "vehicle_condition_is_zero": "Vehicle condition is exactly zero.",
    "distance_missing": "Distance could not be computed.",
    "restaurant_coords_sign_flipped": "Restaurant coordinates appear sign-flipped.",
    "coord_status": "Coordinate diagnostic: valid, sign_flipped, or invalid.",
    "weather_group": "Grouped weather: clear, cloudy_fog, or storm_wind.",
    "vehicle_type": "Vehicle used for delivery.",
    "order_type": "Order type: Snack, Meal, Drinks, or Buffet.",
    "day_of_week": "Day of week, Monday=0.",
    "is_weekend": "Order date falls on Saturday or Sunday.",
    "festival_missing": "Festival indicator is missing.",
    "age_missing": "Courier age is missing.",
    "rating_missing": "Courier rating is missing.",
    "weather_missing": "Weather is missing.",
    "traffic_missing": "Traffic density is missing.",
    "age_clipped": "Courier age clipped to 18–50.",
    "rating_clipped": "Courier rating clipped to 1–5.",
    "age_under_18": "Courier age is below 18.",
    "rating_out_of_range": "Courier rating is outside the expected 1–5 range.",
    "courier_delivery_count": "Number of deliveries by the courier in this dataset.",
    "restaurant_grid": "Rounded restaurant coordinates (2 decimal places).",
    "delivery_grid": "Rounded delivery coordinates (2 decimal places).",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance between two points in kilometres."""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def add_distance_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Compute distance and missing-distance flags from cleaned coordinates."""
    distances = []
    for row in frame.iter_rows(named=True):
        if row["coordinates_invalid"]:
            distances.append(None)
        else:
            distances.append(
                haversine_km(
                    row["restaurant_latitude_repaired"],
                    row["restaurant_longitude_repaired"],
                    row["delivery_latitude"],
                    row["delivery_longitude"],
                )
            )

    frame = frame.with_columns(pl.Series("distance_km", distances, dtype=pl.Float64))
    return frame.with_columns(pl.col("distance_km").is_null().alias("distance_missing"))


def add_time_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Derive order-hour, day-of-week, and time-window features."""
    frame = frame.with_columns(
        pl.when(pl.col("order_minutes").is_not_null())
        .then(((pl.col("order_minutes") // 60) % 24).cast(pl.Int32))
        .otherwise(None)
        .alias("order_hour")
    )
    frame = frame.with_columns(
        (pl.col("order_hour").cast(pl.Float64) * (2 * math.pi / 24)).sin().alias("order_hour_sin"),
        (pl.col("order_hour").cast(pl.Float64) * (2 * math.pi / 24)).cos().alias("order_hour_cos"),
        (pl.col("order_date").dt.weekday() - 1).alias("day_of_week"),
    )
    return frame.with_columns(
        (pl.col("day_of_week") >= 5).alias("is_weekend"),
        pl.when(pl.col("order_hour").is_null())
        .then(None)
        .otherwise(
            ((pl.col("order_hour") >= 11) & (pl.col("order_hour") < 14))
            | ((pl.col("order_hour") >= 17) & (pl.col("order_hour") < 21))
        )
        .alias("is_rush_hour"),
        pl.when(pl.col("order_hour").is_null())
        .then(None)
        .otherwise((pl.col("order_hour") >= 7) & (pl.col("order_hour") < 22))
        .alias("is_meal_period"),
    )


def add_courier_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Expose raw, clipped, binned, and flagged courier attributes."""
    frame = frame.with_columns(
        pl.col("age_raw").alias("age"),
        pl.col("rating_raw").alias("rating"),
        pl.when(pl.col("age_raw").is_null())
        .then(None)
        .otherwise(((pl.col("age_raw") // 5) * 5).clip(15, 50))
        .cast(pl.Int32)
        .alias("age_bin"),
        pl.when(pl.col("rating_raw").is_null())
        .then(None)
        .when(pl.col("rating_raw") < 3.0)
        .then(pl.lit("low"))
        .when(pl.col("rating_raw") < 4.5)
        .then(pl.lit("mid"))
        .when(pl.col("rating_raw") < 5.0)
        .then(pl.lit("high"))
        .when(pl.col("rating_raw") == 5.0)
        .then(pl.lit("max"))
        .otherwise(pl.lit("out_of_range"))
        .alias("rating_bin"),
    )
    return frame


def add_order_context_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Add binary, ordinal, and grouped order-context features."""
    return frame.with_columns(
        pl.when(pl.col("traffic").is_null())
        .then(None)
        .when(pl.col("traffic") == "Low")
        .then(0)
        .when(pl.col("traffic") == "Medium")
        .then(1)
        .when(pl.col("traffic") == "High")
        .then(2)
        .when(pl.col("traffic") == "Jam")
        .then(3)
        .otherwise(None)
        .cast(pl.Int32)
        .alias("traffic_ordinal"),
        pl.when(pl.col("festival") == "Yes")
        .then(1)
        .when(pl.col("festival") == "No")
        .then(0)
        .otherwise(None)
        .cast(pl.Int32)
        .alias("festival_binary"),
        (pl.col("multiple_deliveries") > 0).alias("has_multiple_deliveries"),
        (pl.col("vehicle_condition") == 0).alias("vehicle_condition_is_zero"),
        pl.when(pl.col("weather").is_null())
        .then(None)
        .when(pl.col("weather") == "Sunny")
        .then(pl.lit("clear"))
        .when(pl.col("weather").is_in(["Cloudy", "Fog"]))
        .then(pl.lit("cloudy_fog"))
        .when(pl.col("weather").is_in(["Stormy", "Sandstorms", "Windy"]))
        .then(pl.lit("storm_wind"))
        .otherwise(None)
        .alias("weather_group"),
    )


def add_coordinate_grid_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Add optional rounded-coordinate grid categories for experiments."""
    return frame.with_columns(
        pl.when(pl.col("coordinates_invalid"))
        .then(None)
        .otherwise(
            pl.col("restaurant_latitude_repaired").round(2).cast(pl.String)
            + ","
            + pl.col("restaurant_longitude_repaired").round(2).cast(pl.String)
        )
        .alias("restaurant_grid"),
        pl.when(pl.col("coordinates_invalid"))
        .then(None)
        .otherwise(
            pl.col("delivery_latitude").round(2).cast(pl.String)
            + ","
            + pl.col("delivery_longitude").round(2).cast(pl.String)
        )
        .alias("delivery_grid"),
    )


def add_distance_decile(frame: pl.DataFrame) -> pl.DataFrame:
    """Add a decile rank for valid distances."""
    valid_count = frame.filter(pl.col("distance_km").is_not_null()).height
    if valid_count == 0:
        return frame.with_columns(pl.lit(None, dtype=pl.Int32).alias("distance_decile"))
    return frame.with_columns(
        pl.when(pl.col("distance_km").is_not_null())
        .then(
            ((pl.col("distance_km").rank("ordinal") - 1) * 10 / pl.lit(valid_count))
            .floor()
            .clip(0, 9)
            .cast(pl.Int32)
        )
        .otherwise(None)
        .alias("distance_decile")
    )


def add_courier_frequency(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the number of deliveries observed for each courier."""
    return frame.with_columns(
        pl.col("courier_id").count().over("courier_id").alias("courier_delivery_count")
    )


def build_feature_frame(cleaned: pl.DataFrame) -> pl.DataFrame:
    """Build the semantic candidate feature frame from cleaned data."""
    frame = add_distance_features(cleaned)
    frame = add_time_features(frame)
    frame = add_courier_features(frame)
    frame = add_order_context_features(frame)
    frame = add_distance_decile(frame)
    frame = add_courier_frequency(frame)
    frame = add_coordinate_grid_features(frame)
    return frame.select([*METADATA_COLUMNS, *ALL_FEATURES])


def feature_set_table() -> pl.DataFrame:
    """Return a table describing each candidate feature and its role."""
    rows = []
    for feature in ALL_FEATURES:
        sets = []
        if feature in BASELINE_FEATURES:
            sets.append("baseline")
        if feature in LINEAR_EXTRA_FEATURES:
            sets.append("linear_extra")
        if feature in OPTIONAL_FEATURES:
            sets.append("optional")
        rows.append(
            {
                "feature": feature,
                "description": FEATURE_DESCRIPTIONS.get(feature, ""),
                "sets": ", ".join(sets) or "unassigned",
            }
        )
    return pl.DataFrame(rows)


def feature_manifest_extra() -> dict[str, Any]:
    """Return manifest metadata describing the engineered feature frame."""
    return {
        "target_feature": config.TARGET_FEATURE_NAME,
        "metadata_columns": list(METADATA_COLUMNS),
        "baseline_features": list(BASELINE_FEATURES),
        "linear_extra_features": list(LINEAR_EXTRA_FEATURES),
        "optional_features": list(OPTIONAL_FEATURES),
        "excluded_features": list(EXCLUDED_FEATURES),
        "leakage_fields": list(LEAKAGE_FIELDS),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "ordinal_features": list(ORDINAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "binary_features": list(BINARY_FEATURES),
    }


def write_feature_artifacts(
    features: pl.DataFrame,
    *,
    clean_path: Path = config.CLEAN_DATA_PARQUET,
    parquet_path: Path = config.FEATURES_DATA_PARQUET,
    csv_path: Path = config.FEATURES_DATA_CSV,
    manifest_path: Path = config.FEATURES_MANIFEST_PATH,
) -> dict[str, Any]:
    """Write the engineered feature dataset and its manifest."""
    return write_dataset_artifacts(
        features,
        parquet_path=parquet_path,
        csv_path=csv_path,
        manifest_path=manifest_path,
        dataset_name="zomato_features",
        source_files=[clean_path],
        version=config.FEATURE_VERSION,
        extra_manifest=feature_manifest_extra(),
    )
