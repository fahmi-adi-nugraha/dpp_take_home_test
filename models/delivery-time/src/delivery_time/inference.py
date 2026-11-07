"""Inference helpers for the delivery-time model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import polars as pl

from delivery_time.data import parse_time_minutes
from delivery_time.features import haversine_km

WEATHER_CATEGORIES = ("Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy")
TRAFFIC_CATEGORIES = ("Low", "Medium", "High", "Jam")
CITY_CATEGORIES = ("Metropolitan", "Semi-Urban", "Urban")
FESTIVAL_CATEGORIES = ("Yes", "No")

_TRAFFIC_ORDINAL = {value: index for index, value in enumerate(TRAFFIC_CATEGORIES)}
_FESTIVAL_BINARY = {"No": 0, "Yes": 1}


@dataclass(frozen=True)
class DeliveryTimeInput:
    """Raw business inputs needed to construct the delivery-time features."""

    restaurant_latitude: float | None = None
    restaurant_longitude: float | None = None
    delivery_latitude: float | None = None
    delivery_longitude: float | None = None
    order_time: str | None = None
    weather: str | None = None
    traffic: str | None = None
    city: str | None = None
    festival: str | None = None
    multiple_deliveries: int | None = None
    vehicle_condition: int | None = None
    courier_age: float | None = None
    courier_rating: float | None = None


@dataclass(frozen=True)
class DeliveryTimePrediction:
    """Prediction result and input-quality warnings."""

    prediction_minutes: float
    warnings: tuple[str, ...] = ()


def normalize_city(value: str | None) -> str | None:
    """Normalize the dataset's city spelling variant."""
    if value is None:
        return None
    city = value.strip()
    if not city:
        return None
    return "Metropolitan" if city == "Metropolitian" else city


def _coordinates_invalid(
    restaurant_latitude: float | None,
    restaurant_longitude: float | None,
    delivery_latitude: float | None,
    delivery_longitude: float | None,
) -> bool:
    """Mirror the coordinate validity policy from the cleaning pipeline."""
    coordinates = (
        restaurant_latitude,
        restaurant_longitude,
        delivery_latitude,
        delivery_longitude,
    )
    if any(value is None for value in coordinates):
        return True
    assert restaurant_latitude is not None
    assert restaurant_longitude is not None
    assert delivery_latitude is not None
    assert delivery_longitude is not None
    return (
        (restaurant_latitude == 0 and restaurant_longitude == 0)
        or (delivery_latitude == delivery_longitude and 0 < delivery_latitude < 1)
        or not (-90 <= restaurant_latitude <= 90)
        or not (-180 <= restaurant_longitude <= 180)
        or not (-90 <= delivery_latitude <= 90)
        or not (-180 <= delivery_longitude <= 180)
    )


def _repaired_coordinates(
    restaurant_latitude: float | None,
    restaurant_longitude: float | None,
    delivery_latitude: float | None,
    delivery_longitude: float | None,
) -> tuple[float | None, float | None]:
    """Repair sign-flipped restaurant coordinates as in the cleaning pipeline."""
    if restaurant_latitude is None or restaurant_longitude is None:
        return restaurant_latitude, restaurant_longitude
    if delivery_latitude is not None and restaurant_latitude < 0 < delivery_latitude:
        restaurant_latitude = abs(restaurant_latitude)
    if delivery_longitude is not None and restaurant_longitude < 0 < delivery_longitude:
        restaurant_longitude = abs(restaurant_longitude)
    return restaurant_latitude, restaurant_longitude


def _weather_group(weather: str | None) -> str | None:
    """Map the raw weather category to the model's grouped weather feature."""
    if weather is None:
        return None
    if weather == "Sunny":
        return "clear"
    if weather in {"Cloudy", "Fog"}:
        return "cloudy_fog"
    if weather in {"Stormy", "Sandstorms", "Windy"}:
        return "storm_wind"
    return None


def build_inference_features(payload: DeliveryTimeInput) -> pl.DataFrame:
    """Construct the one-row feature frame expected by the trained pipeline."""
    order_minutes = parse_time_minutes(payload.order_time)
    order_hour = None if order_minutes is None else (order_minutes // 60) % 24

    restaurant_latitude, restaurant_longitude = _repaired_coordinates(
        payload.restaurant_latitude,
        payload.restaurant_longitude,
        payload.delivery_latitude,
        payload.delivery_longitude,
    )
    coordinates_invalid = _coordinates_invalid(
        payload.restaurant_latitude,
        payload.restaurant_longitude,
        payload.delivery_latitude,
        payload.delivery_longitude,
    )
    distance_km = None
    if not coordinates_invalid:
        assert restaurant_latitude is not None
        assert restaurant_longitude is not None
        assert payload.delivery_latitude is not None
        assert payload.delivery_longitude is not None
        distance_km = haversine_km(
            restaurant_latitude,
            restaurant_longitude,
            payload.delivery_latitude,
            payload.delivery_longitude,
        )

    weather = None if payload.weather is None else payload.weather.strip() or None
    city = normalize_city(payload.city)
    multiple_deliveries = payload.multiple_deliveries

    row = {
        "distance_km": distance_km,
        "coordinates_invalid": coordinates_invalid,
        "traffic_ordinal": _TRAFFIC_ORDINAL.get(payload.traffic),
        "weather": weather,
        "city": city,
        "festival_binary": _FESTIVAL_BINARY.get(payload.festival),
        "multiple_deliveries": multiple_deliveries,
        "has_multiple_deliveries": (
            None if multiple_deliveries is None else multiple_deliveries > 0
        ),
        "vehicle_condition": payload.vehicle_condition,
        "age": payload.courier_age,
        "rating": payload.courier_rating,
        "order_hour": order_hour,
        "is_rush_hour": (
            None if order_hour is None else (11 <= order_hour < 14) or (17 <= order_hour < 21)
        ),
        "is_meal_period": (None if order_hour is None else 7 <= order_hour < 22),
        "time_order_missing": order_minutes is None,
        "city_missing": city is None,
        "multiple_missing": multiple_deliveries is None,
        "weather_group": _weather_group(weather),
    }

    return pl.DataFrame([row]).with_columns(
        pl.col("distance_km").cast(pl.Float64),
        pl.col("coordinates_invalid").cast(pl.Boolean),
        pl.col("traffic_ordinal").cast(pl.Int32),
        pl.col("weather").cast(pl.String),
        pl.col("city").cast(pl.String),
        pl.col("festival_binary").cast(pl.Int32),
        pl.col("multiple_deliveries").cast(pl.Int64),
        pl.col("has_multiple_deliveries").cast(pl.Boolean),
        pl.col("vehicle_condition").cast(pl.Int64),
        pl.col("age").cast(pl.Float64),
        pl.col("rating").cast(pl.Float64),
        pl.col("order_hour").cast(pl.Int32),
        pl.col("is_rush_hour").cast(pl.Boolean),
        pl.col("is_meal_period").cast(pl.Boolean),
        pl.col("time_order_missing").cast(pl.Boolean),
        pl.col("city_missing").cast(pl.Boolean),
        pl.col("multiple_missing").cast(pl.Boolean),
        pl.col("weather_group").cast(pl.String),
    )


def _input_warnings(features: pl.DataFrame) -> tuple[str, ...]:
    """Return user-facing warnings for inputs that required fallback handling."""
    row = features.row(0, named=True)
    warnings: list[str] = []
    if row["coordinates_invalid"]:
        warnings.append("Coordinates were missing or invalid; distance_km was left null.")
    if row["time_order_missing"]:
        warnings.append("Order time was missing; time-window features were left null.")
    for column, label in (
        ("weather", "Weather"),
        ("city", "City"),
        ("traffic_ordinal", "Traffic density"),
        ("festival_binary", "Festival indicator"),
        ("multiple_deliveries", "Multiple-delivery count"),
        ("vehicle_condition", "Vehicle condition"),
        ("age", "Courier age"),
        ("rating", "Courier rating"),
    ):
        if row[column] is None:
            warnings.append(f"{label} was missing; the model used its missing-value handling.")
    return tuple(warnings)


@dataclass
class DeliveryTimePredictor:
    """Load and run the delivery-time pipeline."""

    model: Any
    feature_names: tuple[str, ...]
    model_version: str | None = None
    selected_feature_set: str | None = None
    model_name: str = "delivery-time-xgboost"

    @classmethod
    def load(cls, model_path: Path) -> DeliveryTimePredictor:
        """Load the fitted pipeline and its feature metadata."""
        if not model_path.exists():
            raise FileNotFoundError(f"Delivery-time model artifact not found: {model_path}")

        model = joblib.load(model_path)
        metadata_path = model_path.with_name("model_metadata.json")
        schema_path = model_path.with_name("feature_schema.json")
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        feature_names = tuple(metadata.get("feature_names", ()))
        if not feature_names and schema_path.exists():
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            feature_names = tuple(item["name"] for item in schema.get("features", ()))
        if not feature_names:
            raise ValueError(f"No feature metadata found next to {model_path}")

        return cls(
            model=model,
            feature_names=feature_names,
            model_version=metadata.get("model_version"),
            selected_feature_set=metadata.get("selected_feature_set"),
        )

    def predict(self, payload: DeliveryTimeInput) -> DeliveryTimePrediction:
        """Build features and return the predicted delivery time in minutes."""
        features = build_inference_features(payload)
        values = features.select(list(self.feature_names)).to_numpy()
        prediction = float(self.model.predict(values)[0])
        return DeliveryTimePrediction(
            prediction_minutes=prediction,
            warnings=_input_warnings(features),
        )
