"""Tests for delivery-time inference feature construction and prediction."""

from __future__ import annotations

import polars as pl
import pytest
from delivery_time import config, features
from delivery_time.data import clean_data
from delivery_time.features import build_feature_frame
from delivery_time.inference import (
    DeliveryTimeInput,
    DeliveryTimePredictor,
    build_inference_features,
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


def make_payload(**overrides: object) -> DeliveryTimeInput:
    values = {
        "restaurant_latitude": 12.9,
        "restaurant_longitude": 77.5,
        "delivery_latitude": 13.0,
        "delivery_longitude": 77.6,
        "order_time": "12:00",
        "weather": "Cloudy",
        "traffic": "Jam",
        "city": "Metropolitian",
        "festival": "Yes",
        "multiple_deliveries": 2,
        "vehicle_condition": 0,
        "courier_age": 30.0,
        "courier_rating": 4.7,
    }
    values.update(overrides)
    return DeliveryTimeInput(**values)


def test_inference_features_match_training_feature_construction() -> None:
    raw = pl.DataFrame([RAW_DEFAULTS])
    training_features = build_feature_frame(clean_data(raw))
    inference_features = build_inference_features(make_payload())

    training_row = training_features.row(0, named=True)
    inference_row = inference_features.row(0, named=True)
    selected = (*features.BASELINE_FEATURES, "weather_group")

    for name in selected:
        if isinstance(training_row[name], float):
            assert inference_row[name] == pytest.approx(training_row[name])
        else:
            assert inference_row[name] == training_row[name]


def test_inference_features_handle_missing_values() -> None:
    row = build_inference_features(DeliveryTimeInput()).row(0, named=True)

    assert row["distance_km"] is None
    assert row["coordinates_invalid"] is True
    assert row["traffic_ordinal"] is None
    assert row["weather"] is None
    assert row["city"] is None
    assert row["festival_binary"] is None
    assert row["multiple_deliveries"] is None
    assert row["has_multiple_deliveries"] is None
    assert row["order_hour"] is None
    assert row["is_rush_hour"] is None
    assert row["is_meal_period"] is None
    assert row["time_order_missing"] is True
    assert row["city_missing"] is True
    assert row["multiple_missing"] is True
    assert row["weather_group"] is None


def test_predictor_uses_feature_order_and_reports_warnings() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.values = None

        def predict(self, values):
            self.values = values
            return [26.5]

    model = FakeModel()
    predictor = DeliveryTimePredictor(
        model=model,
        feature_names=(
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
            "weather_group",
        ),
        model_version="1",
        selected_feature_set="plus_weather_group",
    )
    result = predictor.predict(DeliveryTimeInput())

    assert model.values is not None
    assert model.values.shape == (1, 18)
    assert result.prediction_minutes == 26.5
    assert result.warnings
    assert any("distance_km" in warning for warning in result.warnings)


def test_predictor_loads_the_local_artifact() -> None:
    if not config.MODEL_PATH.exists():
        pytest.skip("The delivery-time model artifact is not available.")

    predictor = DeliveryTimePredictor.load(config.MODEL_PATH)
    result = predictor.predict(make_payload())

    assert predictor.model_version == "1"
    assert predictor.selected_feature_set == "plus_weather_group"
    assert 0 < result.prediction_minutes < 120
    assert result.warnings == ()
