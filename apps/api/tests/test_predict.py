"""Tests for prediction endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

DELIVERY_PAYLOAD = {
    "restaurant": {"latitude": 12.9716, "longitude": 77.5946},
    "delivery": {"latitude": 12.9352, "longitude": 77.6245},
    "order_time": "12:30",
    "weather": "Sunny",
    "traffic": "Medium",
    "city": "Urban",
    "festival": "No",
    "multiple_deliveries": 1,
    "vehicle_condition": 1,
    "courier_age": 30,
    "courier_rating": 4.7,
}

SENTIMENT_PAYLOAD = {
    "text": "The food was great and the delivery was fast.",
}


def test_predict_delivery_time(loaded_client: TestClient) -> None:
    response = loaded_client.post("/predict/delivery-time", json=DELIVERY_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction_minutes"] == 27.42
    assert body["model"] == "delivery-time-xgboost"
    assert body["model_version"] == "1"
    assert body["selected_feature_set"] == "plus_weather_group"
    assert body["prediction_point"] == "order_placement"
    assert body["warnings"] == []


def test_predict_rejects_invalid_coordinates(loaded_client: TestClient) -> None:
    payload = {
        **DELIVERY_PAYLOAD,
        "restaurant": {"latitude": 100, "longitude": 77.5946},
    }
    response = loaded_client.post("/predict/delivery-time", json=payload)

    assert response.status_code == 422


def test_predict_rejects_invalid_order_time(loaded_client: TestClient) -> None:
    payload = {**DELIVERY_PAYLOAD, "order_time": "25:00"}
    response = loaded_client.post("/predict/delivery-time", json=payload)

    assert response.status_code == 422


def test_predict_returns_503_when_delivery_model_is_unavailable(
    unloaded_client: TestClient,
) -> None:
    response = unloaded_client.post("/predict/delivery-time", json=DELIVERY_PAYLOAD)

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_predict_sentiment(loaded_client: TestClient) -> None:
    response = loaded_client.post("/predict/sentiment", json=SENTIMENT_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == 4
    assert body["label_name"] == "5 stars"
    assert body["confidence"] == 0.78
    assert body["probabilities"]["5 stars"] == 0.78
    assert body["model"] == "bert-base-uncased"
    assert body["model_version"] == "1"


def test_predict_sentiment_returns_503_when_model_is_unavailable(
    sentiment_unloaded_client: TestClient,
) -> None:
    response = sentiment_unloaded_client.post(
        "/predict/sentiment",
        json=SENTIMENT_PAYLOAD,
    )

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_predict_sentiment_rejects_empty_text(loaded_client: TestClient) -> None:
    response = loaded_client.post("/predict/sentiment", json={"text": ""})

    assert response.status_code == 422
