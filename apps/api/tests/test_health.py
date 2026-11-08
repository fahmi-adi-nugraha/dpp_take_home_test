"""Tests for health and model-status endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_reports_ready_models(loaded_client: TestClient) -> None:
    response = loaded_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    statuses = {item["model"]: item for item in body["models"]}
    assert statuses["delivery-time"]["loaded"] is True
    assert statuses["delivery-time"]["model_version"] == "1"
    assert statuses["delivery-time"]["selected_feature_set"] == "plus_weather_group"
    assert statuses["sentiment"]["loaded"] is True
    assert statuses["sentiment"]["model_version"] == "1"
    assert statuses["sentiment"]["device"] == "cpu"


def test_health_reports_degraded_without_delivery_model(
    unloaded_client: TestClient,
) -> None:
    response = unloaded_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    statuses = {item["model"]: item for item in body["models"]}
    assert statuses["delivery-time"]["loaded"] is False
    assert statuses["delivery-time"]["error"]


def test_health_reports_degraded_without_sentiment_model(
    sentiment_unloaded_client: TestClient,
) -> None:
    response = sentiment_unloaded_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    statuses = {item["model"]: item for item in body["models"]}
    assert statuses["sentiment"]["loaded"] is False
    assert statuses["sentiment"]["error"]


def test_models_lists_both_model_families(loaded_client: TestClient) -> None:
    response = loaded_client.get("/models")

    assert response.status_code == 200
    models = {item["model"]: item for item in response.json()["models"]}
    assert set(models) == {"delivery-time", "sentiment"}
