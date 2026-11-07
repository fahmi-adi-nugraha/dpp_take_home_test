"""Shared API test fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from delivery_time.inference import DeliveryTimePrediction
from fastapi.testclient import TestClient
from inference_api.main import create_app
from inference_api.registry import ModelStatus
from inference_api.settings import Settings


class FakePredictor:
    """Small predictor stub so API tests do not depend on model artifacts."""

    model_name = "delivery-time-xgboost"
    model_version = "1"
    selected_feature_set = "plus_weather_group"

    def predict(self, payload):
        return DeliveryTimePrediction(prediction_minutes=27.42)


class FakeRegistry:
    """Registry stub with configurable delivery-time readiness."""

    def __init__(self, *, loaded: bool) -> None:
        path = Path("models/delivery-time/artifacts/model.joblib")
        self.delivery_time = FakePredictor() if loaded else None
        self.delivery_time_status = ModelStatus(
            model="delivery-time",
            loaded=loaded,
            path=path,
            error=None if loaded else "Model artifact is unavailable.",
            model_version="1" if loaded else None,
            selected_feature_set="plus_weather_group" if loaded else None,
        )
        self.sentiment_status = ModelStatus(
            model="sentiment",
            loaded=False,
            path=Path("models/yelp-sentiment/artifacts/bert/model"),
            error="Sentiment inference is not implemented yet.",
        )

    def load(self) -> None:
        """Keep the injected state unchanged."""

    @property
    def delivery_time_ready(self) -> bool:
        return self.delivery_time is not None

    def statuses(self) -> list[ModelStatus]:
        return [self.delivery_time_status, self.sentiment_status]


def _client(*, loaded: bool) -> Iterator[TestClient]:
    app = create_app(
        settings=Settings(),
        registry=FakeRegistry(loaded=loaded),
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture
def loaded_client() -> Iterator[TestClient]:
    yield from _client(loaded=True)


@pytest.fixture
def unloaded_client() -> Iterator[TestClient]:
    yield from _client(loaded=False)
