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
from yelp_sentiment.inference import SentimentPrediction


class FakePredictor:
    """Small predictor stub so API tests do not depend on model artifacts."""

    model_name = "delivery-time-xgboost"
    model_version = "1"
    selected_feature_set = "plus_weather_group"

    def predict(self, payload):
        return DeliveryTimePrediction(prediction_minutes=27.42)


class FakeSentimentPredictor:
    """Small sentiment stub with the exported-model contract."""

    model_name = "bert-base-uncased"
    model_version = "1"

    def predict(self, payload):
        return SentimentPrediction(
            label=4,
            label_name="5 stars",
            confidence=0.78,
            probabilities=(0.01, 0.02, 0.04, 0.15, 0.78),
            label_map={
                0: "1 star",
                1: "2 star",
                2: "3 stars",
                3: "4 stars",
                4: "5 stars",
            },
        )


class FakeRegistry:
    """Registry stub with configurable delivery and sentiment readiness."""

    def __init__(
        self,
        *,
        delivery_loaded: bool = True,
        sentiment_loaded: bool = True,
    ) -> None:
        delivery_path = Path("models/delivery-time/artifacts/model.joblib")
        sentiment_path = Path("models/yelp-sentiment/artifacts/bert/model.pt")
        self.delivery_time = FakePredictor() if delivery_loaded else None
        self.sentiment = FakeSentimentPredictor() if sentiment_loaded else None
        self.delivery_time_status = ModelStatus(
            model="delivery-time",
            loaded=delivery_loaded,
            path=delivery_path,
            error=None if delivery_loaded else "Model artifact is unavailable.",
            model_version="1" if delivery_loaded else None,
            selected_feature_set="plus_weather_group" if delivery_loaded else None,
        )
        self.sentiment_status = ModelStatus(
            model="sentiment",
            loaded=sentiment_loaded,
            path=sentiment_path,
            error=None if sentiment_loaded else "Sentiment model is unavailable.",
            model_version="1" if sentiment_loaded else None,
        )

    def load(self) -> None:
        """Keep the injected state unchanged."""

    @property
    def delivery_time_ready(self) -> bool:
        return self.delivery_time is not None

    @property
    def sentiment_ready(self) -> bool:
        return self.sentiment is not None

    @property
    def all_ready(self) -> bool:
        return self.delivery_time_ready and self.sentiment_ready

    def statuses(self) -> list[ModelStatus]:
        return [self.delivery_time_status, self.sentiment_status]


def _client(
    *,
    delivery_loaded: bool = True,
    sentiment_loaded: bool = True,
) -> Iterator[TestClient]:
    app = create_app(
        settings=Settings(),
        registry=FakeRegistry(
            delivery_loaded=delivery_loaded,
            sentiment_loaded=sentiment_loaded,
        ),
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture
def loaded_client() -> Iterator[TestClient]:
    yield from _client()


@pytest.fixture
def unloaded_client() -> Iterator[TestClient]:
    yield from _client(delivery_loaded=False)


@pytest.fixture
def sentiment_unloaded_client() -> Iterator[TestClient]:
    yield from _client(sentiment_loaded=False)
