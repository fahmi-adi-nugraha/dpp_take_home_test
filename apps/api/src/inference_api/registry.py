"""Model registry and startup loading."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from delivery_time.inference import DeliveryTimePredictor
from yelp_sentiment.inference import SentimentPredictor

from .settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelStatus:
    """Readiness information for one model family."""

    model: str
    loaded: bool
    path: Path
    error: str | None = None
    model_version: str | None = None
    selected_feature_set: str | None = None


class ModelRegistry:
    """Load models once and expose their readiness to the API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.delivery_time: DeliveryTimePredictor | None = None
        self.sentiment: SentimentPredictor | None = None
        self.delivery_time_status = ModelStatus(
            model="delivery-time",
            loaded=False,
            path=settings.delivery_model_path,
            error="Model has not been loaded yet.",
        )
        self.sentiment_status = ModelStatus(
            model="sentiment",
            loaded=False,
            path=settings.sentiment_model_path,
            error="Model has not been loaded yet.",
        )

    def load(self) -> None:
        """Load available models without crashing the application on failure."""
        self._load_delivery_time()
        self._load_sentiment()

    def _load_delivery_time(self) -> None:
        try:
            self.delivery_time = DeliveryTimePredictor.load(self.settings.delivery_model_path)
        except Exception as exc:
            self.delivery_time = None
            self.delivery_time_status = ModelStatus(
                model="delivery-time",
                loaded=False,
                path=self.settings.delivery_model_path,
                error=str(exc),
            )
            logger.exception("Could not load the delivery-time model")
            return

        self.delivery_time_status = ModelStatus(
            model="delivery-time",
            loaded=True,
            path=self.settings.delivery_model_path,
            model_version=self.delivery_time.model_version,
            selected_feature_set=self.delivery_time.selected_feature_set,
        )
        logger.info("Loaded delivery-time model from %s", self.settings.delivery_model_path)

    def _load_sentiment(self) -> None:
        try:
            self.sentiment = SentimentPredictor.load(
                self.settings.sentiment_model_path,
                self.settings.sentiment_tokenizer_path,
            )
        except Exception as exc:
            self.sentiment = None
            self.sentiment_status = ModelStatus(
                model="sentiment",
                loaded=False,
                path=self.settings.sentiment_model_path,
                error=str(exc),
            )
            logger.exception("Could not load the sentiment model")
            return

        self.sentiment_status = ModelStatus(
            model="sentiment",
            loaded=True,
            path=self.settings.sentiment_model_path,
            model_version=self.sentiment.model_version,
        )
        logger.info("Loaded sentiment model from %s", self.settings.sentiment_model_path)

    @property
    def delivery_time_ready(self) -> bool:
        """Return whether the delivery-time model is ready for prediction."""
        return self.delivery_time is not None

    @property
    def sentiment_ready(self) -> bool:
        """Return whether the sentiment model is ready for prediction."""
        return self.sentiment is not None

    @property
    def all_ready(self) -> bool:
        """Return whether every model family is ready."""
        return self.delivery_time_ready and self.sentiment_ready

    def statuses(self) -> list[ModelStatus]:
        """Return the status of every model family exposed by the API."""
        return [self.delivery_time_status, self.sentiment_status]
