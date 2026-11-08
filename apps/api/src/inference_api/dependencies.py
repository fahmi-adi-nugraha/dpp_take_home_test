"""FastAPI dependency providers."""

from __future__ import annotations

from typing import Annotated

from delivery_time.inference import DeliveryTimePredictor
from fastapi import Depends, HTTPException, Request, status
from yelp_sentiment.inference import SentimentPredictor

from .registry import ModelRegistry
from .settings import Settings


def get_settings(request: Request) -> Settings:
    """Return the settings stored on the application."""
    return request.app.state.settings


def get_registry(request: Request) -> ModelRegistry:
    """Return the model registry stored on the application."""
    return request.app.state.registry


def get_delivery_predictor(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> DeliveryTimePredictor:
    """Return the delivery predictor or a clear service-unavailable error."""
    if registry.delivery_time is None:
        detail = "Delivery-time model is unavailable."
        if registry.delivery_time_status.error:
            detail = f"{detail} {registry.delivery_time_status.error}"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )
    return registry.delivery_time


def get_sentiment_predictor(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> SentimentPredictor:
    """Return the sentiment predictor or a clear service-unavailable error."""
    if registry.sentiment is None:
        detail = "Sentiment model is unavailable."
        if registry.sentiment_status.error:
            detail = f"{detail} {registry.sentiment_status.error}"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )
    return registry.sentiment
