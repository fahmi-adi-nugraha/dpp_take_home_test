"""Sentiment prediction route."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from yelp_sentiment.inference import SentimentInput, SentimentPredictor

from ..dependencies import get_sentiment_predictor
from ..schemas import SentimentRequest, SentimentResponse

router = APIRouter(tags=["prediction"])


@router.post("/predict/sentiment", response_model=SentimentResponse)
def predict_sentiment(
    payload: SentimentRequest,
    predictor: Annotated[SentimentPredictor, Depends(get_sentiment_predictor)],
) -> SentimentResponse:
    """Predict five-class sentiment from review text."""
    result = predictor.predict(SentimentInput(text=payload.text))
    return SentimentResponse(
        label=result.label,
        label_name=result.label_name,
        confidence=result.confidence,
        probabilities=result.probabilities_by_label,
        model=predictor.model_name,
        model_version=predictor.model_version,
    )
