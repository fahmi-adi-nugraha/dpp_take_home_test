"""Sentiment prediction placeholder route."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..schemas import SentimentRequest, SentimentResponse

router = APIRouter(tags=["prediction"])


@router.post(
    "/predict/sentiment",
    response_model=SentimentResponse,
    responses={501: {"description": "Sentiment model is not implemented yet."}},
)
def predict_sentiment(payload: SentimentRequest) -> SentimentResponse:
    """Return 501 until the Yelp sentiment model is integrated."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Sentiment inference is not available yet. "
            "The Yelp BERT model has not been integrated into the API."
        ),
    )
