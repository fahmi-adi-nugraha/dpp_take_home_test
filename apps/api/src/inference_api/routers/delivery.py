"""Delivery-time prediction route."""

from __future__ import annotations

from typing import Annotated

from delivery_time.inference import DeliveryTimePredictor
from fastapi import APIRouter, Depends

from ..dependencies import get_delivery_predictor
from ..schemas import DeliveryTimeRequest, DeliveryTimeResponse

router = APIRouter(tags=["prediction"])


@router.post("/predict/delivery-time", response_model=DeliveryTimeResponse)
def predict_delivery_time(
    payload: DeliveryTimeRequest,
    predictor: Annotated[DeliveryTimePredictor, Depends(get_delivery_predictor)],
) -> DeliveryTimeResponse:
    """Predict delivery time in minutes from raw order inputs."""
    result = predictor.predict(payload.to_inference_input())
    return DeliveryTimeResponse(
        prediction_minutes=round(result.prediction_minutes, 2),
        model=predictor.model_name,
        model_version=predictor.model_version,
        selected_feature_set=predictor.selected_feature_set,
        warnings=list(result.warnings),
    )
