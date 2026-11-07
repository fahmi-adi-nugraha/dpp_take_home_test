"""Health and model-status routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ..dependencies import get_registry
from ..registry import ModelRegistry, ModelStatus
from ..schemas import HealthResponse, ModelsResponse, ModelStatusResponse

router = APIRouter(tags=["health"])


def _status_response(status: ModelStatus) -> ModelStatusResponse:
    return ModelStatusResponse(
        model=status.model,
        loaded=status.loaded,
        path=str(status.path),
        error=status.error,
        model_version=status.model_version,
        selected_feature_set=status.selected_feature_set,
    )


@router.get("/health", response_model=HealthResponse)
def health(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> HealthResponse:
    """Report service health and model readiness."""
    return HealthResponse(
        status="ok" if registry.delivery_time_ready else "degraded",
        models=[_status_response(status) for status in registry.statuses()],
    )


@router.get("/models", response_model=ModelsResponse)
def models(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> ModelsResponse:
    """List the loaded and unavailable model families."""
    return ModelsResponse(models=[_status_response(status) for status in registry.statuses()])
