"""Local FastAPI inference service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .registry import ModelRegistry
from .routers import delivery, health, sentiment
from .settings import Settings


def create_app(
    settings: Settings | None = None,
    registry: ModelRegistry | None = None,
) -> FastAPI:
    """Create the FastAPI application with injectable startup state."""
    app_settings = settings or Settings()
    app_registry = registry or ModelRegistry(app_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=app_settings.log_level.upper())
        app_registry.load()
        yield

    app = FastAPI(
        title="DPP Inference API",
        version="0.1.0",
        description="Local inference service for the DPP take-home models.",
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.registry = app_registry
    app.include_router(health.router)
    app.include_router(delivery.router)
    app.include_router(sentiment.router)
    return app


app = create_app()
