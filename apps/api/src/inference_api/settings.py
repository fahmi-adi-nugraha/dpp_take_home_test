"""Application settings for the local inference API."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Environment-driven API configuration."""

    model_config = SettingsConfigDict(env_ignore_empty=True, extra="ignore")

    delivery_model_path: Path = REPO_ROOT / "models/delivery-time/artifacts/model.joblib"
    sentiment_model_path: Path = REPO_ROOT / "models/yelp-sentiment/artifacts/bert/model.pt"
    sentiment_tokenizer_path: Path = REPO_ROOT / "models/yelp-sentiment/artifacts/bert/tokenizer"
    sentiment_device: Literal["auto", "cpu", "cuda"] = "auto"
    log_level: str = "INFO"
