"""Tests for API settings."""

from __future__ import annotations

from inference_api.settings import Settings


def test_sentiment_device_can_be_set_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("SENTIMENT_DEVICE", "cpu")

    settings = Settings()

    assert settings.sentiment_device == "cpu"


def test_sentiment_device_defaults_to_auto(monkeypatch) -> None:
    monkeypatch.delenv("SENTIMENT_DEVICE", raising=False)

    assert Settings().sentiment_device == "auto"
