"""Tests for sentiment inference helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from yelp_sentiment import config
from yelp_sentiment.inference import SentimentInput, SentimentPredictor


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }


class FakeModel:
    def __call__(self, **kwargs):
        return SimpleNamespace(logits=torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]]))


def test_predict_returns_probabilities_and_label() -> None:
    predictor = SentimentPredictor(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        max_length=16,
        label_map={
            0: "1 star",
            1: "2 star",
            2: "3 stars",
            3: "4 stars",
            4: "5 stars",
        },
        model_name="tiny-bert",
        model_version="1",
        device=torch.device("cpu"),
    )

    result = predictor.predict(SentimentInput(text="great food"))

    assert result.label == 4
    assert result.label_name == "5 stars"
    assert result.confidence == pytest.approx(0.6364, rel=1e-3)
    assert len(result.probabilities) == 5
    assert sum(result.probabilities) == pytest.approx(1.0)
    assert result.probabilities_by_label["5 stars"] == pytest.approx(result.confidence)


def test_predictor_loads_the_exported_artifact_if_available() -> None:
    if not config.BERT_EXPORT_PATH.is_file() or not config.BERT_TOKENIZER_DIR.is_dir():
        pytest.skip("The exported sentiment artifact is not available.")

    predictor = SentimentPredictor.load(
        config.BERT_EXPORT_PATH,
        config.BERT_TOKENIZER_DIR,
        device="cpu",
    )
    result = predictor.predict(SentimentInput(text="The food was great."))

    assert result.label in range(5)
    assert result.label_name in config.LABEL_MAP.values()
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.probabilities) == 5
