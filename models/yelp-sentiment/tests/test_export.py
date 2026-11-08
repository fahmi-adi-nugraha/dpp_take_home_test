"""Tests for the Yelp sentiment export bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from yelp_sentiment.export import export_bert_model
from yelp_sentiment.inference import SentimentInput, SentimentPredictor


def _tiny_model(vocab_size: int) -> BertForSequenceClassification:
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=5,
    )
    return BertForSequenceClassification(config)


def test_export_writes_a_loadable_bundle(tmp_path: Path) -> None:
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "great",
        "food",
        "bad",
        "delivery",
        "fast",
        "cold",
    ]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
    model = _tiny_model(len(vocab))

    output_path = tmp_path / "model.pt"
    tokenizer_dir = tmp_path / "tokenizer"
    manifest_path = tmp_path / "export_manifest.json"
    paths = export_bert_model(
        model,
        tokenizer,
        output_path=output_path,
        tokenizer_dir=tokenizer_dir,
        manifest_path=manifest_path,
        evaluation_path=None,
    )

    assert Path(paths["model"]) == output_path
    assert output_path.is_file()
    assert tokenizer_dir.is_dir()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_name"] == "bert-base-uncased"
    assert manifest["max_length"] == 256
    assert manifest["label_map"]["4"] == "5 stars"

    bundle = torch.load(output_path, map_location="cpu", weights_only=True)
    assert bundle["format_version"] == 1
    assert "state_dict" in bundle
    assert "model_config" in bundle

    predictor = SentimentPredictor.load(output_path, tokenizer_dir, device="cpu")
    result = predictor.predict(SentimentInput(text="great food"))
    assert result.label in range(5)
    assert pytest.approx(sum(result.probabilities)) == 1.0
