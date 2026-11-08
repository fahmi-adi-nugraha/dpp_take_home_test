"""Inference helpers for the exported Yelp sentiment model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

from yelp_sentiment import config


@dataclass(frozen=True)
class SentimentInput:
    """A single review passed to the sentiment model."""

    text: str


@dataclass(frozen=True)
class SentimentPrediction:
    """Five-class sentiment prediction with label metadata."""

    label: int
    label_name: str
    confidence: float
    probabilities: tuple[float, ...]
    label_map: dict[int, str]

    @property
    def probabilities_by_label(self) -> dict[str, float]:
        """Return probabilities keyed by human-readable labels."""
        return {
            self.label_map[index]: probability
            for index, probability in enumerate(self.probabilities)
        }


def load_model_bundle(path: Path) -> dict[str, Any]:
    """Load the deployment bundle saved by the export command."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    required = {"state_dict", "model_config", "label_map", "max_length"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")
    return bundle


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve the requested inference device.

    ``auto`` uses CUDA when a usable CUDA runtime is present and otherwise
    falls back to CPU. ``cpu`` always forces CPU. ``cuda`` requires a usable
    CUDA device and raises an error instead of silently falling back.
    """
    normalized = preference.strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized not in {"auto", "cuda"}:
        raise ValueError(
            f"Unsupported device preference '{preference}'. Use 'auto', 'cpu', or 'cuda'."
        )

    try:
        cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception as exc:
        if normalized == "cuda":
            raise RuntimeError(
                "CUDA was requested but the CUDA runtime could not be checked."
            ) from exc
        return torch.device("cpu")

    if cuda_available:
        return torch.device("cuda")
    if normalized == "cuda":
        raise RuntimeError(
            "CUDA was requested but no usable CUDA device is available. "
            "Install a CUDA-enabled PyTorch build and verify the GPU driver, "
            "or use SENTIMENT_DEVICE=cpu."
        )
    return torch.device("cpu")


@dataclass
class SentimentPredictor:
    """Load and run the exported BERT sentiment model."""

    model: Any
    tokenizer: Any
    max_length: int
    label_map: dict[int, str]
    model_name: str
    model_version: str
    device: torch.device

    @classmethod
    def load(
        cls,
        model_path: Path,
        tokenizer_path: Path,
        *,
        device: str = "auto",
    ) -> SentimentPredictor:
        """Load a `.pt` bundle and its tokenizer."""
        if not model_path.is_file():
            raise FileNotFoundError(f"Sentiment model artifact not found: {model_path}")
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"Sentiment tokenizer not found: {tokenizer_path}")

        bundle = load_model_bundle(model_path)
        model_config = BertConfig.from_dict(bundle["model_config"])
        model = BertForSequenceClassification(model_config)
        model.load_state_dict(bundle["state_dict"])
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        selected_device = resolve_device(device)
        model.to(selected_device)
        model.eval()

        label_map = {int(index): str(label) for index, label in bundle["label_map"].items()}
        return cls(
            model=model,
            tokenizer=tokenizer,
            max_length=int(bundle["max_length"]),
            label_map=label_map,
            model_name=str(bundle.get("model_name", config.BERT_SEARCH_MODEL_NAME)),
            model_version=str(bundle.get("model_version", config.MODEL_VERSION)),
            device=selected_device,
        )

    def predict(self, payload: SentimentInput) -> SentimentPrediction:
        """Predict the five-class sentiment distribution for one review."""
        encoded = self.tokenizer(
            [payload.text],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        label = int(max(range(len(probabilities)), key=probabilities.__getitem__))
        return SentimentPrediction(
            label=label,
            label_name=self.label_map[label],
            confidence=float(probabilities[label]),
            probabilities=tuple(float(value) for value in probabilities),
            label_map=self.label_map,
        )
