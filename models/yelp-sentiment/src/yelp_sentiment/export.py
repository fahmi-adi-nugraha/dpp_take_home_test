"""Export the trained BERT sentiment model as a deployment bundle."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from yelp_sentiment import config
from yelp_sentiment.inference import SentimentInput, SentimentPredictor

_PARITY_TEXTS = (
    "The food was great and the delivery was fast.",
    "The order arrived late and the food was cold.",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(f"{path.name}.part")
    part_path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(part_path, path)


def _probabilities(model, tokenizer, texts: tuple[str, ...], max_length: int) -> list[list[float]]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).tolist()
    return probabilities


def export_bert_model(
    model=None,
    tokenizer=None,
    *,
    model_dir: Path = config.BERT_MODEL_DIR,
    output_path: Path = config.BERT_EXPORT_PATH,
    tokenizer_dir: Path = config.BERT_TOKENIZER_DIR,
    manifest_path: Path = config.BERT_EXPORT_MANIFEST_PATH,
    evaluation_path: Path | None = config.EVALUATION_METRICS_PATH,
    verify: bool = True,
) -> dict[str, str]:
    """Export the HF checkpoint into a state-dict `.pt` deployment bundle."""
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.to("cpu")
    model.eval()
    training_params = _read_json(config.BERT_BEST_PARAMS_PATH)
    max_length = int(training_params.get("max_length", config.BERT_REDUCED_MAX_LENGTH))
    evaluation = _read_json(evaluation_path) if evaluation_path is not None else {}
    test_metrics = evaluation.get("models", {}).get("bert", {}).get("metrics")

    bundle = {
        "format_version": 1,
        "model_name": config.BERT_SEARCH_MODEL_NAME,
        "model_version": config.MODEL_VERSION,
        "model_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "max_length": max_length,
        "label_map": dict(config.LABEL_MAP),
        "training_params": training_params,
        "test_metrics": test_metrics,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_name(f"{output_path.name}.part")
    torch.save(bundle, part_path)
    os.replace(part_path, output_path)
    tokenizer.save_pretrained(tokenizer_dir)

    _write_json(
        manifest_path,
        {
            "format_version": bundle["format_version"],
            "model_name": bundle["model_name"],
            "model_version": bundle["model_version"],
            "output_path": str(output_path),
            "tokenizer_dir": str(tokenizer_dir),
            "max_length": max_length,
            "label_map": bundle["label_map"],
            "training_params": training_params,
            "test_metrics": test_metrics,
            "created_at_utc": bundle["created_at_utc"],
        },
    )

    if verify:
        source_probabilities = _probabilities(model, tokenizer, _PARITY_TEXTS, max_length)
        exported = SentimentPredictor.load(
            output_path,
            tokenizer_dir,
            device="cpu",
        )
        exported_probabilities = [
            list(exported.predict(SentimentInput(text=text)).probabilities)
            for text in _PARITY_TEXTS
        ]
        if not torch.allclose(
            torch.tensor(source_probabilities),
            torch.tensor(exported_probabilities),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError("Exported model probabilities do not match the source model.")

    return {
        "model": str(output_path),
        "tokenizer": str(tokenizer_dir),
        "manifest": str(manifest_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the Yelp BERT sentiment model.")
    parser.add_argument("--model-dir", type=Path, default=config.BERT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=config.BERT_EXPORT_PATH)
    parser.add_argument("--tokenizer-dir", type=Path, default=config.BERT_TOKENIZER_DIR)
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the source/export probability parity check.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = export_bert_model(
        model_dir=args.model_dir,
        output_path=args.output,
        tokenizer_dir=args.tokenizer_dir,
        verify=not args.skip_verify,
    )
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
