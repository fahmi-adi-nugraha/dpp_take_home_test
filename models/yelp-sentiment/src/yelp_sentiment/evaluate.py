"""Evaluation helpers and final model comparison for Yelp sentiment."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from yelp_sentiment import config, features

logger = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """Raised when models or evaluation artifacts cannot be loaded."""


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation tables and artifact paths."""

    metrics: dict[str, Any]
    metrics_table: pl.DataFrame
    hyperparameters_table: pl.DataFrame
    per_class_table: pl.DataFrame
    confusion_table: pl.DataFrame
    predictions_table: pl.DataFrame
    artifact_paths: dict[str, str]


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """Return accuracy, macro F1, and weighted F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def per_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> pl.DataFrame:
    """Return per-class precision, recall, F1, and support."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(5)),
        output_dict=True,
        zero_division=0,
    )
    return pl.DataFrame(
        [
            {
                "label": label,
                "label_name": config.LABEL_MAP[label],
                "precision": float(report[str(label)]["precision"]),
                "recall": float(report[str(label)]["recall"]),
                "f1": float(report[str(label)]["f1-score"]),
                "support": int(report[str(label)]["support"]),
            }
            for label in range(5)
        ]
    )


def confusion_matrix_table(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> pl.DataFrame:
    """Return a long-form confusion matrix with readable labels."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    return pl.DataFrame(
        [
            {
                "true_label": true_label,
                "predicted_label": predicted_label,
                "true_name": config.LABEL_MAP[true_label],
                "predicted_name": config.LABEL_MAP[predicted_label],
                "count": int(matrix[true_label, predicted_label]),
            }
            for true_label in range(5)
            for predicted_label in range(5)
        ]
    )


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(f"{path.name}.part")
    part_path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(part_path, path)


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(f"{path.name}.part")
    frame.write_parquet(part_path)
    os.replace(part_path, path)


def load_logistic_model() -> tuple[Any, dict[str, Any]]:
    """Load the final logistic pipeline and its hyperparameters."""
    if not config.LOGISTIC_MODEL_PATH.is_file():
        raise EvaluationError(f"Logistic model not found: {config.LOGISTIC_MODEL_PATH}")
    if not config.LOGISTIC_BEST_PARAMS_PATH.is_file():
        raise EvaluationError(f"Logistic parameters not found: {config.LOGISTIC_BEST_PARAMS_PATH}")
    model = joblib.load(config.LOGISTIC_MODEL_PATH)
    params = json.loads(config.LOGISTIC_BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    return model, params


def load_bert_model() -> tuple[Any, Any, dict[str, Any]]:
    """Load the final BERT model, tokenizer, and hyperparameters."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not config.BERT_MODEL_DIR.is_dir():
        raise EvaluationError(f"BERT model not found: {config.BERT_MODEL_DIR}")
    if not config.BERT_BEST_PARAMS_PATH.is_file():
        raise EvaluationError(f"BERT parameters not found: {config.BERT_BEST_PARAMS_PATH}")
    model = AutoModelForSequenceClassification.from_pretrained(config.BERT_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_DIR)
    params = json.loads(config.BERT_BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    return model, tokenizer, params


def predict_logistic(model, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Predict classes and probabilities with the logistic pipeline."""
    probabilities = model.predict_proba(list(texts))
    return probabilities.argmax(axis=1), probabilities


def predict_bert(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int = config.BERT_EVAL_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict classes and probabilities with the BERT model."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions: list[int] = []
    probabilities: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            batch_probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        probabilities.append(batch_probabilities)
        predictions.extend(batch_probabilities.argmax(axis=1).tolist())

    return np.asarray(predictions), np.vstack(probabilities)


def _hyperparameters_table(model_params: dict[str, Any]) -> pl.DataFrame:
    rows = []
    for model_name, params in model_params.items():
        for parameter, value in params.items():
            rows.append(
                {
                    "model": model_name,
                    "parameter": parameter,
                    "value": json.dumps(value, default=str)
                    if isinstance(value, (dict, list, tuple))
                    else str(value),
                }
            )
    return pl.DataFrame(rows)


def _prediction_table(
    frame: pl.DataFrame,
    model_name: str,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> pl.DataFrame:
    return (
        frame.select(
            "review_id",
            "label",
            pl.col("text").str.slice(0, 240).alias("text_excerpt"),
        )
        .with_columns(
            pl.lit(model_name).alias("model"),
            pl.Series("predicted_label", y_pred),
            pl.Series("confidence", probabilities.max(axis=1)),
        )
        .with_columns(
            pl.col("label")
            .replace_strict(config.LABEL_MAP, return_dtype=pl.String)
            .alias("actual_name"),
            pl.col("predicted_label")
            .replace_strict(config.LABEL_MAP, return_dtype=pl.String)
            .alias("predicted_name"),
        )
    )


def evaluate_models() -> EvaluationResult:
    """Evaluate final logistic and BERT models on the untouched test split."""
    feature_frame = features.load_model_features()
    test_frame = feature_frame.filter(pl.col(config.SPLIT_COLUMN) == "test")
    if test_frame.height == 0:
        raise EvaluationError("The test split is empty.")

    texts = test_frame[config.TEXT_COLUMN].to_list()
    y_true = test_frame[config.TARGET_COLUMN].to_numpy()

    logistic_model, logistic_params = load_logistic_model()
    logistic_pred, logistic_probabilities = predict_logistic(logistic_model, texts)

    bert_model, bert_tokenizer, bert_params = load_bert_model()
    bert_max_length = int(bert_params.get("max_length", config.MAX_LENGTH_FINAL))
    bert_pred, bert_probabilities = predict_bert(
        bert_model,
        bert_tokenizer,
        texts,
        max_length=bert_max_length,
    )

    models = {
        "logistic_regression": {
            "params": logistic_params,
            "predictions": logistic_pred,
            "probabilities": logistic_probabilities,
        },
        "bert": {
            "params": bert_params,
            "predictions": bert_pred,
            "probabilities": bert_probabilities,
        },
    }

    metrics_rows = []
    per_class_frames = []
    confusion_frames = []
    prediction_frames = []
    metrics_payload: dict[str, Any] = {}
    for model_name, payload in models.items():
        predictions = payload["predictions"]
        params = payload["params"]
        metrics = classification_metrics(y_true, predictions)
        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "hyperparameters": json.dumps(params, default=str),
            }
        )
        per_class_frames.append(
            per_class_metrics(y_true, predictions).with_columns(pl.lit(model_name).alias("model"))
        )
        confusion_frames.append(
            confusion_matrix_table(y_true, predictions).with_columns(
                pl.lit(model_name).alias("model")
            )
        )
        prediction_frames.append(
            _prediction_table(
                test_frame,
                model_name,
                predictions,
                payload["probabilities"],
            )
        )
        metrics_payload[model_name] = {
            "hyperparameters": params,
            "metrics": metrics,
            "per_class": per_class_metrics(y_true, predictions).to_dicts(),
            "test_rows": test_frame.height,
        }

    metrics_table = pl.DataFrame(metrics_rows).sort("macro_f1", descending=True)
    hyperparameters_table = _hyperparameters_table(
        {name: payload["params"] for name, payload in models.items()}
    )
    per_class_table = pl.concat(per_class_frames)
    confusion_table = pl.concat(confusion_frames)
    predictions_table = pl.concat(prediction_frames)

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "test_rows": test_frame.height,
        "models": metrics_payload,
        "ranking": metrics_table.select("model", "accuracy", "macro_f1").to_dicts(),
    }

    _write_json(config.EVALUATION_METRICS_PATH, payload)
    _write_parquet(config.EVALUATION_METRICS_TABLE_PATH, metrics_table)
    _write_parquet(config.EVALUATION_HYPERPARAMETERS_PATH, hyperparameters_table)
    _write_parquet(config.EVALUATION_PER_CLASS_PATH, per_class_table)
    _write_parquet(config.EVALUATION_CONFUSION_PATH, confusion_table)
    _write_parquet(config.EVALUATION_PREDICTIONS_PATH, predictions_table)

    comparison_lines = [
        "# Yelp Sentiment Model Evaluation",
        "",
        "All metrics are computed on the untouched 50,000-review test split.",
        "",
        "| Model | Accuracy | Macro F1 | Weighted F1 | Hyperparameters |",
        "|---|---:|---:|---:|---|",
    ]
    for row in metrics_table.iter_rows(named=True):
        comparison_lines.append(
            f"| {row['model']} | {row['accuracy']:.2f} | "
            f"{row['macro_f1']:.2f} | {row['weighted_f1']:.2f} | "
            f"`{row['hyperparameters']}` |"
        )
    config.EVALUATION_COMPARISON_PATH.write_text(
        "\n".join(comparison_lines) + "\n",
        encoding="utf-8",
    )

    return EvaluationResult(
        metrics=payload,
        metrics_table=metrics_table,
        hyperparameters_table=hyperparameters_table,
        per_class_table=per_class_table,
        confusion_table=confusion_table,
        predictions_table=predictions_table,
        artifact_paths={
            "metrics": str(config.EVALUATION_METRICS_PATH),
            "metrics_table": str(config.EVALUATION_METRICS_TABLE_PATH),
            "hyperparameters": str(config.EVALUATION_HYPERPARAMETERS_PATH),
            "per_class": str(config.EVALUATION_PER_CLASS_PATH),
            "confusion": str(config.EVALUATION_CONFUSION_PATH),
            "predictions": str(config.EVALUATION_PREDICTIONS_PATH),
            "comparison": str(config.EVALUATION_COMPARISON_PATH),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Yelp sentiment models.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run final test evaluation."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        result = evaluate_models()
    except Exception:
        logger.exception("Evaluation failed")
        return 1
    for row in result.metrics_table.iter_rows(named=True):
        logger.info(
            "%s: accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f",
            row["model"],
            row["accuracy"],
            row["macro_f1"],
            row["weighted_f1"],
        )
    logger.info("Artifacts: %s", result.artifact_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
