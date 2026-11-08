"""ASHA hyperparameter search and final Yelp sentiment training."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset

from yelp_sentiment import config, features
from yelp_sentiment.evaluate import classification_metrics, per_class_metrics

logger = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """Raised when a tuning or training run fails."""


@dataclass(frozen=True)
class TrainingResult:
    """Fine-tuning outputs."""

    logistic_params: dict[str, Any]
    bert_params: dict[str, Any]
    artifact_paths: dict[str, str]


class ReviewDataset(Dataset):
    """On-the-fly tokenized review dataset."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int,
    ) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.texts[index],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": int(self.labels[index]),
        }


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


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _load_model_frame() -> pl.DataFrame:
    return features.load_model_features()


def _search_train_validation() -> tuple[pl.DataFrame, pl.DataFrame]:
    frame = _load_model_frame()
    train = frame.filter(pl.col(config.SPLIT_COLUMN) == "train")
    validation = frame.filter(pl.col(config.SPLIT_COLUMN) == "validation")
    if train.height == 0 or validation.height == 0:
        raise TrainingError("Reduced train and validation splits are required.")
    return train, validation


def _logistic_params_from_config(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "tfidf_max_features": int(params["tfidf_max_features"]),
        "tfidf_ngram_range": str(params["tfidf_ngram_range"]),
        "tfidf_min_df": int(params["tfidf_min_df"]),
        "tfidf_max_df": float(params["tfidf_max_df"]),
        "tfidf_sublinear_tf": bool(params["tfidf_sublinear_tf"]),
        "clf_C": float(params["clf_C"]),
        "clf_fit_intercept": bool(params["clf_fit_intercept"]),
    }


def make_logistic_pipeline(params: dict[str, Any]) -> Pipeline:
    """Build the TF-IDF logistic regression pipeline."""
    ngram_range = (1, 2) if params["tfidf_ngram_range"] == "unigram_bigram" else (1, 1)
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=int(params["tfidf_max_features"]),
                    ngram_range=ngram_range,
                    min_df=int(params["tfidf_min_df"]),
                    max_df=float(params["tfidf_max_df"]),
                    sublinear_tf=bool(params["tfidf_sublinear_tf"]),
                    stop_words="english",
                    dtype=np.float32,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=float(params["clf_C"]),
                    solver="lbfgs",
                    max_iter=2000,
                    fit_intercept=bool(params["clf_fit_intercept"]),
                    class_weight=None,
                ),
            ),
        ]
    )


def _logistic_search_space():
    from ray import tune

    return {
        "tfidf_max_features": tune.choice([10_000, 20_000, 50_000, 100_000]),
        "tfidf_ngram_range": tune.choice(["unigram", "unigram_bigram"]),
        "tfidf_min_df": tune.choice([2, 3, 5, 10]),
        "tfidf_max_df": tune.choice([0.50, 0.75, 0.90]),
        "tfidf_sublinear_tf": tune.choice([True, False]),
        "clf_C": tune.loguniform(1e-3, 1e2),
        "clf_fit_intercept": tune.choice([True, False]),
    }


def _logistic_trainable(params: dict[str, Any]) -> None:
    from ray import tune

    train, validation = _search_train_validation()
    search_train = features.stratified_sample(
        train,
        config.LOGISTIC_SEARCH_TRAIN_ROWS,
        seed=config.ASHA_SEED,
    )
    validation_labels = validation[config.TARGET_COLUMN].to_numpy()
    validation_texts = validation[config.TEXT_COLUMN].to_list()

    for rung in range(1, config.ASHA_MAX_T + 1):
        n_rows = max(1, round(search_train.height * rung / config.ASHA_MAX_T))
        subset = features.stratified_sample(
            search_train,
            n_rows,
            seed=config.ASHA_SEED + rung,
        )
        pipeline = make_logistic_pipeline(_logistic_params_from_config(params))
        pipeline.fit(
            subset[config.TEXT_COLUMN].to_list(),
            subset[config.TARGET_COLUMN].to_numpy(),
        )
        predictions = pipeline.predict(validation_texts)
        metrics = classification_metrics(validation_labels, predictions)
        tune.report(
            {
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )


def _bert_params_from_config(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "learning_rate": float(params["learning_rate"]),
        "per_device_train_batch_size": int(params["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(params["gradient_accumulation_steps"]),
        "weight_decay": float(params["weight_decay"]),
        "warmup_ratio": float(params["warmup_ratio"]),
        "lr_scheduler_type": str(params["lr_scheduler_type"]),
        "label_smoothing_factor": float(params["label_smoothing_factor"]),
        "hidden_dropout_prob": float(params["hidden_dropout_prob"]),
        "attention_probs_dropout_prob": float(params["attention_probs_dropout_prob"]),
        "max_length": config.BERT_REDUCED_MAX_LENGTH,
        "num_train_epochs": config.BERT_REDUCED_EPOCHS,
    }


def _bert_search_space():
    from ray import tune

    return {
        "learning_rate": tune.loguniform(1e-5, 5e-5),
        "per_device_train_batch_size": tune.choice(list(config.BERT_TRAIN_BATCH_SIZES)),
        "gradient_accumulation_steps": tune.choice([1, 2, 4]),
        "weight_decay": tune.choice([0.0, 0.01, 0.10]),
        "warmup_ratio": tune.choice([0.0, 0.06, 0.10]),
        "lr_scheduler_type": tune.choice(["linear", "cosine"]),
        "label_smoothing_factor": tune.choice([0.0, 0.05, 0.10]),
        "hidden_dropout_prob": tune.choice([0.10, 0.20]),
        "attention_probs_dropout_prob": tune.choice([0.10, 0.20]),
    }


def _build_bert_model(params: dict[str, Any]):
    from transformers import AutoConfig, AutoModelForSequenceClassification

    model_config = AutoConfig.from_pretrained(config.BERT_SEARCH_MODEL_NAME)
    model_config.num_labels = 5
    model_config.hidden_dropout_prob = float(params["hidden_dropout_prob"])
    model_config.attention_probs_dropout_prob = float(params["attention_probs_dropout_prob"])
    return AutoModelForSequenceClassification.from_pretrained(
        config.BERT_SEARCH_MODEL_NAME,
        config=model_config,
    )


def _evaluate_bert(model, loader, device) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].tolist())
    return classification_metrics(labels, predictions)


def _train_bert_epochs(
    train_frame: pl.DataFrame,
    validation_frame: pl.DataFrame,
    params: dict[str, Any],
    *,
    report: bool,
) -> tuple[Any, list[dict[str, Any]]]:
    from transformers import (
        AutoTokenizer,
        DataCollatorWithPadding,
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(config.BERT_SEARCH_MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer)
    train_dataset = ReviewDataset(
        train_frame[config.TEXT_COLUMN].to_list(),
        train_frame[config.TARGET_COLUMN].to_list(),
        tokenizer,
        int(params["max_length"]),
    )
    validation_dataset = ReviewDataset(
        validation_frame[config.TEXT_COLUMN].to_list(),
        validation_frame[config.TARGET_COLUMN].to_list(),
        tokenizer,
        int(params["max_length"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(params["per_device_train_batch_size"]),
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.BERT_EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_bert_model(params)
    model.to(device)
    accumulation = int(params["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = steps_per_epoch * config.BERT_REDUCED_EPOCHS
    warmup_steps = int(total_steps * float(params["warmup_ratio"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    if params["lr_scheduler_type"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.BERT_REDUCED_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=torch.cuda.is_available(),
            ):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=float(params["label_smoothing_factor"]),
                )
            scaler.scale(loss / accumulation).backward()
            if step % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            epoch_loss += float(loss.detach().cpu()) * input_ids.size(0)

        if len(train_loader) % accumulation != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        metrics = _evaluate_bert(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_dataset),
            **metrics,
        }
        history.append(row)
        logger.info("BERT epoch %d: %s", epoch, metrics)
        if report:
            from ray import tune

            tune.report(
                {
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "weighted_f1": metrics["weighted_f1"],
                }
            )
    return model, history


def _bert_trainable(params: dict[str, Any]) -> None:
    train, validation = _search_train_validation()
    search_train = features.stratified_sample(
        train,
        config.BERT_SEARCH_TRAIN_ROWS,
        seed=config.ASHA_SEED,
    )
    search_validation = features.stratified_sample(
        validation,
        config.BERT_SEARCH_VALIDATION_ROWS,
        seed=config.ASHA_SEED + 1,
    )
    params = _bert_params_from_config(params)
    _model, _history = _train_bert_epochs(
        search_train,
        search_validation,
        params,
        report=True,
    )
    torch.cuda.empty_cache()


def _asha_scheduler():
    from ray.tune.schedulers import ASHAScheduler

    return ASHAScheduler(
        time_attr="training_iteration",
        metric="macro_f1",
        mode="max",
        max_t=config.ASHA_MAX_T,
        grace_period=config.ASHA_GRACE_PERIOD,
        reduction_factor=config.ASHA_REDUCTION_FACTOR,
        brackets=1,
    )


def _results_table(results) -> pl.DataFrame:
    rows = []
    for result in results:
        metrics = result.metrics or {}
        rows.append(
            {
                "trial": str(result.path),
                "training_iteration": metrics.get("training_iteration"),
                "macro_f1": metrics.get("macro_f1"),
                "accuracy": metrics.get("accuracy"),
                "weighted_f1": metrics.get("weighted_f1"),
                "config_json": json.dumps(result.config, default=str),
                "error": str(result.error) if result.error else None,
            }
        )
    return pl.DataFrame(rows)


def _run_asha(
    trainable,
    *,
    search_space: dict[str, Any],
    num_samples: int,
    resources: dict[str, float],
    name: str,
) -> tuple[dict[str, Any], pl.DataFrame, float]:
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    import ray
    from ray import tune
    from ray.tune import RunConfig

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            runtime_env={"py_executable": sys.executable},
        )
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources=resources),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=_asha_scheduler(),
            max_concurrent_trials=1 if resources.get("gpu", 0) else 4,
        ),
        run_config=RunConfig(
            name=name,
            storage_path=str(config.ARTIFACTS_DIR / "ray_results"),
        ),
    )
    results = tuner.fit()
    try:
        best_result = results.get_best_result(metric="macro_f1", mode="max")
    except Exception as exc:
        raise TrainingError(f"{name} produced no successful trials.") from exc
    if best_result is None or not best_result.metrics or best_result.config is None:
        raise TrainingError(f"{name} produced no successful trials.")
    return (
        dict(best_result.config),
        _results_table(results),
        float(best_result.metrics["macro_f1"]),
    )


def tune_logistic() -> tuple[dict[str, Any], pl.DataFrame, float]:
    """Run ASHA for the TF-IDF logistic baseline."""
    return _run_asha(
        _logistic_trainable,
        search_space=_logistic_search_space(),
        num_samples=config.LOGISTIC_TRIALS,
        resources={"cpu": 2},
        name="logistic_asha",
    )


def tune_bert() -> tuple[dict[str, Any], pl.DataFrame, float]:
    """Run ASHA for BERT on the reduced search subset."""
    return _run_asha(
        _bert_trainable,
        search_space=_bert_search_space(),
        num_samples=config.BERT_TRIALS,
        resources={"gpu": 1, "cpu": 2},
        name="bert_asha",
    )


def train_logistic(params: dict[str, Any], best_search_macro_f1: float) -> dict[str, Any]:
    """Fit validation and final logistic models with the best ASHA config."""
    train, validation = _search_train_validation()
    train_texts = train[config.TEXT_COLUMN].to_list()
    train_labels = train[config.TARGET_COLUMN].to_numpy()
    validation_texts = validation[config.TEXT_COLUMN].to_list()
    validation_labels = validation[config.TARGET_COLUMN].to_numpy()

    validation_pipeline = make_logistic_pipeline(params)
    validation_pipeline.fit(train_texts, train_labels)
    validation_predictions = validation_pipeline.predict(validation_texts)
    validation_metrics = classification_metrics(
        validation_labels,
        validation_predictions,
    )

    train_validation = pl.concat([train, validation])
    final_pipeline = make_logistic_pipeline(params)
    final_pipeline.fit(
        train_validation[config.TEXT_COLUMN].to_list(),
        train_validation[config.TARGET_COLUMN].to_numpy(),
    )

    config.LOGISTIC_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, config.LOGISTIC_MODEL_PATH)
    _write_json(config.LOGISTIC_BEST_PARAMS_PATH, params)
    _write_json(
        config.LOGISTIC_VALIDATION_METRICS_PATH,
        {
            "model": "logistic_regression",
            "params": params,
            "validation_rows": validation.height,
            "metrics": validation_metrics,
            "per_class": per_class_metrics(
                validation_labels,
                validation_predictions,
            ).to_dicts(),
            "final_fit_rows": train_validation.height,
            "best_search_macro_f1": best_search_macro_f1,
        },
    )
    return params


def train_bert(
    params: dict[str, Any],
    *,
    best_search_macro_f1: float,
    trials: pl.DataFrame,
) -> dict[str, Any]:
    """Refit BERT on the reduced train + validation set."""
    from transformers import AutoTokenizer

    params = _bert_params_from_config(params)
    train, validation = _search_train_validation()
    train_validation = pl.concat([train, validation])
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_SEARCH_MODEL_NAME)
    model, history = _train_bert_epochs(
        train_validation,
        validation,
        params,
        report=False,
    )
    config.BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.BERT_MODEL_DIR)
    tokenizer.save_pretrained(config.BERT_MODEL_DIR)
    _write_json(
        config.BERT_TRAINING_HISTORY_PATH,
        {"params": params, "history": history},
    )
    _write_json(config.BERT_BEST_PARAMS_PATH, params)
    _write_json(
        config.BERT_VALIDATION_METRICS_PATH,
        {
            "model": "bert",
            "params": params,
            "search_validation_rows": config.BERT_SEARCH_VALIDATION_ROWS,
            "best_search_macro_f1": best_search_macro_f1,
            "final_fit_rows": train_validation.height,
        },
    )
    _write_parquet(config.BERT_TUNING_TRIALS_PATH, trials)

    from yelp_sentiment.export import export_bert_model

    export_bert_model(
        model,
        tokenizer,
        evaluation_path=None,
    )
    torch.cuda.empty_cache()
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Yelp sentiment models.")
    parser.add_argument(
        "--model",
        choices=["all", "logistic", "bert"],
        default="all",
        help="Model family to tune and fit.",
    )
    parser.add_argument("--logistic-trials", type=int, default=config.LOGISTIC_TRIALS)
    parser.add_argument("--bert-trials", type=int, default=config.BERT_TRIALS)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run ASHA tuning, final refits, and artifact export."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        paths: dict[str, str] = {}
        logistic_params: dict[str, Any] = {}
        bert_params: dict[str, Any] = {}

        if args.model in {"all", "logistic"}:
            logger.info("Running ASHA for logistic regression")
            config.LOGISTIC_TRIALS = args.logistic_trials
            best_params, trials, best_value = tune_logistic()
            logistic_params = train_logistic(best_params, best_value)
            _write_parquet(config.LOGISTIC_TUNING_TRIALS_PATH, trials)
            paths["logistic_best_params"] = str(config.LOGISTIC_BEST_PARAMS_PATH)
            paths["logistic_model"] = str(config.LOGISTIC_MODEL_PATH)

        if args.model in {"all", "bert"}:
            logger.info("Running ASHA for BERT")
            config.BERT_TRIALS = args.bert_trials
            best_params, trials, best_value = tune_bert()
            bert_params = train_bert(
                best_params,
                best_search_macro_f1=best_value,
                trials=trials,
            )
            paths["bert_best_params"] = str(config.BERT_BEST_PARAMS_PATH)
            paths["bert_model"] = str(config.BERT_MODEL_DIR)

        logger.info("Logistic params: %s", logistic_params)
        logger.info("BERT params: %s", bert_params)
        logger.info("Artifacts: %s", paths)
        return 0
    except Exception:
        logger.exception("Training failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
