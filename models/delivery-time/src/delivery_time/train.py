"""Training, tuning, ablation, and artifact export for delivery time."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from delivery_time import config, features
from delivery_time.data import sha256_file
from delivery_time.evaluate import (
    permutation_importance_table,
    prediction_frame,
    regression_metrics,
)

logger = logging.getLogger(__name__)

TARGET = config.TARGET_FEATURE_NAME

DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "tree_method": "hist",
    "n_jobs": 1,
    "random_state": config.RANDOM_SEED,
}

ABLATION_XGB_PARAMS = {
    **DEFAULT_XGB_PARAMS,
    "n_estimators": 300,
}


@dataclass(frozen=True)
class DatasetSplit:
    """Train, validation, and test frames with metadata."""

    train: pl.DataFrame
    validation: pl.DataFrame
    test: pl.DataFrame
    seed: int
    strategy: str


@dataclass(frozen=True)
class AblationSpec:
    """A named feature set or row filter to evaluate."""

    name: str
    feature_names: tuple[str, ...]
    description: str
    eligible_for_final: bool = True
    drop_invalid_coordinates: bool = False


@dataclass(frozen=True)
class TrainingResult:
    """Materials produced by a full training run."""

    split: DatasetSplit
    metrics: dict[str, Any]
    ablation_results: pl.DataFrame
    selected_spec: AblationSpec
    xgb_search: RandomizedSearchCV
    final_pipeline: Pipeline
    test_predictions: pl.DataFrame
    permutation_importance: pl.DataFrame
    artifact_paths: dict[str, str]


def load_feature_data(path: Path = config.FEATURES_DATA_PARQUET) -> pl.DataFrame:
    """Load the engineered feature frame."""
    return pl.read_parquet(path)


def split_dataset(
    frame: pl.DataFrame,
    *,
    train_fraction: float = config.TRAIN_FRACTION,
    validation_fraction: float = config.VALIDATION_FRACTION,
    test_fraction: float = config.TEST_FRACTION,
    seed: int = config.RANDOM_SEED,
    time_based: bool = False,
) -> DatasetSplit:
    """Create a reproducible 80/10/10 split."""
    if not math.isclose(
        train_fraction + validation_fraction + test_fraction,
        1.0,
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError("Split fractions must sum to 1.0")

    if time_based:
        ordered = frame.sort("order_date")
        n_rows = ordered.height
        train_end = round(n_rows * train_fraction)
        validation_end = round(n_rows * (train_fraction + validation_fraction))
        return DatasetSplit(
            train=ordered[:train_end],
            validation=ordered[train_end:validation_end],
            test=ordered[validation_end:],
            seed=seed,
            strategy="time",
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(frame.height)
    n_train = round(frame.height * train_fraction)
    n_validation = round(frame.height * validation_fraction)
    train_indices = np.sort(shuffled[:n_train])
    validation_indices = np.sort(shuffled[n_train : n_train + n_validation])
    test_indices = np.sort(shuffled[n_train + n_validation :])
    return DatasetSplit(
        train=frame[train_indices],
        validation=frame[validation_indices],
        test=frame[test_indices],
        seed=seed,
        strategy="random",
    )


def validate_feature_names(feature_names: tuple[str, ...]) -> None:
    """Reject unknown, metadata, and leakage fields."""
    if TARGET in feature_names:
        raise ValueError("Target cannot be a feature")
    excluded = set(feature_names) & set(features.EXCLUDED_FEATURES)
    if excluded:
        raise ValueError(f"Excluded features requested: {sorted(excluded)}")
    unknown = set(feature_names) - set(features.ALL_FEATURES)
    if unknown:
        raise ValueError(f"Unknown feature names: {sorted(unknown)}")


def build_xy(
    frame: pl.DataFrame,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a NumPy feature matrix and target vector from Polars."""
    validate_feature_names(feature_names)
    x = frame.select(feature_names).to_numpy()
    y = frame.select(TARGET).cast(pl.Float64).to_numpy().ravel()
    return x, y


def _to_float(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def make_preprocessor(
    feature_names: tuple[str, ...],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build preprocessing for numeric and categorical feature groups."""
    categorical = tuple(name for name in feature_names if name in features.CATEGORICAL_FEATURES)
    numeric = tuple(name for name in feature_names if name not in categorical)
    numeric_indices = [feature_names.index(name) for name in numeric]
    categorical_indices = [feature_names.index(name) for name in categorical]

    if scale_numeric:
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = Pipeline([("to_float", FunctionTransformer(_to_float, validate=False))])

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if numeric_indices:
        transformers.append(("numeric", numeric_pipeline, numeric_indices))
    if categorical_indices:
        transformers.append(("categorical", categorical_pipeline, categorical_indices))
    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_linear_pipeline(
    feature_names: tuple[str, ...],
    *,
    alpha: float = 1.0,
) -> Pipeline:
    """Build a scaled Ridge regression pipeline."""
    return Pipeline(
        [
            ("preprocess", make_preprocessor(feature_names, scale_numeric=True)),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def make_xgboost_pipeline(
    feature_names: tuple[str, ...],
    params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build an XGBoost regression pipeline."""
    model_params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    return Pipeline(
        [
            ("preprocess", make_preprocessor(feature_names, scale_numeric=False)),
            ("model", XGBRegressor(**model_params)),
        ]
    )


def _metrics_for_model(
    model,
    frame: pl.DataFrame,
    feature_names: tuple[str, ...],
) -> dict[str, float]:
    x, y = build_xy(frame, feature_names)
    return regression_metrics(y, model.predict(x))


def tune_linear(
    split: DatasetSplit,
    feature_names: tuple[str, ...],
    *,
    n_iter: int = config.LINEAR_SEARCH_ITERATIONS,
) -> RandomizedSearchCV:
    """Randomized alpha search for the Ridge baseline."""
    x_train, y_train = build_xy(split.train, feature_names)
    search = RandomizedSearchCV(
        make_linear_pipeline(feature_names),
        {"model__alpha": loguniform(1e-3, 1e3)},
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=KFold(
            n_splits=config.CV_FOLDS,
            shuffle=True,
            random_state=config.RANDOM_SEED,
        ),
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    return search


def tune_xgboost(
    split: DatasetSplit,
    feature_names: tuple[str, ...],
    *,
    n_iter: int = config.XGBOOST_SEARCH_ITERATIONS,
) -> RandomizedSearchCV:
    """Randomized hyperparameter search for XGBoost."""
    x_train, y_train = build_xy(split.train, feature_names)
    search = RandomizedSearchCV(
        make_xgboost_pipeline(feature_names),
        {
            "model__n_estimators": randint(200, 800),
            "model__max_depth": randint(2, 9),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.4, 0.6),
            "model__min_child_weight": randint(1, 20),
            "model__reg_alpha": loguniform(1e-3, 10.0),
            "model__reg_lambda": loguniform(1e-3, 10.0),
            "model__gamma": loguniform(1e-3, 5.0),
        },
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=KFold(
            n_splits=config.CV_FOLDS,
            shuffle=True,
            random_state=config.RANDOM_SEED,
        ),
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    return search


def _base_features() -> tuple[str, ...]:
    return tuple(features.BASELINE_FEATURES)


def _with(*extra: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*features.BASELINE_FEATURES, *extra]))


def build_ablation_specs() -> tuple[AblationSpec, ...]:
    """Return the feature-set and row-filter ablations."""
    baseline = _base_features()
    return (
        AblationSpec(
            "baseline",
            baseline,
            "Checklist baseline feature set.",
        ),
        AblationSpec(
            "no_distance",
            tuple(name for name in baseline if name != "distance_km"),
            "Drop distance while keeping the coordinate-validity flag.",
        ),
        AblationSpec(
            "drop_invalid_coordinates",
            baseline,
            "Drop rows with invalid coordinate pairs.",
            eligible_for_final=False,
            drop_invalid_coordinates=True,
        ),
        AblationSpec(
            "distance_raw",
            tuple(name for name in baseline if name != "distance_km") + ("distance_km_raw",),
            "Use unrepaired restaurant coordinates for distance.",
            eligible_for_final=False,
        ),
        AblationSpec(
            "multiple_count_only",
            tuple(name for name in baseline if name != "has_multiple_deliveries"),
            "Use the multiple-delivery count without the binary flag.",
        ),
        AblationSpec(
            "multiple_binary_only",
            tuple(name for name in baseline if name != "multiple_deliveries"),
            "Use the multiple-delivery binary flag without the count.",
        ),
        AblationSpec(
            "hour_numeric_only",
            tuple(name for name in baseline if name not in {"is_rush_hour", "is_meal_period"}),
            "Use numeric order hour without the derived time-window flags.",
        ),
        AblationSpec(
            "hour_with_cyclic",
            _with("order_hour_sin", "order_hour_cos"),
            "Add cyclical hour encodings to the baseline.",
        ),
        AblationSpec(
            "hour_onehot",
            tuple(name for name in baseline if name != "order_hour") + ("order_hour_label",),
            "Replace numeric hour with a categorical hour label.",
        ),
        AblationSpec(
            "no_high_signal_missingness",
            tuple(name for name in baseline if name not in {"city_missing", "multiple_missing"}),
            "Drop the city and multiple-delivery missingness indicators.",
            eligible_for_final=False,
        ),
        AblationSpec(
            "plus_vehicle_type",
            _with("vehicle_type"),
            "Add vehicle type.",
        ),
        AblationSpec(
            "plus_weather_group",
            _with("weather_group"),
            "Add grouped weather categories.",
        ),
        AblationSpec(
            "plus_day_of_week",
            _with("day_of_week", "is_weekend"),
            "Add day-of-week and weekend indicators.",
        ),
        AblationSpec(
            "plus_coord_status",
            _with("coord_status", "restaurant_coords_sign_flipped"),
            "Add coordinate status and sign-flipped flags.",
        ),
        AblationSpec(
            "plus_festival_missing",
            _with("festival_missing"),
            "Add the suspicious festival-missingness flag.",
        ),
        AblationSpec(
            "plus_missing_indicators",
            _with("age_missing", "rating_missing", "weather_missing", "traffic_missing"),
            "Add the lower-signal missingness indicators.",
        ),
        AblationSpec(
            "clipped_courier",
            tuple(name for name in baseline if name not in {"age", "rating"})
            + (
                "age_clipped",
                "rating_clipped",
                "age_under_18",
                "rating_out_of_range",
            ),
            "Replace raw courier age and rating with clipped variants and flags.",
        ),
        AblationSpec(
            "plus_vehicle_condition_is_zero",
            _with("vehicle_condition_is_zero"),
            "Add the vehicle-condition-zero flag.",
        ),
        AblationSpec(
            "plus_distance_decile",
            _with("distance_decile"),
            "Add the distance decile.",
        ),
        AblationSpec(
            "plus_order_type",
            _with("order_type"),
            "Add order type.",
        ),
        AblationSpec(
            "plus_courier_count",
            _with("courier_delivery_count"),
            "Add courier delivery count.",
        ),
    )


def run_ablations(
    split: DatasetSplit,
    specs: tuple[AblationSpec, ...],
    *,
    params: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Fit each ablation specification and score it on validation."""
    records = []
    for spec in specs:
        train = split.train
        validation = split.validation
        if spec.drop_invalid_coordinates:
            train = train.filter(~pl.col("coordinates_invalid"))
            validation = validation.filter(~pl.col("coordinates_invalid"))

        x_train, y_train = build_xy(train, spec.feature_names)
        x_validation, y_validation = build_xy(validation, spec.feature_names)
        model = make_xgboost_pipeline(spec.feature_names, params)
        model.fit(x_train, y_train)
        metrics = regression_metrics(
            y_validation,
            model.predict(x_validation),
        )
        records.append(
            {
                "name": spec.name,
                "description": spec.description,
                "n_features": len(spec.feature_names),
                "n_train": train.height,
                "n_validation": validation.height,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "eligible_for_final": spec.eligible_for_final,
            }
        )
        logger.info("Ablation %s: validation RMSE %.4f", spec.name, metrics["rmse"])

    results = pl.DataFrame(records)
    baseline_rmse = results.filter(pl.col("name") == "baseline")["rmse"].item()
    return results.with_columns(
        (pl.col("rmse") - baseline_rmse).alias("delta_rmse_vs_baseline")
    ).sort("rmse")


def select_best_ablation(
    results: pl.DataFrame,
    specs: tuple[AblationSpec, ...],
    *,
    min_improvement: float = 0.01,
) -> AblationSpec:
    """Choose the best eligible ablation, preferring the checklist baseline."""
    baseline_rmse = results.filter(pl.col("name") == "baseline")["rmse"].item()
    eligible = results.filter(pl.col("eligible_for_final")).sort("rmse")
    best = eligible.row(0, named=True)
    if best["name"] == "baseline" or baseline_rmse - best["rmse"] < min_improvement:
        selected_name = "baseline"
    else:
        selected_name = best["name"]
    return next(spec for spec in specs if spec.name == selected_name)


def _package_versions() -> dict[str, str]:
    packages = [
        "scikit-learn",
        "xgboost",
        "polars",
        "numpy",
        "joblib",
        "plotnine",
        "marimo",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(f"{path.name}.part")
    part_path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(part_path, path)


def _feature_schema(
    frame: pl.DataFrame,
    feature_names: tuple[str, ...],
    selected_spec: AblationSpec,
) -> dict[str, Any]:
    categorical = [name for name in feature_names if name in features.CATEGORICAL_FEATURES]
    numeric = [name for name in feature_names if name not in categorical]
    categories = {}
    for name in categorical:
        values = frame[name].drop_nulls().unique().sort().to_list()
        categories[name] = [str(value) for value in values]
    return {
        "target": TARGET,
        "prediction_point": "order_placement",
        "feature_set": selected_spec.name,
        "feature_set_description": selected_spec.description,
        "features": [{"name": name, "dtype": str(frame.schema[name])} for name in feature_names],
        "categorical_features": categorical,
        "numeric_features": numeric,
        "categorical_levels": categories,
        "missing_value_policy": (
            "XGBoost numeric features keep nulls; categorical features are "
            "filled with a literal 'missing' level before one-hot encoding."
        ),
    }


def write_training_artifacts(
    *,
    final_pipeline: Pipeline,
    metrics: dict[str, Any],
    ablation_results: pl.DataFrame,
    test_predictions: pl.DataFrame,
    permutation_importance: pl.DataFrame,
    feature_schema: dict[str, Any],
    model_metadata: dict[str, Any],
) -> dict[str, str]:
    """Persist the trained pipeline and experiment outputs."""
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, config.MODEL_PATH)
    ablation_results.write_parquet(config.ABLATION_RESULTS_PARQUET)
    ablation_results.write_csv(config.ABLATION_RESULTS_CSV)
    test_predictions.write_parquet(config.TEST_PREDICTIONS_PATH)
    _write_json(config.METRICS_PATH, metrics)
    _write_json(config.FEATURE_SCHEMA_PATH, feature_schema)
    _write_json(config.MODEL_METADATA_PATH, model_metadata)
    permutation_importance.write_csv(config.ARTIFACTS_DIR / "permutation_importance.csv")

    final_metrics = metrics["models"]["final_xgboost"]["test"]
    config.MODEL_CARD_PATH.write_text(
        "\n".join(
            [
                "# Delivery-Time Model Card",
                "",
                "## Model",
                "",
                "- Model: tuned XGBoost regression",
                f"- Selected feature set: `{metrics['selected_feature_set']}`",
                f"- Target: `{TARGET}`",
                "- Prediction point: order placement",
                "",
                "## Test metrics",
                "",
                f"- RMSE: {final_metrics['rmse']:.4f}",
                f"- MAE: {final_metrics['mae']:.4f}",
                f"- R²: {final_metrics['r2']:.4f}",
                "",
                "## Best hyperparameters",
                "",
                "```json",
                json.dumps(metrics["best_params"], indent=2, default=str),
                "```",
                "",
                "## Notes",
                "",
                "- Pickup time and pickup delay are excluded from all features.",
                "- The final model was refit on train + validation before the test evaluation.",
                "- See `metrics.json`, `feature_schema.json`, and `model_metadata.json`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "model": str(config.MODEL_PATH),
        "metrics": str(config.METRICS_PATH),
        "feature_schema": str(config.FEATURE_SCHEMA_PATH),
        "model_metadata": str(config.MODEL_METADATA_PATH),
        "ablation_results": str(config.ABLATION_RESULTS_PARQUET),
        "test_predictions": str(config.TEST_PREDICTIONS_PATH),
        "model_card": str(config.MODEL_CARD_PATH),
        "permutation_importance": str(config.ARTIFACTS_DIR / "permutation_importance.csv"),
    }


def run_training_pipeline() -> TrainingResult:
    """Run baselines, ablations, tuning, final fit, and artifact export."""
    logger.info("Loading feature frame")
    frame = load_feature_data()
    split = split_dataset(frame)
    logger.info(
        "Split sizes: train=%d validation=%d test=%d",
        split.train.height,
        split.validation.height,
        split.test.height,
    )

    baseline_features = _base_features()
    linear_features = tuple(dict.fromkeys([*baseline_features, *features.LINEAR_EXTRA_FEATURES]))

    x_train, y_train = build_xy(split.train, baseline_features)
    x_validation, y_validation = build_xy(split.validation, baseline_features)

    dummy = DummyRegressor(strategy="mean").fit(x_train, y_train)
    dummy_metrics = {
        "train": regression_metrics(y_train, dummy.predict(x_train)),
        "validation": regression_metrics(y_validation, dummy.predict(x_validation)),
    }

    logger.info("Tuning Ridge baseline")
    ridge_baseline_search = tune_linear(split, baseline_features)
    ridge_baseline_metrics = {
        "train": _metrics_for_model(
            ridge_baseline_search.best_estimator_, split.train, baseline_features
        ),
        "validation": _metrics_for_model(
            ridge_baseline_search.best_estimator_,
            split.validation,
            baseline_features,
        ),
    }

    logger.info("Tuning Ridge with linear extras")
    ridge_linear_search = tune_linear(split, linear_features)
    ridge_linear_metrics = {
        "train": _metrics_for_model(
            ridge_linear_search.best_estimator_, split.train, linear_features
        ),
        "validation": _metrics_for_model(
            ridge_linear_search.best_estimator_,
            split.validation,
            linear_features,
        ),
    }

    logger.info("Fitting XGBoost baseline")
    xgb_baseline = make_xgboost_pipeline(baseline_features).fit(x_train, y_train)
    xgb_baseline_metrics = {
        "train": regression_metrics(y_train, xgb_baseline.predict(x_train)),
        "validation": regression_metrics(y_validation, xgb_baseline.predict(x_validation)),
    }

    logger.info("Running feature ablations")
    specs = build_ablation_specs()
    ablation_results = run_ablations(split, specs, params=ABLATION_XGB_PARAMS)
    selected_spec = select_best_ablation(ablation_results, specs)
    logger.info("Selected feature set: %s", selected_spec.name)

    logger.info("Tuning XGBoost on selected feature set")
    xgb_search = tune_xgboost(split, selected_spec.feature_names)
    best_params = {
        key.removeprefix("model__"): value
        for key, value in xgb_search.best_params_.items()
        if key.startswith("model__")
    }
    xgb_tuned_metrics = {
        "train": _metrics_for_model(
            xgb_search.best_estimator_,
            split.train,
            selected_spec.feature_names,
        ),
        "validation": _metrics_for_model(
            xgb_search.best_estimator_,
            split.validation,
            selected_spec.feature_names,
        ),
        "cv_rmse": -float(xgb_search.best_score_),
    }

    logger.info("Refitting final model on train + validation")
    train_validation = pl.concat([split.train, split.validation])
    x_train_validation, y_train_validation = build_xy(train_validation, selected_spec.feature_names)
    x_test_selected, y_test_selected = build_xy(split.test, selected_spec.feature_names)
    final_pipeline = make_xgboost_pipeline(
        selected_spec.feature_names,
        {**best_params, "n_jobs": -1},
    )
    final_pipeline.fit(x_train_validation, y_train_validation)
    test_predictions_array = final_pipeline.predict(x_test_selected)
    test_metrics = regression_metrics(y_test_selected, test_predictions_array)

    test_predictions = prediction_frame(
        split.test,
        test_predictions_array,
        split_name="test",
        extra_columns=("distance_decile", "coord_status"),
    )
    importance = permutation_importance_table(
        final_pipeline,
        x_test_selected,
        y_test_selected,
        selected_spec.feature_names,
        n_repeats=10,
        random_state=config.RANDOM_SEED,
    )

    metrics = {
        "models": {
            "dummy_mean": dummy_metrics,
            "ridge_baseline": ridge_baseline_metrics,
            "ridge_linear_extra": ridge_linear_metrics,
            "xgboost_baseline": xgb_baseline_metrics,
            "xgboost_tuned": xgb_tuned_metrics,
            "final_xgboost": {"test": test_metrics},
        },
        "selected_feature_set": selected_spec.name,
        "selected_features": list(selected_spec.feature_names),
        "best_params": best_params,
        "split": {
            "strategy": split.strategy,
            "seed": split.seed,
            "train_rows": split.train.height,
            "validation_rows": split.validation.height,
            "test_rows": split.test.height,
            "train_fraction": config.TRAIN_FRACTION,
            "validation_fraction": config.VALIDATION_FRACTION,
            "test_fraction": config.TEST_FRACTION,
        },
        "target": TARGET,
        "prediction_point": "order_placement",
    }

    feature_schema = _feature_schema(frame, selected_spec.feature_names, selected_spec)
    model_metadata = {
        "model_version": config.MODEL_VERSION,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "target": TARGET,
        "prediction_point": "order_placement",
        "selected_feature_set": selected_spec.name,
        "feature_names": list(selected_spec.feature_names),
        "best_params": best_params,
        "split": metrics["split"],
        "dataset": {
            "features_path": str(config.FEATURES_DATA_PARQUET.relative_to(config.PROJECT_ROOT)),
            "features_sha256": sha256_file(config.FEATURES_DATA_PARQUET),
            "features_manifest_sha256": (
                sha256_file(config.FEATURES_MANIFEST_PATH)
                if config.FEATURES_MANIFEST_PATH.exists()
                else None
            ),
        },
        "versions": _package_versions(),
        "git_commit": _git_commit(),
    }
    artifact_paths = write_training_artifacts(
        final_pipeline=final_pipeline,
        metrics=metrics,
        ablation_results=ablation_results,
        test_predictions=test_predictions,
        permutation_importance=importance,
        feature_schema=feature_schema,
        model_metadata=model_metadata,
    )

    return TrainingResult(
        split=split,
        metrics=metrics,
        ablation_results=ablation_results,
        selected_spec=selected_spec,
        xgb_search=xgb_search,
        final_pipeline=final_pipeline,
        test_predictions=test_predictions,
        permutation_importance=importance,
        artifact_paths=artifact_paths,
    )


def main() -> int:
    """Run the full training pipeline from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        result = run_training_pipeline()
    except Exception:
        logger.exception("Training failed")
        return 1

    final_metrics = result.metrics["models"]["final_xgboost"]["test"]
    logger.info("Selected feature set: %s", result.selected_spec.name)
    logger.info(
        "Test RMSE: %.4f | MAE: %.4f | R²: %.4f",
        final_metrics["rmse"],
        final_metrics["mae"],
        final_metrics["r2"],
    )
    logger.info("Artifacts: %s", result.artifact_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
