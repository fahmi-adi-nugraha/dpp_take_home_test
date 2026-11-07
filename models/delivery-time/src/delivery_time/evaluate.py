"""Evaluation helpers for the delivery-time models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PREDICTION_COLUMNS = [
    "id",
    "order_date",
    "target",
    "distance_km",
    "coordinates_invalid",
    "coord_status",
    "traffic_ordinal",
    "weather",
    "city",
    "festival_binary",
    "multiple_deliveries",
    "order_hour",
    "distance_decile",
]


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return RMSE, MAE, and R² for a regression prediction."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def prediction_frame(
    frame: pl.DataFrame,
    y_pred: np.ndarray,
    *,
    split_name: str,
    extra_columns: Sequence[str] = (),
) -> pl.DataFrame:
    """Return a row-aligned prediction and residual table."""
    columns = list(
        dict.fromkeys(
            column for column in [*PREDICTION_COLUMNS, *extra_columns] if column in frame.columns
        )
    )
    return (
        frame.select(columns)
        .with_columns(pl.Series("prediction", y_pred))
        .with_columns(
            (pl.col("prediction") - pl.col("target")).alias("residual"),
            (pl.col("prediction") - pl.col("target")).abs().alias("abs_error"),
            pl.lit(split_name).alias("split"),
        )
    )


def error_by_category(
    predictions: pl.DataFrame,
    column: str,
) -> pl.DataFrame:
    """Summarize prediction error by a categorical or discrete column."""
    return (
        predictions.group_by(column)
        .agg(
            pl.len().alias("n"),
            pl.col("target").mean().alias("mean_target"),
            pl.col("prediction").mean().alias("mean_prediction"),
            pl.col("abs_error").mean().alias("mae"),
            (pl.col("residual").pow(2).mean().sqrt()).alias("rmse"),
        )
        .sort("mae", descending=True)
    )


def residual_summary(predictions: pl.DataFrame) -> pl.DataFrame:
    """Return a compact residual summary."""
    return predictions.select(
        pl.len().alias("n"),
        pl.col("residual").mean().alias("mean_residual"),
        pl.col("residual").median().alias("median_residual"),
        pl.col("residual").std().alias("std_residual"),
        pl.col("abs_error").mean().alias("mae"),
        (pl.col("residual").pow(2).mean().sqrt()).alias("rmse"),
    )


def permutation_importance_table(
    model,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pl.DataFrame:
    """Return permutation importance in a Polars table."""
    result = permutation_importance(
        model,
        x,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return pl.DataFrame(
        {
            "feature": list(feature_names),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort("importance_mean", descending=True)
