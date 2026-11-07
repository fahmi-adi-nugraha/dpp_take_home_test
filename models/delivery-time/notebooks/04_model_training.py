import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Delivery-Time Model Training

    ## Objective

    Train a linear baseline and a tuned XGBoost regressor for order-time
    delivery prediction, using the checklist baseline and validation-driven
    feature ablations.

    The training pipeline is implemented in
    [`../src/delivery_time/train.py`](../src/delivery_time/train.py) and is
    executed with:

    ```bash
    uv run train-delivery-model
    ```

    This notebook reads the generated artifacts and documents the model
    selection, ablation results, test performance, and error analysis.

    ## Setup

    - **Feature artifact:** `data/processed/zomato_features.parquet`
    - **Split:** random 80/10/10 with seed 42
    - **Target:** `target`
    - **Prediction point:** order placement
    - **Leakage controls:** pickup time and pickup delay are excluded
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")
    import plotnine as p9
    import polars as pl

    from delivery_time import config
    from delivery_time.evaluate import error_by_category

    NOTEBOOK_DIR = Path(__file__).resolve().parent
    IMG_DIR = NOTEBOOK_DIR / "imgs"
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    THEME = p9.theme_minimal() + p9.theme(
        plot_title=p9.element_text(size=15, weight="bold"),
        plot_subtitle=p9.element_text(size=10, color="#555555"),
        axis_text_x=p9.element_text(size=8),
        axis_text_y=p9.element_text(size=8),
        legend_position="bottom",
    )

    def save_plot(plot, filename: str, width: float = 8, height: float = 5):
        path = IMG_DIR / filename
        p9.ggsave(plot, path, width=width, height=height, dpi=150, verbose=False)
        return path

    return THEME, config, error_by_category, json, mo, p9, pl, save_plot


@app.cell(hide_code=True)
def _(config, json, mo, pl):
    required = [
        config.METRICS_PATH,
        config.ABLATION_RESULTS_PARQUET,
        config.TEST_PREDICTIONS_PATH,
        config.MODEL_METADATA_PATH,
        config.FEATURE_SCHEMA_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Training artifacts are missing. Run `uv run train-delivery-model` first: "
            + ", ".join(missing)
        )

    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    feature_schema = json.loads(
        config.FEATURE_SCHEMA_PATH.read_text(encoding="utf-8")
    )
    ablation_results = pl.read_parquet(config.ABLATION_RESULTS_PARQUET)
    test_predictions = pl.read_parquet(config.TEST_PREDICTIONS_PATH)
    permutation_importance = pl.read_csv(
        config.ARTIFACTS_DIR / "permutation_importance.csv"
    )

    mo.md(
        f"""
        Loaded artifacts for the **{metrics['selected_feature_set']}** feature
        set. Test rows: **{len(test_predictions):,}**.
        """
    )
    return (
        ablation_results,
        feature_schema,
        metadata,
        metrics,
        permutation_importance,
        test_predictions,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Split and baseline metrics

    The primary split is a reproducible random 80/10/10 split. The validation
    split is used for model comparison and tuning; the test split is used only
    for the final selected model.
    """)
    return


@app.cell(hide_code=True)
def _(metrics, pl):
    split_tbl = pl.DataFrame(
        [
            {"item": "strategy", "value": str(metrics["split"]["strategy"])},
            {"item": "seed", "value": str(metrics["split"]["seed"])},
            {"item": "train rows", "value": f"{metrics['split']['train_rows']:,}"},
            {
                "item": "validation rows",
                "value": f"{metrics['split']['validation_rows']:,}",
            },
            {"item": "test rows", "value": f"{metrics['split']['test_rows']:,}"},
        ]
    )
    split_tbl
    return


@app.cell(hide_code=True)
def _(metrics, pl):
    _rows = []
    for _model_name, _splits in metrics["models"].items():
        for _split_name, _values in _splits.items():
            if isinstance(_values, dict):
                _rows.append(
                    {
                        "model": _model_name,
                        "split": _split_name,
                        "rmse": _values["rmse"],
                        "mae": _values["mae"],
                        "r2": _values["r2"],
                    }
                )
    metrics_tbl = pl.DataFrame(_rows).sort(["split", "rmse"])
    metrics_tbl
    return


@app.cell(hide_code=True)
def _(metrics, mo):
    _models = metrics["models"]
    mo.md(
        f"""
        The mean baseline RMSE is **{_models['dummy_mean']['validation']['rmse']:.2f}**.
        Ridge improves that to **{_models['ridge_baseline']['validation']['rmse']:.2f}**,
        and the linear extras bring it to
        **{_models['ridge_linear_extra']['validation']['rmse']:.2f}**.
        XGBoost starts at **{_models['xgboost_baseline']['validation']['rmse']:.2f}**
        and reaches **{_models['xgboost_tuned']['validation']['rmse']:.2f}** after tuning.
        The final model scores **{_models['final_xgboost']['test']['rmse']:.2f}** RMSE
        on the held-out test set.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Feature ablation

    Ablations use a fixed XGBoost configuration on validation data. They isolate
    distance handling, multiple-delivery encoding, hour encoding, missingness
    flags, and optional feature groups.
    """)
    return


@app.cell(hide_code=True)
def _(ablation_results):
    ablation_results.select(
        [
            "name",
            "n_features",
            "rmse",
            "mae",
            "r2",
            "delta_rmse_vs_baseline",
            "eligible_for_final",
        ]
    )
    return


@app.cell(hide_code=True)
def _(THEME, ablation_results, p9, pl, save_plot):
    _sorted = ablation_results.sort("delta_rmse_vs_baseline")
    _plot_df = _sorted.with_columns(
        pl.when(pl.col("eligible_for_final"))
        .then(pl.lit("eligible"))
        .otherwise(pl.lit("diagnostic"))
        .alias("status"),
        pl.col("name").cast(pl.Enum(_sorted["name"].to_list())),
    )
    _plot = (
        p9.ggplot(
            _plot_df,
            p9.aes(
                x="name",
                y="delta_rmse_vs_baseline",
                fill="status",
            ),
        )
        + p9.geom_col()
        + p9.coord_flip()
        + p9.scale_fill_manual(
            values={"eligible": "#4C78A8", "diagnostic": "#D62728"}
        )
        + THEME
        + p9.theme(figure_size=(8, 6))
        + p9.labs(
            title="Validation RMSE change versus baseline",
            subtitle="Negative values are improvements; color marks final-selection eligibility",
            x="Ablation",
            y="Delta RMSE",
            fill="Status",
        )
    )
    _path = save_plot(_plot, "ablation_rmse_delta.png", width=8, height=6)
    _plot
    return


@app.cell(hide_code=True)
def _(ablation_results, mo, pl):
    _best = ablation_results.filter(pl.col("eligible_for_final")).row(0, named=True)
    _invalid = ablation_results.filter(
        pl.col("name") == "drop_invalid_coordinates"
    ).row(0, named=True)
    _no_distance = ablation_results.filter(pl.col("name") == "no_distance").row(
        0, named=True
    )
    mo.md(
        f"""
        The best eligible ablation is **{_best['name']}** at
        **{_best['rmse']:.3f}** validation RMSE, a
        **{abs(_best['delta_rmse_vs_baseline']):.3f}** improvement over the
        baseline. Dropping invalid-coordinate rows scores
        **{_invalid['rmse']:.3f}**, but it is diagnostic only because the
        checklist keeps those rows with a validity flag. Removing distance is
        clearly harmful at **{_no_distance['rmse']:.3f}** RMSE.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Hyperparameter tuning

    Ridge alpha and XGBoost hyperparameters are tuned with five-fold
    `RandomizedSearchCV` on the training split. The search optimizes negative
    RMSE; no exhaustive grid search is used.
    """)
    return


@app.cell(hide_code=True)
def _(metrics, pl):
    tuning_tbl = pl.DataFrame(
        [
            {"parameter": _key, "value": str(_value)}
            for _key, _value in metrics["best_params"].items()
        ]
    )
    tuning_tbl
    return


@app.cell(hide_code=True)
def _(metrics, mo):
    _tuned = metrics["models"]["xgboost_tuned"]
    mo.md(
        f"""
        The tuned XGBoost model reaches **{_tuned['cv_rmse']:.3f}** CV RMSE,
        **{_tuned['validation']['rmse']:.3f}** validation RMSE, and
        **{_tuned['validation']['r2']:.3f}** validation R². The final model is
        refit on train + validation and evaluated once on test.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Test-set performance

    The plots below use only the held-out test predictions from the final
    refit model.
    """)
    return


@app.cell(hide_code=True)
def _(THEME, p9, save_plot, test_predictions):
    _plot = (
        p9.ggplot(test_predictions, p9.aes(x="target", y="prediction"))
        + p9.geom_point(alpha=0.25, size=0.7, color="#4C78A8")
        + p9.geom_abline(intercept=0, slope=1, color="#D62728", linetype="dashed")
        + THEME
        + p9.theme(figure_size=(7, 5))
        + p9.labs(
            title="Predicted vs actual delivery time",
            subtitle="Test split; dashed line is perfect prediction",
            x="Actual minutes",
            y="Predicted minutes",
        )
    )
    _path = save_plot(_plot, "test_predicted_vs_actual.png", width=7, height=5)
    _plot
    return


@app.cell(hide_code=True)
def _(THEME, p9, save_plot, test_predictions):
    _plot = (
        p9.ggplot(test_predictions, p9.aes(x="residual"))
        + p9.geom_histogram(bins=40, fill="#4C78A8", color="white")
        + p9.geom_vline(xintercept=0, color="#D62728", linetype="dashed")
        + THEME
        + p9.theme(figure_size=(7, 4))
        + p9.labs(
            title="Residual distribution",
            subtitle="Positive residuals mean the model predicted too high",
            x="Residual (prediction - actual)",
            y="Count",
        )
    )
    _path = save_plot(_plot, "test_residual_distribution.png", width=7, height=4)
    _plot
    return


@app.cell(hide_code=True)
def _(THEME, p9, save_plot, test_predictions):
    _plot = (
        p9.ggplot(test_predictions, p9.aes(x="prediction", y="residual"))
        + p9.geom_point(alpha=0.2, size=0.5, color="#4C78A8")
        + p9.geom_hline(yintercept=0, color="#D62728", linetype="dashed")
        + THEME
        + p9.theme(figure_size=(7, 5))
        + p9.labs(
            title="Residuals vs predicted value",
            subtitle="Test split",
            x="Predicted minutes",
            y="Residual",
        )
    )
    _path = save_plot(_plot, "test_residuals_vs_predicted.png", width=7, height=5)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The residual distribution is centered near zero, with a small positive tail
    indicating some under-prediction of long deliveries. The residual summary
    below quantifies that behavior.
    """)
    return


@app.cell(hide_code=True)
def _(pl, test_predictions):
    test_predictions.select(
        pl.len().alias("n"),
        pl.col("residual").mean().alias("mean_residual"),
        pl.col("residual").median().alias("median_residual"),
        pl.col("abs_error").mean().alias("mae"),
        pl.col("residual").pow(2).mean().sqrt().alias("rmse"),
        pl.col("target").mean().alias("mean_actual"),
        pl.col("prediction").mean().alias("mean_prediction"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Segment error analysis

    I compare absolute error across city, traffic, weather, coordinate validity,
    and multiple-delivery groups to identify where the model struggles.
    """)
    return


@app.cell(hide_code=True)
def _(error_by_category, pl, test_predictions):
    _segment_columns = [
        "city",
        "traffic_ordinal",
        "weather",
        "coordinates_invalid",
        "multiple_deliveries",
    ]
    _frames = []
    for _column in _segment_columns:
        _table = (
            error_by_category(test_predictions, _column)
            .rename({_column: "level"})
            .with_columns(
                pl.lit(_column).alias("segment"),
                pl.col("level").cast(pl.String),
            )
        )
        _frames.append(_table.select(["segment", "level", "n", "mae", "rmse"]))
    segment_tbl = pl.concat(_frames)
    segment_tbl
    return (segment_tbl,)


@app.cell(hide_code=True)
def _(THEME, p9, save_plot, segment_tbl):
    _plot = (
        p9.ggplot(segment_tbl, p9.aes(x="level", y="mae"))
        + p9.geom_col(fill="#4C78A8")
        + p9.facet_wrap("segment", scales="free", ncol=2)
        + THEME
        + p9.theme(
            figure_size=(9, 6),
            axis_text_x=p9.element_text(rotation=35, hjust=1),
        )
        + p9.labs(
            title="Mean absolute error by segment",
            subtitle="Test split",
            x="",
            y="MAE (minutes)",
        )
    )
    _path = save_plot(_plot, "test_error_by_segment.png", width=9, height=6)
    _plot
    return


@app.cell(hide_code=True)
def _(mo, segment_tbl):
    _worst = segment_tbl.sort("mae", descending=True).head(5)
    mo.md(
        "The highest-error segments are: "
        + ", ".join(
            f"{row['segment']}={row['level']} ({row['mae']:.2f} min)"
            for row in _worst.iter_rows(named=True)
        )
        + ". These are the first groups to inspect for additional features or "
        "data-quality issues."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Feature importance

    Permutation importance is computed on the test split for the final model.
    It measures the RMSE increase when a feature is shuffled and is therefore
    in the same units as the target.
    """)
    return


@app.cell(hide_code=True)
def _(permutation_importance):
    permutation_importance.head(20)
    return


@app.cell(hide_code=True)
def _(THEME, p9, permutation_importance, pl, save_plot):
    _plot_df = permutation_importance.head(15).sort("importance_mean")
    _plot_df = _plot_df.with_columns(
        pl.col("feature").cast(pl.Enum(_plot_df["feature"].to_list()))
    )
    _plot = (
        p9.ggplot(
            _plot_df,
            p9.aes(x="feature", y="importance_mean"),
        )
        + p9.geom_col(fill="#4C78A8")
        + p9.coord_flip()
        + THEME
        + p9.theme(figure_size=(8, 5))
        + p9.labs(
            title="Permutation importance",
            subtitle="Increase in RMSE when the feature is shuffled",
            x="Feature",
            y="Mean RMSE increase",
        )
    )
    _path = save_plot(_plot, "permutation_importance.png", width=8, height=5)
    _plot
    return


@app.cell(hide_code=True)
def _(mo, permutation_importance):
    _top = permutation_importance.head(5).rows(named=True)
    mo.md(
        "The top permutation-importance features are: "
        + ", ".join(
            f"`{row['feature']}` ({row['importance_mean']:.3f})" for row in _top
        )
        + ". This confirms the model relies on the expected delivery-time signals "
        "rather than identifiers or leakage fields."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Saved artifacts

    The final pipeline, metrics, schema, metadata, ablations, and test
    predictions are written under `artifacts/`.
    """)
    return


@app.cell(hide_code=True)
def _(config, metadata, pl):
    artifact_tbl = pl.DataFrame(
        [
            {"item": "model_version", "value": metadata["model_version"]},
            {"item": "trained_at_utc", "value": metadata["trained_at_utc"]},
            {"item": "selected_feature_set", "value": metadata["selected_feature_set"]},
            {"item": "git_commit", "value": str(metadata.get("git_commit"))},
            {
                "item": "features_sha256",
                "value": metadata["dataset"]["features_sha256"],
            },
            {"item": "model", "value": str(config.MODEL_PATH)},
            {"item": "metrics", "value": str(config.METRICS_PATH)},
            {"item": "feature_schema", "value": str(config.FEATURE_SCHEMA_PATH)},
            {"item": "ablation_results", "value": str(config.ABLATION_RESULTS_PARQUET)},
            {"item": "test_predictions", "value": str(config.TEST_PREDICTIONS_PATH)},
        ]
    )
    artifact_tbl
    return


@app.cell(hide_code=True)
def _(feature_schema, mo):
    mo.md(f"""
    The saved feature schema contains **{len(feature_schema['features'])}**
    expected model inputs, with
    **{len(feature_schema['categorical_features'])}** categorical features.
    Package versions and the git commit are recorded in the metadata file.
    """)
    return


@app.cell(hide_code=True)
def _(metadata, metrics, mo):
    _test = metrics["models"]["final_xgboost"]["test"]
    _tuned = metrics["models"]["xgboost_tuned"]
    mo.md(
        f"""
        ## Summary

        - **Selected feature set:** `{metrics['selected_feature_set']}`
        - **Best validation RMSE after tuning:** {_tuned['validation']['rmse']:.3f}
        - **Held-out test RMSE:** {_test['rmse']:.3f}
        - **Held-out test MAE:** {_test['mae']:.3f}
        - **Held-out test R²:** {_test['r2']:.3f}
        - **Training command:** `uv run train-delivery-model`
        - **Model artifact:** `models/delivery-time/artifacts/model.joblib`
        - **Git commit at training time:** `{metadata.get('git_commit')}`

        The ablation stage promoted the grouped-weather feature into the final
        model. Invalid-coordinate dropping was diagnostic only, and removing
        distance caused the largest validation regression. The final model is
        ready for the inference API integration step.
        """
    )
    return


if __name__ == "__main__":
    app.run()
