import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Yelp Sentiment Fine-Tuning

    ## Objective

    We tune and fit two five-class sentiment models on the reduced hold-out data: a TF-IDF logistic regression baseline and a BERT-family classifier. The test split is not used in this phase.

    ## Search method

    Hyperparameter search uses Ray Tune's `ASHAScheduler`, the asynchronous successive-halving variant of Hyperband. Poor trials are stopped after early resource rungs; promising trials continue.

    - Logistic resource: 25,000 → 50,000 → 100,000 training rows.
    - BERT resource: 1 → 2 → 3 epochs.
    - Primary metric: validation macro F1.
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

    from yelp_sentiment import config

    try:
        NOTEBOOK_DIR = Path(__file__).resolve().parent
    except NameError:
        NOTEBOOK_DIR = Path.cwd() / "models" / "yelp-sentiment" / "notebooks"

    IMG_DIR = NOTEBOOK_DIR / "imgs"
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    THEME = p9.theme_minimal() + p9.theme(
        plot_title=p9.element_text(size=15, weight="bold"),
        plot_subtitle=p9.element_text(size=10, color="#555555"),
        axis_text_x=p9.element_text(size=8),
        axis_text_y=p9.element_text(size=8),
        legend_position="bottom",
    )

    def save_plot(plot, filename: str, width: float = 8, height: float = 5) -> Path:
        path = IMG_DIR / filename
        p9.ggsave(plot, path, width=width, height=height, dpi=150, verbose=False)
        return path

    return IMG_DIR, THEME, config, json, mo, p9, pl, save_plot


@app.cell(hide_code=True)
def _(config, json, mo, pl):
    required = [
        config.LOGISTIC_BEST_PARAMS_PATH,
        config.LOGISTIC_VALIDATION_METRICS_PATH,
        config.LOGISTIC_TUNING_TRIALS_PATH,
        config.BERT_BEST_PARAMS_PATH,
        config.BERT_VALIDATION_METRICS_PATH,
        config.BERT_TUNING_TRIALS_PATH,
        config.BERT_TRAINING_HISTORY_PATH,
        config.MODEL_FEATURES_MANIFEST_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Fine-tuning artifacts are missing. Run training first: " + ", ".join(missing)
        )

    logistic_params = json.loads(config.LOGISTIC_BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    logistic_validation = json.loads(
        config.LOGISTIC_VALIDATION_METRICS_PATH.read_text(encoding="utf-8")
    )
    bert_params = json.loads(config.BERT_BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    bert_validation = json.loads(config.BERT_VALIDATION_METRICS_PATH.read_text(encoding="utf-8"))
    bert_history = json.loads(config.BERT_TRAINING_HISTORY_PATH.read_text(encoding="utf-8"))
    model_manifest = json.loads(config.MODEL_FEATURES_MANIFEST_PATH.read_text(encoding="utf-8"))
    logistic_trials = pl.read_parquet(config.LOGISTIC_TUNING_TRIALS_PATH)
    bert_trials = pl.read_parquet(config.BERT_TUNING_TRIALS_PATH)

    mo.md(
        f"""
        Loaded the reduced model frame and both tuning studies. The model frame has **{model_manifest["output"]["rows"]:,}** rows.
        """
    )
    return (
        bert_history,
        bert_params,
        bert_trials,
        bert_validation,
        logistic_params,
        logistic_trials,
        logistic_validation,
        model_manifest,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Data and split

    The model frame is a stratified subset of the full cleaned data. The test split remains complete.
    """)
    return


@app.cell(hide_code=True)
def _(model_manifest, pl):
    split_tbl = pl.DataFrame(
        [
            {"split": split, "rows": count}
            for split, count in model_manifest["split"]["counts"].items()
        ]
    ).sort("split")
    split_tbl
    return (split_tbl,)


@app.cell(hide_code=True)
def _(mo, model_manifest):
    mo.md(
        f"""
        The search and final refit use **{model_manifest["split"]["counts"]["train"]:,}** training rows and **{model_manifest["split"]["counts"]["validation"]:,}** validation rows. The full **{model_manifest["split"]["counts"]["test"]:,}**-row test split is held out.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. ASHA search results

    The trial tables below show the configurations explored by ASHA and the macro F1 achieved at the final reported resource rung. Pruned trials may have lower reported resource usage than completed trials.
    """)
    return


@app.cell(hide_code=True)
def _(logistic_trials, pl):
    logistic_trials.select(
        "trial",
        "training_iteration",
        "macro_f1",
        "accuracy",
        "weighted_f1",
        "error",
    ).sort("macro_f1", descending=True)
    return


@app.cell(hide_code=True)
def _(THEME, logistic_trials, p9, pl, save_plot):
    _plot_df = logistic_trials.filter(pl.col("macro_f1").is_not_null()).sort("trial")
    _plot = (
        p9.ggplot(_plot_df, p9.aes(x="trial", y="macro_f1"))
        + p9.geom_point(size=2, color="#4C78A8")
        + p9.geom_line(color="#4C78A8", alpha=0.5)
        + THEME
        + p9.theme(figure_size=(8, 4))
        + p9.labs(
            title="Logistic ASHA trial results",
            subtitle="Validation macro F1 at the final reported rung",
            x="Trial number",
            y="Macro F1",
        )
    )
    _path = save_plot(_plot, "04_logistic_asha_trials.png", width=8, height=4)
    _plot
    return


@app.cell(hide_code=True)
def _(bert_trials, pl):
    bert_trials.select(
        "trial",
        "training_iteration",
        "macro_f1",
        "accuracy",
        "weighted_f1",
        "error",
    ).sort("macro_f1", descending=True)
    return


@app.cell(hide_code=True)
def _(THEME, bert_trials, p9, pl, save_plot):
    _plot_df = bert_trials.filter(pl.col("macro_f1").is_not_null()).sort("trial")
    _plot = (
        p9.ggplot(_plot_df, p9.aes(x="trial", y="macro_f1"))
        + p9.geom_point(size=2, color="#4C78A8")
        + p9.geom_line(color="#4C78A8", alpha=0.5)
        + THEME
        + p9.theme(figure_size=(8, 4))
        + p9.labs(
            title="BERT ASHA trial results",
            subtitle="Validation macro F1 at the final reported rung",
            x="Trial number",
            y="Macro F1",
        )
    )
    _path = save_plot(_plot, "04_bert_asha_trials.png", width=8, height=4)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Selected hyperparameters

    The exact parameter sets below were used for the final refits.
    """)
    return


@app.cell(hide_code=True)
def _(bert_params, logistic_params, pl):
    _rows = []
    for name, value in logistic_params.items():
        _rows.append({"model": "logistic_regression", "parameter": name, "value": str(value)})
    for name, value in bert_params.items():
        _rows.append({"model": "bert", "parameter": name, "value": str(value)})
    selected_params_tbl = pl.DataFrame(_rows)
    selected_params_tbl
    return (selected_params_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Validation performance

    Logistic validation metrics come from the full reduced validation split. BERT validation metrics are the best search-subset validation macro F1 from ASHA; the final BERT model was then refit on the reduced train + validation data.
    """)
    return


@app.cell(hide_code=True)
def _(logistic_validation, pl):
    _metrics = logistic_validation["metrics"]
    logistic_metrics_tbl = pl.DataFrame(
        [
            {"metric": "accuracy", "value": _metrics["accuracy"]},
            {"metric": "macro_f1", "value": _metrics["macro_f1"]},
            {"metric": "weighted_f1", "value": _metrics["weighted_f1"]},
        ]
    ).with_columns(pl.col("value").round(2))
    logistic_metrics_tbl
    return (logistic_metrics_tbl,)


@app.cell(hide_code=True)
def _(logistic_validation, pl):
    pl.DataFrame(logistic_validation["per_class"]).with_columns(
        pl.col("precision").round(2),
        pl.col("recall").round(2),
        pl.col("f1").round(2),
    )
    return


@app.cell(hide_code=True)
def _(bert_validation, pl):
    pl.DataFrame(
        [
            {
                "metric": "best_search_macro_f1",
                "value": bert_validation["best_search_macro_f1"],
            },
            {
                "metric": "search_validation_rows",
                "value": bert_validation["search_validation_rows"],
            },
            {
                "metric": "final_fit_rows",
                "value": bert_validation["final_fit_rows"],
            },
        ]
    ).with_columns(pl.col("value").round(2))
    return


@app.cell(hide_code=True)
def _(bert_history, pl):
    training_history_tbl = pl.DataFrame(bert_history["history"]).with_columns(
        pl.col("train_loss").round(2),
        pl.col("macro_f1").round(2),
        pl.col("accuracy").round(2),
        pl.col("weighted_f1").round(2),
    )
    training_history_tbl
    return (training_history_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The final BERT refit trains on the reduced train + validation frame. Its per-epoch validation scores are therefore diagnostic only; the held-out test scores are reported in the evaluation notebook.
    """)
    return


@app.cell(hide_code=True)
def _(THEME, bert_history, p9, pl, save_plot):
    _plot_df = pl.DataFrame(bert_history["history"])
    _plot = (
        p9.ggplot(_plot_df, p9.aes(x="epoch", y="macro_f1"))
        + p9.geom_line(color="#4C78A8")
        + p9.geom_point(color="#4C78A8", size=2)
        + THEME
        + p9.theme(figure_size=(7, 4))
        + p9.labs(
            title="Final BERT refit history",
            subtitle="Validation-set macro F1 by epoch; validation was merged into the refit training frame, so these are not held-out scores",
            x="Epoch",
            y="Macro F1",
        )
    )
    _path = save_plot(_plot, "04_bert_final_history.png", width=7, height=4)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Artifacts

    The fine-tuning phase writes the tuned policies and final model artifacts for evaluation.
    """)
    return


@app.cell(hide_code=True)
def _(config, pl):
    artifacts_tbl = pl.DataFrame(
        [
            {"artifact": "logistic params", "path": str(config.LOGISTIC_BEST_PARAMS_PATH)},
            {"artifact": "logistic model", "path": str(config.LOGISTIC_MODEL_PATH)},
            {"artifact": "logistic trials", "path": str(config.LOGISTIC_TUNING_TRIALS_PATH)},
            {"artifact": "bert params", "path": str(config.BERT_BEST_PARAMS_PATH)},
            {"artifact": "bert model", "path": str(config.BERT_MODEL_DIR)},
            {"artifact": "bert trials", "path": str(config.BERT_TUNING_TRIALS_PATH)},
            {"artifact": "bert history", "path": str(config.BERT_TRAINING_HISTORY_PATH)},
        ]
    )
    artifacts_tbl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    We used ASHA to tune a TF-IDF logistic baseline and BERT on the reduced hold-out data. The best parameter sets were saved, both models were refit for final use, and the test split remained untouched for the evaluation phase.
    """)
    return


if __name__ == "__main__":
    app.run()
