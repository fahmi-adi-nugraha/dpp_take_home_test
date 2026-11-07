import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Yelp Sentiment Evaluation

    ## Objective

    We compare the final logistic regression and BERT models on the untouched 50,000-review test split. The hyperparameters shown beside the metrics are the exact configurations saved during fine-tuning.

    ## Metrics

    - accuracy
    - macro F1
    - weighted F1
    - per-class precision, recall, and F1
    - confusion matrices
    - misclassified review examples
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
    LABEL_ORDER = tuple(config.LABEL_MAP[index] for index in range(5))

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

    return IMG_DIR, LABEL_ORDER, THEME, config, json, mo, p9, pl, save_plot


@app.cell(hide_code=True)
def _(config, json, mo, pl):
    required = [
        config.EVALUATION_METRICS_PATH,
        config.EVALUATION_METRICS_TABLE_PATH,
        config.EVALUATION_HYPERPARAMETERS_PATH,
        config.EVALUATION_PER_CLASS_PATH,
        config.EVALUATION_CONFUSION_PATH,
        config.EVALUATION_PREDICTIONS_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Evaluation artifacts are missing. Run evaluation first: " + ", ".join(missing)
        )

    metrics = json.loads(config.EVALUATION_METRICS_PATH.read_text(encoding="utf-8"))
    metrics_table = pl.read_parquet(config.EVALUATION_METRICS_TABLE_PATH)
    hyperparameters_table = pl.read_parquet(config.EVALUATION_HYPERPARAMETERS_PATH)
    per_class_table = pl.read_parquet(config.EVALUATION_PER_CLASS_PATH)
    confusion_table = pl.read_parquet(config.EVALUATION_CONFUSION_PATH)
    predictions_table = pl.read_parquet(config.EVALUATION_PREDICTIONS_PATH)

    mo.md(
        f"""
        Loaded evaluation artifacts for **{metrics["test_rows"]:,}** held-out reviews and **{len(metrics["models"])}** models.
        """
    )
    return (
        confusion_table,
        hyperparameters_table,
        metrics,
        metrics_table,
        per_class_table,
        predictions_table,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Model comparison

    The comparison table includes the hyperparameters used for each final model. Macro F1 is the primary comparison metric because it weights each star rating equally.
    """)
    return


@app.cell(hide_code=True)
def _(metrics_table, pl):
    metrics_table.with_columns(
        pl.col("accuracy").round(2),
        pl.col("macro_f1").round(2),
        pl.col("weighted_f1").round(2),
    )
    return


@app.cell(hide_code=True)
def _(hyperparameters_table):
    hyperparameters_table
    return


@app.cell(hide_code=True)
def _(mo, metrics_table):
    _best = metrics_table.row(0, named=True)
    mo.md(
        f"""
        The best test model is **{_best["model"]}**, with **{_best["accuracy"]:.2%}** accuracy and **{_best["macro_f1"]:.2%}** macro F1. The hyperparameters above are the exact configurations used for these test predictions.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Per-class performance

    Per-class metrics show whether either model is weaker on the ambiguous middle star ratings.
    """)
    return


@app.cell(hide_code=True)
def _(per_class_table, pl):
    per_class_table.with_columns(
        pl.col("precision").round(2),
        pl.col("recall").round(2),
        pl.col("f1").round(2),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Confusion matrices

    The confusion matrices make adjacent-class errors visible. A good five-class sentiment model should still concentrate most errors near the true rating.
    """)
    return


@app.cell(hide_code=True)
def _(LABEL_ORDER, THEME, confusion_table, p9, pl, save_plot):
    _plot_df = confusion_table.with_columns(
        pl.col("true_name").cast(pl.Enum(LABEL_ORDER)),
        pl.col("predicted_name").cast(pl.Enum(LABEL_ORDER)),
    )
    _plot = (
        p9.ggplot(
            _plot_df,
            p9.aes(x="predicted_name", y="true_name", fill="count"),
        )
        + p9.geom_tile(color="white")
        + p9.geom_text(p9.aes(label="count"), size=6)
        + p9.facet_wrap("model", ncol=2, scales="free")
        + p9.scale_fill_gradient(low="#F7FBFF", high="#08306B")
        + THEME
        + p9.theme(
            figure_size=(10, 6),
            panel_grid=p9.element_blank(),
            legend_position="none",
        )
        + p9.labs(
            title="Test confusion matrices",
            subtitle="Rows are actual labels; columns are predicted labels",
            x="Predicted",
            y="Actual",
        )
    )
    _path = save_plot(_plot, "05_test_confusion_matrices.png", width=10, height=6)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Misclassified examples

    These are high-confidence errors from the test split. They are useful for checking whether the remaining mistakes are ambiguous reviews, label noise, or a systematic model weakness.
    """)
    return


@app.cell(hide_code=True)
def _(predictions_table, pl):
    (
        predictions_table.filter(pl.col("actual_name") != pl.col("predicted_name"))
        .sort("confidence", descending=True)
        .head(12)
        .select(
            "model",
            "actual_name",
            "predicted_name",
            "confidence",
            "text_excerpt",
        )
        .with_columns(pl.col("confidence").round(2))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Saved evaluation artifacts

    The comparison table, per-class metrics, confusion matrices, predictions, and full JSON payload are stored under `artifacts/evaluation/`.
    """)
    return


@app.cell(hide_code=True)
def _(config, pl):
    artifacts_tbl = pl.DataFrame(
        [
            {"artifact": "metrics json", "path": str(config.EVALUATION_METRICS_PATH)},
            {"artifact": "metrics table", "path": str(config.EVALUATION_METRICS_TABLE_PATH)},
            {"artifact": "hyperparameters", "path": str(config.EVALUATION_HYPERPARAMETERS_PATH)},
            {"artifact": "per-class metrics", "path": str(config.EVALUATION_PER_CLASS_PATH)},
            {"artifact": "confusion matrices", "path": str(config.EVALUATION_CONFUSION_PATH)},
            {"artifact": "test predictions", "path": str(config.EVALUATION_PREDICTIONS_PATH)},
        ]
    )
    artifacts_tbl
    return


@app.cell(hide_code=True)
def _(metrics_table, mo):
    _best = metrics_table.row(0, named=True)
    mo.md(
        f"""
        ## Summary

        - **Best test model:** `{_best["model"]}`
        - **Accuracy:** {_best["accuracy"]:.2%}
        - **Macro F1:** {_best["macro_f1"]:.2%}
        - **Weighted F1:** {_best["weighted_f1"]:.2%}
        - **Test rows:** 50,000

        The test split was used only for this final comparison. The hyperparameters shown above come directly from the fine-tuning artifacts.
        """
    )
    return


if __name__ == "__main__":
    app.run()
