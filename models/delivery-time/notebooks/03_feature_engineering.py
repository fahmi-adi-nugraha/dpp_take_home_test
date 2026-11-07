import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Delivery-Time Feature Engineering

    ## Objective

    Build a semantic, typed feature frame for predicting `Time_taken (min)` at
    order placement. The output keeps raw categorical columns, numeric features,
    flags, and metadata; encoding and imputation are intentionally left to the
    model-training pipeline.

    Prediction happens before pickup, so `pickup_minutes`, `prep_minutes`, and
    `Time_Order_picked` are audit-only fields and must never enter the feature
    sets.

    ## Inputs

    - **Cleaned dataset:** `data/processed/zomato_clean.parquet`
    - **Data dictionary:** [`../DATA_DICTIONARY.md`](../DATA_DICTIONARY.md)
    """)
    return


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")
    import plotnine as p9
    import polars as pl

    from delivery_time import config
    from delivery_time.data import load_clean_data
    from delivery_time.features import (
        BASELINE_FEATURES,
        EXCLUDED_FEATURES,
        LEAKAGE_FIELDS,
        LINEAR_EXTRA_FEATURES,
        OPTIONAL_FEATURES,
        build_feature_frame,
        feature_set_table,
        write_feature_artifacts,
    )

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

    return (
        BASELINE_FEATURES,
        EXCLUDED_FEATURES,
        IMG_DIR,
        LEAKAGE_FIELDS,
        LINEAR_EXTRA_FEATURES,
        OPTIONAL_FEATURES,
        THEME,
        build_feature_frame,
        config,
        feature_set_table,
        load_clean_data,
        mo,
        p9,
        pl,
        save_plot,
        write_feature_artifacts,
    )


@app.cell(hide_code=True)
def _(config, load_clean_data, mo):
    cleaned = load_clean_data()
    mo.md(
        f"""
        Loaded the cleaned dataset from
        `{config.CLEAN_DATA_PARQUET.relative_to(config.PROJECT_ROOT)}`:
        **{cleaned.height:,} rows × {cleaned.width} columns**.
        """
    )
    return (cleaned,)


@app.cell(hide_code=True)
def _(cleaned):
    cleaned.glimpse()
    return


@app.cell(hide_code=True)
def _(cleaned):
    cleaned.select(
        [
            "id",
            "order_date",
            "age_raw",
            "rating_raw",
            "order_minutes",
            "prep_minutes",
            "coordinates_invalid",
            "coord_status",
            "city",
            "multiple_deliveries",
            "target",
        ]
    ).head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The cleaned input is row-preserving and already typed. The timestamps are in
    minutes since midnight, the target is complete, and the coordinate flags are
    ready to support distance construction.
    """)
    return


@app.cell(hide_code=True)
def _(build_feature_frame, cleaned, mo):
    features = build_feature_frame(cleaned)
    mo.md(
        f"""
        Built **{features.height:,} rows × {features.width} feature columns**
        including metadata, candidate features, and optional experimental
        fields.
        """
    )
    return (features,)


@app.cell(hide_code=True)
def _(features):
    features.select(
        [
            "id",
            "distance_km",
            "traffic_ordinal",
            "weather",
            "city",
            "festival_binary",
            "multiple_deliveries",
            "has_multiple_deliveries",
            "age",
            "rating",
            "order_hour",
            "is_rush_hour",
            "is_meal_period",
            "target",
        ]
    ).head(10)
    return


@app.cell(hide_code=True)
def _(feature_set_table):
    feature_set_table()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The feature table separates the baseline from linear-only and optional
    fields. The baseline uses the minimal high-signal feature set; optional fields are available
    for ablations but are not part of the default model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Geospatial features

    Distance is computed from repaired restaurant coordinates when possible.
    Invalid coordinate pairs keep a null distance and are represented by the
    `coordinates_invalid` flag. The sign-flipped rows are repaired with `abs()`.
    """)
    return


@app.cell(hide_code=True)
def _(features, pl):
    _valid = features.filter(pl.col("distance_km").is_not_null())
    distance_tbl = pl.DataFrame(
        [
            {
                "rows": features.height,
                "distance_valid": _valid.height,
                "distance_missing": features["distance_km"].null_count(),
                "min_km": _valid["distance_km"].min(),
                "median_km": _valid["distance_km"].median(),
                "mean_km": _valid["distance_km"].mean(),
                "max_km": _valid["distance_km"].max(),
                "spearman_target": _valid.select(
                    pl.corr(
                        pl.col("distance_km").rank(),
                        pl.col("target").rank(),
                    )
                ).item(),
            }
        ]
    )
    distance_tbl
    return (distance_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Distance is available for the valid and repaired rows; the 3,640 invalid
    pairs remain in the frame with a null distance. The moderate positive
    relationship with the target is consistent with the EDA, and the missing
    distance is handled by XGBoost rather than by dropping rows.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Time features

    The cleaned times are converted into order hour, cyclical hour encodings,
    day-of-week metadata, and rush/meal flags. Raw `order_hour` is the default
    tree feature; sin/cos are linear-baseline candidates.
    """)
    return


@app.cell(hide_code=True)
def _(features):
    features.select(
        [
            "order_hour",
            "order_hour_sin",
            "order_hour_cos",
            "day_of_week",
            "is_weekend",
            "is_rush_hour",
            "is_meal_period",
            "time_order_missing",
        ]
    ).head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The hour features are derived from the normalized order time. Missing order
    times produce null hour values and a `time_order_missing` flag; the model
    pipeline should not impute order time from the later pickup time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Categorical and ordinal features

    Nominal categories remain as strings for the training pipeline to encode.
    Traffic density and vehicle condition are represented as ordinal integers.
    """)
    return


@app.cell(hide_code=True)
def _(features, pl):
    _categorical = [
        "weather",
        "weather_group",
        "city",
        "vehicle_type",
        "coord_status",
        "rating_bin",
        "restaurant_grid",
        "delivery_grid",
    ]
    cardinality_tbl = pl.DataFrame(
        [
            {
                "feature": _column,
                "unique": features[_column].n_unique(),
                "missing": features[_column].null_count(),
            }
            for _column in _categorical
        ]
    ).sort("unique", descending=True)
    cardinality_tbl
    return (cardinality_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Weather, city, and vehicle type are low-cardinality and suitable for
    one-hot or native categorical handling. The rounded-coordinate grids are
    high-cardinality experimental fields and should not be one-hot encoded
    without a deliberate encoding strategy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Missingness indicators

    Missingness is retained explicitly. City and multiple-delivery missingness
    are baseline indicators; the remaining indicators are optional or reserved
    for ablation because their target relationship was weak or suspicious.
    """)
    return


@app.cell(hide_code=True)
def _(features, pl):
    _missing_features = [
        "distance_km",
        "age",
        "rating",
        "order_hour",
        "weather",
        "traffic_ordinal",
        "city",
        "multiple_deliveries",
        "festival_binary",
        "vehicle_type",
        "vehicle_condition",
        "coord_status",
    ]
    feature_missing_tbl = pl.DataFrame(
        [
            {
                "feature": _column,
                "missing": features[_column].null_count(),
                "pct": features[_column].null_count() / features.height * 100,
            }
            for _column in _missing_features
        ]
    ).sort("pct", descending=True)
    feature_missing_tbl
    return (feature_missing_tbl,)


@app.cell(hide_code=True)
def _(THEME, feature_missing_tbl, p9, pl, save_plot):
    _plot_df = feature_missing_tbl.filter(pl.col("missing") > 0).sort("pct")
    _plot_df = _plot_df.with_columns(
        pl.col("feature").cast(pl.Enum(_plot_df["feature"].to_list()))
    )
    _plot = (
        p9.ggplot(_plot_df, p9.aes(x="feature", y="pct"))
        + p9.geom_col(fill="#4C78A8")
        + p9.coord_flip()
        + THEME
        + p9.theme(figure_size=(8, 5))
        + p9.labs(
            title="Missingness in the engineered feature frame",
            subtitle="Rows with missing values by candidate feature",
            x="Feature",
            y="Missing (%)",
        )
    )
    _path = save_plot(_plot, "feature_missingness.png", width=8, height=5)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Distance, age, rating, order hour, city, and multiple deliveries carry the
    largest missing shares. The training pipeline will decide whether to impute
    them or rely on XGBoost's native missing-value handling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Feature sets and encoding plan

    The baseline set is the minimal high-signal feature set. Linear-only extras are useful for a
    linear model but are not part of the default tree model. Optional features
    remain available for ablations.
    """)
    return


@app.cell(hide_code=True)
def _(BASELINE_FEATURES, pl):
    baseline_tbl = pl.DataFrame({"baseline_feature": list(BASELINE_FEATURES)})
    baseline_tbl
    return (baseline_tbl,)


@app.cell(hide_code=True)
def _(LINEAR_EXTRA_FEATURES, pl):
    linear_tbl = pl.DataFrame(
        {"linear_extra_feature": list(LINEAR_EXTRA_FEATURES)}
    )
    linear_tbl
    return


@app.cell(hide_code=True)
def _(OPTIONAL_FEATURES, pl):
    optional_tbl = pl.DataFrame({"optional_feature": list(OPTIONAL_FEATURES)})
    optional_tbl
    return


@app.cell(hide_code=True)
def _(pl):
    encoding_tbl = pl.DataFrame(
        [
            {
                "feature": "traffic_ordinal",
                "planned_encoding": "ordinal, already encoded",
            },
            {
                "feature": "vehicle_condition",
                "planned_encoding": "ordinal, already encoded",
            },
            {"feature": "festival_binary", "planned_encoding": "binary"},
            {
                "feature": "weather",
                "planned_encoding": "one-hot or native categorical",
            },
            {
                "feature": "city",
                "planned_encoding": "one-hot or native categorical",
            },
            {
                "feature": "vehicle_type",
                "planned_encoding": "one-hot or native categorical",
            },
            {
                "feature": "coord_status",
                "planned_encoding": "one-hot only if selected",
            },
            {
                "feature": "rating_bin",
                "planned_encoding": "ordinal or one-hot only if selected",
            },
            {
                "feature": "restaurant_grid",
                "planned_encoding": "experimental; high cardinality",
            },
            {
                "feature": "delivery_grid",
                "planned_encoding": "experimental; high cardinality",
            },
        ]
    )
    encoding_tbl
    return (encoding_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    One-hot encoding is appropriate for the low-cardinality nominal fields.
    High-cardinality grid features are experimental and should not be one-hot
    encoded. Imputation, scaling, and encoding belong in the training pipeline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Leakage controls

    The target is measured from order placement, so pickup time and pickup delay
    are not available at prediction time. Identifiers and the target itself are
    metadata, not model features.
    """)
    return


@app.cell(hide_code=True)
def _(pl):
    leakage_tbl = pl.DataFrame(
        [
            {
                "field": "pickup_minutes",
                "status": "excluded",
                "reason": "Post-order field",
            },
            {
                "field": "prep_minutes",
                "status": "excluded",
                "reason": "Derived from pickup time",
            },
            {
                "field": "Time_Order_picked",
                "status": "excluded",
                "reason": "Raw pickup timestamp",
            },
            {
                "field": "id",
                "status": "metadata",
                "reason": "Identifier",
            },
            {
                "field": "courier_id",
                "status": "metadata",
                "reason": "Identifier",
            },
            {
                "field": "order_date",
                "status": "metadata",
                "reason": "Split/traceability metadata",
            },
            {
                "field": "target",
                "status": "metadata",
                "reason": "Target",
            },
        ]
    )
    leakage_tbl
    return (leakage_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The model feature lists contain none of these fields. Any courier-level
    target encoding must be added later with out-of-fold validation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Output

    The feature artifact is a semantic frame, not an encoded model matrix. It is
    written as Parquet and CSV with a manifest containing feature sets, dtypes,
    and leakage metadata.
    """)
    return


@app.cell(hide_code=True)
def _(config, features, pl, write_feature_artifacts):
    manifest = write_feature_artifacts(features)
    output_tbl = pl.DataFrame(
        [
            {
                "artifact": "parquet",
                "path": manifest["outputs"]["parquet"]["path"],
                "sha256": manifest["outputs"]["parquet"]["sha256"],
            },
            {
                "artifact": "csv",
                "path": manifest["outputs"]["csv"]["path"],
                "sha256": manifest["outputs"]["csv"]["sha256"],
            },
            {
                "artifact": "manifest",
                "path": str(config.FEATURES_MANIFEST_PATH.relative_to(config.PROJECT_ROOT)),
                "sha256": "",
            },
        ]
    )
    output_tbl
    return (manifest,)


@app.cell(hide_code=True)
def _(features):
    features.select(
        [
            "id",
            "distance_km",
            "traffic_ordinal",
            "weather",
            "city",
            "festival_binary",
            "multiple_deliveries",
            "has_multiple_deliveries",
            "vehicle_condition",
            "age",
            "rating",
            "order_hour",
            "is_rush_hour",
            "is_meal_period",
            "time_order_missing",
            "city_missing",
            "multiple_missing",
            "target",
        ]
    ).head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    **Baseline features**

    - Geospatial: `distance_km`, `coordinates_invalid`.
    - Order context: `traffic_ordinal`, `weather`, `city`, `festival_binary`,
      `multiple_deliveries`, `has_multiple_deliveries`, `vehicle_condition`.
    - Courier: `age`, `rating`.
    - Time: `order_hour`, `is_rush_hour`, `is_meal_period`, `time_order_missing`.
    - Missingness: `city_missing`, `multiple_missing`.

    **Linear-only extras**

    - `order_hour_sin`, `order_hour_cos`, `age_bin`, `rating_bin`.

    **Optional or experimental**

    - Distance/coordinate diagnostics, weather grouping, vehicle type, order
      type, day-of-week, missingness flags, clipped courier values, rounded
      coordinate grids, and courier delivery count.

    **Excluded**

    - `pickup_minutes`, `prep_minutes`, `Time_Order_picked`, identifiers, and the
      target itself.

    **Outputs**

    - `data/processed/zomato_features.parquet`
    - `data/processed/zomato_features.csv`
    - `data/processed/zomato_features_manifest.json`

    **Next step:** `04_model_training.py` reads the feature artifact, applies
    model-specific preprocessing, and runs the XGBoost baseline and feature
    ablations.
    """)
    return


if __name__ == "__main__":
    app.run()
