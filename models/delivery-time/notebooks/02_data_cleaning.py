import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Delivery-Time Data Cleaning

    ## Objective

    Convert the raw Zomato delivery dataset into a typed, auditable cleaned
    dataset for feature engineering and model training. The target is
    `Time_taken (min)`, interpreted as total delivery time starting at order
    placement. This notebook keeps the raw data immutable and writes both
    Parquet and CSV outputs under `data/processed/`.

    ## Inputs

    - **Raw file:** `data/raw/zomato/Zomato Dataset.csv`
    - **Rows:** 45,584
    - **Columns:** 20
    - **Data dictionary:** [`../DATA_DICTIONARY.md`](../DATA_DICTIONARY.md)
    - **Feature checklist:** [`../feature_engineering_checklist.md`](../feature_engineering_checklist.md)

    The cleaning notebook does not impute model features. It parses types,
    standardizes representations, flags anomalies, and preserves missing values
    for the model pipeline.
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
    from delivery_time.data import (
        classify_time_format,
        clean_data,
        load_raw_data,
        write_clean_artifacts,
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
        IMG_DIR,
        THEME,
        classify_time_format,
        clean_data,
        config,
        load_raw_data,
        mo,
        p9,
        pl,
        save_plot,
        write_clean_artifacts,
    )


@app.cell(hide_code=True)
def _(clean_data, load_raw_data, mo):
    raw = load_raw_data()
    cleaned = clean_data(raw)
    mo.md(
        f"""
        Loaded **{raw.height:,} raw rows × {raw.width} columns** and built a
        cleaned frame with **{cleaned.height:,} rows × {cleaned.width} columns**.

        The cleaned frame keeps identifiers, parsed timestamps, raw and derived
        courier attributes, coordinate-quality flags, missingness indicators,
        normalized categories, and the target.
        """
    )
    return cleaned, raw


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Input snapshot

    The raw columns are all text, so the first pass is about understanding the
    representations before parsing. I inspect the schema, the first rows, and
    the raw time formats.
    """)
    return


@app.cell(hide_code=True)
def _(raw):
    raw.glimpse()
    return


@app.cell(hide_code=True)
def _(raw):
    raw.head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The raw data is consistently rectangular: 45,584 rows and 20 columns. The
    missing values are literal `"NaN"` strings rather than nulls, and the time
    columns mix normal clock times with fractional-day and after-midnight
    representations. These representations need explicit parsing before they
    can be used as features.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Issue inventory

    | Issue | Evidence | Proposed action | Validation |
    |---|---|---|---|
    | Missing values encoded as text | `"NaN"` appears across age, ratings, time, weather, traffic, multiple deliveries, festival, and city | Parse as null and add missingness indicators | Compare raw token counts with cleaned null counts |
    | Mixed time formats | `HH:MM`, fractional day, `1`, and `24:xx:00` | Normalize to minutes since midnight | Show raw and parsed samples; check pickup delay range |
    | City spelling | `Metropolitian` | Normalize to `Metropolitan` | Value counts before/after |
    | Ratings above 5.0 | 53 values above 5.0, all 6.0 | Flag and create clipped copy | Count flags and compare clipped range |
    | Courier age below 18 | 38 rows | Flag and create clipped copy | Count flags and compare clipped range |
    | Invalid coordinates | 3,640 rows with zero/small paired coordinates | Flag; do not compute distance | Coordinate-status counts |
    | Sign-flipped coordinates | 431 rows | Flag and create repaired coordinates | Compare raw and repaired samples |
    | Duplicate rows | None observed | Keep the duplicate check in the pipeline | Count duplicates before/after |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Missing values

    I keep missing values as nulls rather than imputing in the cleaning layer.
    The model pipeline can choose an imputation strategy, while missingness
    indicators preserve information that may be predictive.
    """)
    return


@app.cell(hide_code=True)
def _(cleaned, pl, raw):
    _fields = [
        ("Delivery_person_Age", "age_raw"),
        ("Delivery_person_Ratings", "rating_raw"),
        ("Time_Orderd", "order_minutes"),
        ("City", "city"),
        ("multiple_deliveries", "multiple_deliveries"),
        ("Weather_conditions", "weather"),
        ("Road_traffic_density", "traffic"),
        ("Festival", "festival"),
    ]
    missing_tbl = pl.DataFrame(
        [
            {
                "column": _source,
                "raw_nan": raw.filter(
                    raw[_source].str.strip_chars() == "NaN"
                ).height,
                "clean_null": cleaned[_target].null_count(),
            }
            for _source, _target in _fields
        ]
    )
    missing_tbl
    return (missing_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The raw and cleaned counts match for every field, which confirms that the
    parser is treating every literal `"NaN"` as a null. Ratings, age, and order
    time are the largest missingness groups; city and multiple deliveries are
    smaller but potentially informative.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Duplicate rows

    The raw dataset has no duplicate rows. The check is retained in the cleaning
    function so that future refreshes of the source data are protected.
    """)
    return


@app.cell(hide_code=True)
def _(cleaned, pl, raw):
    duplicate_tbl = pl.DataFrame(
        [
            {
                "stage": "raw",
                "rows": raw.height,
                "duplicate_rows": raw.is_duplicated().sum(),
            },
            {
                "stage": "cleaned",
                "rows": cleaned.height,
                "duplicate_rows": cleaned.is_duplicated().sum(),
            },
        ]
    )
    duplicate_tbl
    return (duplicate_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    No rows are removed. The cleaning step is row-preserving; only
    representations and derived flags change.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Time normalization

    The time fields need a common representation. I convert both order and
    pickup times to minutes since midnight. This handles `HH:MM`, fractional-day
    values, `1` meaning 24:00, and `24:xx:00` after-midnight values without
    needing an arbitrary date. Pickup delay is retained only for audit and
    leakage checks.
    """)
    return


@app.cell(hide_code=True)
def _(classify_time_format, pl, raw):
    _frames = []
    for _column in ["Time_Orderd", "Time_Order_picked"]:
        _formats = raw[_column].map_elements(
            classify_time_format, return_dtype=pl.String
        )
        _counts = _formats.value_counts(sort=True).rename({"count": "n"})
        _counts = _counts.with_columns(
            pl.lit(_column).alias("column"),
            (pl.col("n") / raw.height * 100).alias("share"),
        )
        _counts = _counts.rename({_column: "format"})
        _frames.append(_counts.select(["column", "format", "n", "share"]))
    time_format_tbl = pl.concat(_frames)
    time_format_tbl
    return (time_format_tbl,)


@app.cell(hide_code=True)
def _(cleaned, pl, raw):
    _raw_time = raw.select(["ID", "Time_Orderd", "Time_Order_picked"]).head(10)
    _clean_time = cleaned.select(
        ["order_minutes", "pickup_minutes", "prep_minutes"]
    ).head(10)
    time_sample_tbl = pl.concat([_raw_time, _clean_time], how="horizontal")
    time_sample_tbl
    return (time_sample_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The parsed samples confirm that `21:55` becomes `1315`, `0.833333333`
    becomes `1200`, `1` becomes `1440`, and `24:05:00` becomes `1445`. Pickup
    delay stays in the expected 5–15 minute range after normalization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. City, ratings, and age

    I normalize the city spelling and create preserved raw, clipped, and flagged
    versions of the courier attributes. The clipped versions are candidates for
    the linear baseline; XGBoost can use the raw values and flags.
    """)
    return


@app.cell(hide_code=True)
def _(cleaned, pl, raw):
    quality_tbl = pl.DataFrame(
        [
            {
                "check": "City uses 'Metropolitian'",
                "count": raw.filter(raw["City"].str.strip_chars() == "Metropolitian").height,
                "action": "Normalize to 'Metropolitan'",
            },
            {
                "check": "Ratings outside 1–5",
                "count": cleaned.filter(pl.col("rating_out_of_range")).height,
                "action": "Flag and retain clipped copy",
            },
            {
                "check": "Courier age below 18",
                "count": cleaned.filter(pl.col("age_under_18")).height,
                "action": "Flag and retain clipped copy",
            },
        ]
    )
    quality_tbl
    return (quality_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The city typo affects the majority of rows and is safe to normalize.
    Ratings above 5.0 and ages below 18 are small enough to flag rather than
    drop. The cleaned frame keeps both raw and clipped values so the model
    training stage can compare them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Coordinates

    The coordinate fields contain two different failure modes: fully invalid
    pairs and sign-flipped restaurant coordinates. I flag both, keep the invalid
    rows, and create repaired restaurant coordinates for the sign-flipped cases.
    Distance is computed later in feature engineering.
    """)
    return


@app.cell(hide_code=True)
def _(cleaned, pl):
    coord_counts = (
        cleaned.group_by("coord_status")
        .len()
        .sort("coord_status")
    )
    coord_counts
    return (coord_counts,)


@app.cell(hide_code=True)
def _(THEME, coord_counts, p9, pl, save_plot):
    _order = ["valid", "sign_flipped", "invalid"]
    _plot_df = coord_counts.with_columns(
        pl.col("coord_status").cast(pl.Enum(_order))
    )
    _plot = (
        p9.ggplot(_plot_df, p9.aes(x="coord_status", y="len"))
        + p9.geom_col(fill="#4C78A8")
        + p9.coord_flip()
        + THEME
        + p9.theme(figure_size=(7, 4))
        + p9.labs(
            title="Coordinate validity status",
            subtitle="Invalid pairs are retained; distance is nulled in feature engineering",
            x="",
            y="Rows",
        )
    )
    _path = save_plot(_plot, "coord_status_counts.png", width=7, height=4)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Most rows are valid. The 431 sign-flipped rows are repaired by taking the
    absolute value of the restaurant coordinates; the 3,640 invalid pairs are
    retained but will not receive a distance value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. Target and leakage checks

    The target is complete and remains integer-valued. Pickup time and prep time
    are retained for diagnostics but marked as post-order fields; feature
    engineering excludes them from the model-available set.
    """)
    return


@app.cell(hide_code=True)
def _(cleaned, pl):
    target_tbl = cleaned.select(
        pl.col("target").min().alias("min"),
        pl.col("target").quantile(0.25).alias("p25"),
        pl.col("target").median().alias("median"),
        pl.col("target").mean().alias("mean"),
        pl.col("target").quantile(0.75).alias("p75"),
        pl.col("target").max().alias("max"),
        pl.col("target").null_count().alias("missing"),
    )
    target_tbl
    return (target_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The target remains between 10 and 54 minutes with a median of 26 and no
    missing values. The cleaning step has not changed the target distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 9. Validation summary

    I compare row counts, column counts, duplicate counts, missing values, and
    the target range before and after cleaning.
    """)
    return


@app.cell(hide_code=True)
def _(cleaned, pl, raw):
    validation_tbl = pl.DataFrame(
        [
            {
                "check": "Rows",
                "raw": raw.height,
                "cleaned": cleaned.height,
            },
            {
                "check": "Columns",
                "raw": raw.width,
                "cleaned": cleaned.width,
            },
            {
                "check": "Duplicate rows",
                "raw": raw.is_duplicated().sum(),
                "cleaned": cleaned.is_duplicated().sum(),
            },
            {
                "check": "Target missing",
                "raw": raw.filter(
                    raw["Time_taken (min)"].str.strip_chars() == "NaN"
                ).height,
                "cleaned": cleaned["target"].null_count(),
            },
            {
                "check": "Target median",
                "raw": round(
                    raw["Time_taken (min)"].cast(pl.Float64).median()
                ),
                "cleaned": round(cleaned["target"].median()),
            },
        ]
    )
    validation_tbl
    return (validation_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Row and column counts are stable where expected: the row count is unchanged,
    the column count grows because the cleaned frame adds parsed and flagged
    fields, and the target distribution is preserved.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 10. Output

    The canonical cleaned artifact is Parquet; the CSV copy is for convenience
    and inspection. The manifest records source and output hashes, schema, row
    counts, and the target definition.
    """)
    return


@app.cell(hide_code=True)
def _(cleaned, config, pl, write_clean_artifacts):
    manifest = write_clean_artifacts(cleaned)
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
                "path": str(config.CLEAN_MANIFEST_PATH.relative_to(config.PROJECT_ROOT)),
                "sha256": "",
            },
        ]
    )
    output_tbl
    return (manifest,)


@app.cell(hide_code=True)
def _(cleaned):
    cleaned.select(
        [
            "id",
            "order_date",
            "age_raw",
            "age_clipped",
            "rating_raw",
            "rating_clipped",
            "order_minutes",
            "pickup_minutes",
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
    ## Summary

    **Problems fixed**

    - Literal `"NaN"` strings are parsed as nulls.
    - Mixed time formats are normalized to minutes since midnight.
    - `Metropolitian` is normalized to `Metropolitan`.
    - Ratings above 5.0 and ages below 18 are flagged and preserved with clipped
      alternatives.
    - Invalid and sign-flipped coordinates are flagged; sign-flipped restaurant
      coordinates are repaired.
    - Duplicate rows are checked and none are removed.
    - The target is complete and unchanged.

    **Outputs**

    - `data/processed/zomato_clean.parquet`
    - `data/processed/zomato_clean.csv`
    - `data/processed/zomato_clean_manifest.json`

    **Next step:** `03_feature_engineering.py` builds the semantic feature frame
    and feature sets from the cleaned Parquet artifact.
    """)
    return


if __name__ == "__main__":
    app.run()
