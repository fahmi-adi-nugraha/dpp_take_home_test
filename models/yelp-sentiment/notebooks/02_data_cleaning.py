import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Yelp Data Cleaning

    ## Objective

    We clean the raw Yelp review text before feature engineering. The cleaning step decodes the literal JSON-style escape sequences, collapses whitespace, validates the five-class target, removes the small number of duplicate reviews created by normalization, and prevents cleaned train/test overlap.

    We preserve case, punctuation, negation, stopwords, URLs, and very short reviews. The cleaned output is the input to the feature-engineering report.

    ## Input

    - **Source:** `data/raw/yelp_review_full/`
    - **Train split:** 650,000 raw reviews
    - **Test split:** 50,000 raw reviews
    - **Target:** `label`, encoded as 0–4 for 1–5 stars
    - **Text field:** `text`
    - **Output:** `data/processed/yelp_clean.parquet`
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

    from yelp_sentiment import config, data

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

    return LABEL_ORDER, THEME, config, data, mo, p9, pl, save_plot


@app.cell(hide_code=True)
def _(data):
    raw_train, raw_test = data.load_raw_data()
    return raw_test, raw_train


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Raw snapshot

    We start with the schema and raw quality checks. The raw dataset has only the review text and the star rating, so cleaning is primarily a text-normalization problem.
    """)
    return


@app.cell(hide_code=True)
def _(pl, raw_train):
    _schema_tbl = pl.DataFrame(
        {
            "column": raw_train.columns,
            "dtype": [str(dtype) for dtype in raw_train.schema.values()],
        }
    )
    _schema_tbl
    return


@app.cell(hide_code=True)
def _(pl, raw_train):
    raw_train.head(3).select(
        "label",
        pl.col("text").str.slice(0, 240).alias("text_excerpt"),
    )
    return


@app.cell(hide_code=True)
def _(pl, raw_test, raw_train):
    _train_texts = raw_train.select("text").unique()
    _test_texts = raw_test.select("text").unique()
    raw_checks_tbl = pl.DataFrame(
        {
            "check": [
                "train rows",
                "test rows",
                "null labels",
                "null texts",
                "exact duplicate train rows",
                "duplicate train texts",
                "raw train/test text overlap",
            ],
            "value": [
                f"{raw_train.height:,}",
                f"{raw_test.height:,}",
                f"{raw_train['label'].null_count() + raw_test['label'].null_count():,}",
                f"{raw_train['text'].null_count() + raw_test['text'].null_count():,}",
                f"{raw_train.height - raw_train.unique().height:,}",
                f"{raw_train.height - raw_train['text'].n_unique():,}",
                f"{_train_texts.join(_test_texts, on='text', how='inner').height:,}",
            ],
        }
    )
    raw_checks_tbl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The raw schema is exactly `label: Int64` and `text: String`. There are no missing values, exact duplicate rows, duplicate raw texts, or raw train/test overlap. The later duplicate problems appear only after escape decoding and whitespace normalization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Cleaning rules

    | Issue | Evidence | Action |
    |---|---|---|
    | Literal JSON escapes | 55.58% of reviews contain a backslash; 50.11% contain literal `\\n` | Decode with JSON string semantics |
    | Escaped whitespace | Decoded text can contain newlines, tabs, and repeated spaces | Collapse all whitespace runs to one space |
    | Case and punctuation | Sentiment cues such as `not`, `!`, and `?` | Preserve |
    | URLs | 2,735 train and 193 test reviews contain `http` | Keep in the first baseline |
    | Very short reviews | 1,540 train and 110 test reviews are under 20 characters | Keep; test filtering later |
    | Cleaned duplicates | 56 train groups after normalization | Deduplicate |
    | Cleaned train/test overlap | 12 cleaned texts | Remove train rows only |
    """)
    return


@app.cell(hide_code=True)
def _(data, raw_test, raw_train):
    clean_result = data.build_clean_dataset(raw_train, raw_test)
    return (clean_result,)


@app.cell(hide_code=True)
def _(clean_result, pl):
    _stats = clean_result.stats
    before_after_tbl = pl.DataFrame(
        [
            {"item": "raw train rows", "value": f"{_stats['raw_train_rows']:,}"},
            {"item": "raw test rows", "value": f"{_stats['raw_test_rows']:,}"},
            {"item": "clean train rows", "value": f"{_stats['clean_train_rows']:,}"},
            {"item": "clean test rows", "value": f"{_stats['clean_test_rows']:,}"},
            {
                "item": "conflicting duplicate rows removed",
                "value": f"{_stats['conflict_rows_removed']:,}",
            },
            {
                "item": "other duplicate rows removed",
                "value": f"{_stats['duplicate_rows_removed']:,}",
            },
            {
                "item": "train/test overlap rows removed",
                "value": f"{_stats['overlap_rows_removed']:,}",
            },
            {
                "item": "empty cleaned train texts",
                "value": f"{_stats['empty_texts_train']:,}",
            },
            {
                "item": "empty cleaned test texts",
                "value": f"{_stats['empty_texts_test']:,}",
            },
        ]
    )
    before_after_tbl
    return


@app.cell(hide_code=True)
def _(clean_result, config, pl):
    _rows = []
    for _label in range(5):
        _rows.append(
            {
                "label": config.LABEL_MAP[_label],
                "train_before": clean_result.stats["label_counts_before_train"][_label],
                "train_after": clean_result.stats["label_counts_after_train"][_label],
                "test_after": clean_result.stats["label_counts_test"][_label],
            }
        )
    label_after_tbl = pl.DataFrame(_rows)
    label_after_tbl
    return (label_after_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The cleaning step retains 649,929 train reviews and 50,000 test reviews. We remove 6 rows from three conflicting-label duplicate groups, 53 extra rows from non-conflicting duplicate groups, and 12 train rows whose cleaned text appears in test. The final class counts remain close to 130,000 per label.
    """)
    return


@app.cell(hide_code=True)
def _(clean_result, data, pl, raw_train):
    _raw_with_ids = data.add_review_ids(raw_train, "train")
    cleaned_examples_tbl = (
        _raw_with_ids.select(
            "review_id",
            pl.col("text").alias("raw_text"),
        )
        .join(
            clean_result.train.select(
                "review_id",
                pl.col("text").alias("clean_text"),
            ),
            on="review_id",
            how="inner",
        )
        .filter(pl.col("raw_text") != pl.col("clean_text"))
        .head(3)
        .select(
            "review_id",
            pl.col("raw_text").str.slice(0, 240).alias("raw_excerpt"),
            pl.col("clean_text").str.slice(0, 240).alias("clean_excerpt"),
        )
    )
    cleaned_examples_tbl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The examples show that decoding changes formatting, not meaning. Case and punctuation remain intact after whitespace normalization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Near-duplicate diagnostic

    Exact duplicates are handled above. We also check near-duplicates by lowercasing and replacing punctuation for diagnostics only; near-duplicates are not removed from the model data.
    """)
    return


@app.cell(hide_code=True)
def _(clean_result, data, pl):
    _near_groups = data.near_duplicate_groups(clean_result.train)
    _near_stats = data.near_duplicate_stats(clean_result.train)
    near_stats_tbl = pl.DataFrame(
        [
            {"metric": "normalized duplicate groups", "value": f"{_near_stats['groups']:,}"},
            {"metric": "rows in duplicate groups", "value": f"{_near_stats['rows']:,}"},
            {"metric": "largest group", "value": f"{_near_stats['max_group_size']:,}"},
        ]
    )
    near_stats_tbl
    return


@app.cell(hide_code=True)
def _(clean_result, data, pl):
    data.near_duplicate_groups(clean_result.train).head(10).select(
        pl.col("_normalized").str.slice(0, 140).alias("normalized_excerpt"),
        pl.col("n"),
        pl.col("n_labels"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The normalized diagnostic finds 210 groups covering 582 rows, with a largest group of 22. These are mostly short generic reviews or repeated phrases. They are reported for awareness but kept because the approved first baseline preserves the review population.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Validation summary

    We check that the cleaned data preserves the intended review population: no empty text, no test overlap, and no unexpected label values.
    """)
    return


@app.cell(hide_code=True)
def _(clean_result, pl):
    _train = clean_result.train
    _test = clean_result.test
    _train_chars = _train.select(pl.col("text").str.len_chars().alias("chars"))["chars"]
    _test_chars = _test.select(pl.col("text").str.len_chars().alias("chars"))["chars"]
    _train_urls = _train.select(pl.col("text").str.contains("http", literal=True).alias("has_url"))[
        "has_url"
    ]
    _test_urls = _test.select(pl.col("text").str.contains("http", literal=True).alias("has_url"))[
        "has_url"
    ]
    _overlap = (
        _train.select("text")
        .unique()
        .join(
            _test.select("text").unique(),
            on="text",
            how="inner",
        )
        .height
    )
    validation_tbl = pl.DataFrame(
        [
            {
                "check": "train reviews under 20 characters",
                "rows": int((_train_chars < 20).sum()),
                "share_pct": round(float((_train_chars < 20).mean() * 100), 2),
            },
            {
                "check": "test reviews under 20 characters",
                "rows": int((_test_chars < 20).sum()),
                "share_pct": round(float((_test_chars < 20).mean() * 100), 2),
            },
            {
                "check": "train reviews containing http",
                "rows": int(_train_urls.sum()),
                "share_pct": round(float(_train_urls.mean() * 100), 2),
            },
            {
                "check": "test reviews containing http",
                "rows": int(_test_urls.sum()),
                "share_pct": round(float(_test_urls.mean() * 100), 2),
            },
            {
                "check": "cleaned train/test overlap",
                "rows": _overlap,
                "share_pct": 0.00,
            },
        ]
    )
    validation_tbl
    return


@app.cell(hide_code=True)
def _(clean_result, data):
    clean_artifacts = data.write_clean_artifacts(clean_result)
    return (clean_artifacts,)


@app.cell(hide_code=True)
def _(clean_artifacts, pl):
    artifacts_tbl = pl.DataFrame(
        [
            {"artifact": "cleaned parquet", "path": str(clean_artifacts.parquet_path)},
            {"artifact": "cleaning manifest", "path": str(clean_artifacts.manifest_path)},
            {"artifact": "cleaned rows", "path": f"{clean_artifacts.manifest['output']['rows']:,}"},
        ]
    )
    artifacts_tbl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    ### Problems and fixes

    - Literal JSON escape sequences were decoded before any downstream text processing.
    - Whitespace was collapsed while preserving case, punctuation, negation, stopwords, and URLs.
    - The three conflicting-label duplicate groups were dropped entirely.
    - The 53 remaining duplicate groups were reduced to one row each.
    - The 12 train rows whose cleaned text appeared in test were removed from train only.
    - Very short reviews are retained for the first baseline.

    ### Output

    - Cleaned train rows: 649,929
    - Cleaned test rows: 50,000
    - Cleaned train/test overlap: 0
    - Output: `data/processed/yelp_clean.parquet`
    - Manifest: `data/processed/yelp_clean_manifest.json`
    """)
    return


if __name__ == "__main__":
    app.run()
