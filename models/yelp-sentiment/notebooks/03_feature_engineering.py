import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Yelp Feature Engineering

    ## Objective

    We prepare the cleaned review data for five-class sentiment training. The model input is the cleaned review text only. We do not add review length, URL counts, punctuation counts, or other handcrafted text statistics by default.

    The feature phase assigns a class-stratified validation split, documents the tokenizer contract, verifies that test data remains untouched, and writes the model-ready Parquet frame.

    ## Input

    - **Cleaned data:** `data/processed/yelp_clean.parquet`
    - **Target:** `label`, five classes from 0–4
    - **Text field:** `text`
    - **Output:** `data/processed/yelp_features.parquet`
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
    from transformers import AutoTokenizer

    from yelp_sentiment import config, data, features

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

    return (
        AutoTokenizer,
        LABEL_ORDER,
        THEME,
        config,
        data,
        features,
        json,
        mo,
        p9,
        pl,
        save_plot,
    )


@app.cell(hide_code=True)
def _(config, data, json):
    clean_frame = data.load_clean_data()
    clean_manifest = json.loads(config.CLEAN_MANIFEST_PATH.read_text(encoding="utf-8"))
    return clean_frame, clean_manifest


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Input snapshot

    The cleaned dataset contains only the fields needed for modeling: review identifiers, the five-class label, and the cleaned review text.
    """)
    return


@app.cell(hide_code=True)
def _(clean_frame, pl):
    _snapshot_tbl = pl.DataFrame(
        [
            {"item": "rows", "value": f"{clean_frame.height:,}"},
            {"item": "columns", "value": f"{clean_frame.width:,}"},
            {"item": "unique review ids", "value": f"{clean_frame['review_id'].n_unique():,}"},
            {"item": "null labels", "value": f"{clean_frame['label'].null_count():,}"},
            {"item": "null texts", "value": f"{clean_frame['text'].null_count():,}"},
        ]
    )
    _snapshot_tbl
    return


@app.cell(hide_code=True)
def _(clean_frame, pl):
    _label_counts = clean_frame.group_by("label").len().sort("label")
    _label_counts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The cleaned data has 699,929 rows, 649,929 train rows, and 50,000 test rows. The five labels remain close to balanced after cleaning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Feature inventory

    | Feature | Source | Transformation | Model input |
    |---|---|---|---|
    | `text` | Cleaned review text | Tokenizer input | Yes |
    | `label` | Raw star rating | Five-class integer | Target |
    | `split` | Cleaning and validation assignment | Train/validation/test | Metadata |

    We exclude review length, word count, punctuation count, uppercase ratio, URL count, character n-grams, and metadata features. Those are potential shortcuts or handcrafted additions that are not part of the approved first baseline.
    """)
    return


@app.cell(hide_code=True)
def _(clean_frame, features):
    feature_frame = features.assign_splits(clean_frame)
    return (feature_frame,)


@app.cell(hide_code=True)
def _(feature_frame, features, pl):
    _counts = features.split_counts(feature_frame)
    split_tbl = pl.DataFrame(
        [{"split": split, "rows": count} for split, count in sorted(_counts.items())]
    )
    split_tbl
    return (split_tbl,)


@app.cell(hide_code=True)
def _(feature_frame, config, pl):
    split_label_tbl = (
        feature_frame.group_by("split", "label")
        .len()
        .with_columns(
            pl.col("label")
            .replace_strict(config.LABEL_MAP, return_dtype=pl.String)
            .alias("label_name")
        )
        .sort("split", "label")
        .select("split", "label_name", "len")
    )
    split_label_tbl
    return (split_label_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The split has 584,936 train reviews, 64,993 validation reviews, and 50,000 test reviews. Validation is stratified by class, and the test split remains untouched.
    """)
    return


@app.cell(hide_code=True)
def _(feature_frame, features, pl):
    _leakage = features.leakage_summary(feature_frame)
    leakage_tbl = pl.DataFrame(
        [
            {
                "check": "exact text overlap between test and non-test",
                "rows": _leakage["test_overlap_texts"],
            }
        ]
    )
    leakage_tbl
    return (leakage_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Tokenizer contract

    We use `bert-base-uncased` for the first baseline and measure token lengths on a stratified sample from the training split only. The fast iteration limit is 256 tokens; the final candidate uses 512. Head-only truncation is the first policy.
    """)
    return


@app.cell(hide_code=True)
def _(AutoTokenizer, config):
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    tokenizer.model_max_length = 1_000_000
    return (tokenizer,)


@app.cell(hide_code=True)
def _(feature_frame, pl, tokenizer, features):
    _train = feature_frame.filter(pl.col("split") == "train")
    _parts = [
        _train.filter(pl.col("label") == label).sample(
            n=10_000,
            seed=42 + label,
        )
        for label in range(5)
    ]
    token_sample = pl.concat(_parts)
    _lengths = features.token_lengths(token_sample["text"].to_list(), tokenizer)
    token_sample = token_sample.with_columns(pl.Series("token_len", _lengths))
    token_stats = features.token_length_stats_from_lengths(_lengths)
    return token_sample, token_stats


@app.cell(hide_code=True)
def _(pl, token_stats):
    token_stats_tbl = pl.DataFrame(
        {
            "statistic": [
                "rows",
                "min",
                "p25",
                "median",
                "mean",
                "p75",
                "p90",
                "p95",
                "p99",
                "max",
                "over 128 tokens",
                "over 256 tokens",
                "over 512 tokens",
            ],
            "value": [
                f"{token_stats['rows']:,}",
                f"{token_stats['min']:,}",
                f"{token_stats['p25']:.2f}",
                f"{token_stats['median']:.2f}",
                f"{token_stats['mean']:.2f}",
                f"{token_stats['p75']:.2f}",
                f"{token_stats['p90']:.2f}",
                f"{token_stats['p95']:.2f}",
                f"{token_stats['p99']:.2f}",
                f"{token_stats['max']:,}",
                f"{token_stats['over_128']:,} ({token_stats['share_over_128_pct']:.2f}%)",
                f"{token_stats['over_256']:,} ({token_stats['share_over_256_pct']:.2f}%)",
                f"{token_stats['over_512']:,} ({token_stats['share_over_512_pct']:.2f}%)",
            ],
        }
    )
    token_stats_tbl
    return


@app.cell(hide_code=True)
def _(THEME, p9, save_plot, token_sample):
    _plot = (
        p9.ggplot(token_sample, p9.aes(x="token_len"))
        + p9.geom_histogram(bins=60, fill="#4C78A8", color="white")
        + p9.geom_vline(xintercept=128, color="#F58518", linetype="dashed")
        + p9.geom_vline(xintercept=256, color="#54A24B", linetype="dashed")
        + p9.geom_vline(xintercept=512, color="#D62728", linetype="dashed")
        + THEME
        + p9.theme(figure_size=(8, 4))
        + p9.labs(
            title="Token length after cleaning",
            subtitle="Dashed lines: 128 (orange), 256 (green), and 512 (red) tokens",
            x="Tokens including special tokens",
            y="Reviews",
        )
    )
    _path = save_plot(_plot, "03_token_length_after_cleaning.png", width=8, height=4)
    _plot
    return


@app.cell(hide_code=True)
def _(mo, token_stats):
    mo.md(
        f"""
        The cleaned sample has a median of {token_stats["median"]:.2f} tokens and a 95th percentile of {token_stats["p95"]:.2f}. About {token_stats["share_over_256_pct"]:.2f}% of reviews exceed 256 tokens and {token_stats["share_over_512_pct"]:.2f}% exceed 512. The 256-token fast baseline and 512-token final candidate remain appropriate.
        """
    )
    return


@app.cell(hide_code=True)
def _(feature_frame, features, token_stats):
    feature_artifacts = features.write_feature_artifacts(
        feature_frame,
        token_stats=token_stats,
    )
    return (feature_artifacts,)


@app.cell(hide_code=True)
def _(feature_artifacts, pl):
    artifacts_tbl = pl.DataFrame(
        [
            {"artifact": "feature parquet", "path": str(feature_artifacts.parquet_path)},
            {"artifact": "feature manifest", "path": str(feature_artifacts.manifest_path)},
            {"artifact": "rows", "path": f"{feature_artifacts.manifest['output']['rows']:,}"},
        ]
    )
    artifacts_tbl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    ### Final feature set

    - `text`: cleaned review text
    - `label`: five-class target
    - `split`: train, validation, or test
    - `review_id`: traceability identifier

    ### Excluded features

    - Review length, word count, punctuation count, uppercase ratio, and URL count.
    - Any metadata not available at inference time.
    - Character n-grams or TF-IDF features for the BERT input.

    ### Output

    - Feature rows: 699,929
    - Split: 584,936 train / 64,993 validation / 50,000 test
    - Test overlap: 0
    - Output: `data/processed/yelp_features.parquet`
    - Manifest: `data/processed/yelp_features_manifest.json`
    """)
    return


if __name__ == "__main__":
    app.run()
