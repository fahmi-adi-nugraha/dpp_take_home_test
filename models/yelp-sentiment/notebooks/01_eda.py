import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Yelp Sentiment EDA

    ## Problem

    The goal is to classify the sentiment of a Yelp review from its text. The raw target is a five-class star rating, and the eventual model will receive only the review text at inference time.

    This is an NLP classification problem. We check label balance, text quality, escape sequences, length distributions, tokenizer lengths, class vocabularies, and a quick TF-IDF baseline before BERT fine-tuning.

    ## Dataset

    - **Source:** [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full)
    - **Local snapshot:** 700,000 rows fetched with `uv run fetch-sentiment-data`
    - **Train split:** 650,000 rows
    - **Test split:** 50,000 rows
    - **Raw location:** `data/raw/yelp_review_full/`
    - **Target:** `label`, encoded as 0–4 for 1–5 stars
    - **Text field:** `text`
    - **Field reference:** [`../DATA_DICTIONARY.md`](../DATA_DICTIONARY.md)

    ### Features

    | Column | Description | Role |
    |---|---|---|
    | `label` | Star rating class: 0 = 1 star, 4 = 5 stars. | **Target** |
    | `text` | Raw review text, including literal escape sequences. | Feature |

    The raw labels are kept as five classes throughout EDA. Any collapse to binary sentiment is a modelling decision for the next step.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    import plotnine as p9
    import polars as pl
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.model_selection import train_test_split

    from yelp_sentiment import config

    try:
        NOTEBOOK_DIR = Path(__file__).resolve().parent
    except NameError:
        NOTEBOOK_DIR = Path.cwd() / "models" / "yelp-sentiment" / "notebooks"

    PROJECT_DIR = NOTEBOOK_DIR.parent
    RAW_DIR = PROJECT_DIR / "data" / "raw" / "yelp_review_full"
    TRAIN_PATH = RAW_DIR / "train-00000-of-00001.parquet"
    TEST_PATH = RAW_DIR / "test-00000-of-00001.parquet"
    IMG_DIR = NOTEBOOK_DIR / "imgs"
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    TOKENIZER_NAME = "bert-base-uncased"
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

    def decode_text(value: str) -> str:
        """Decode the literal JSON-style escapes in the raw review text."""
        try:
            return json.loads(f'"{value}"')
        except json.JSONDecodeError:
            return value

    return (
        LABEL_ORDER,
        LogisticRegression,
        TEST_PATH,
        THEME,
        TOKENIZER_NAME,
        TRAIN_PATH,
        TfidfVectorizer,
        accuracy_score,
        classification_report,
        config,
        confusion_matrix,
        decode_text,
        mo,
        np,
        p9,
        pl,
        save_plot,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(TEST_PATH, TRAIN_PATH, pl):
    train = pl.read_parquet(TRAIN_PATH)
    test = pl.read_parquet(TEST_PATH)
    all_reviews = pl.concat([train, test])
    return all_reviews, test, train


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Data overview

    The dataset has only two columns: the review text and the star rating. We'll start with the schema and a few truncated examples because displaying full review bodies would overwhelm the report.
    """)
    return


@app.cell(hide_code=True)
def _(pl, train):
    _schema_tbl = pl.DataFrame(
        {
            "column": train.columns,
            "dtype": [str(dtype) for dtype in train.schema.values()],
        }
    )
    _schema_tbl
    return


@app.cell(hide_code=True)
def _(pl, train):
    train.head(5).select(
        "label",
        pl.col("text").str.slice(0, 240).alias("text_excerpt"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The schema is exactly `label: Int64` and `text: String`. The examples read like real reviews, but the raw strings contain visible backslash sequences such as `\n` and `\"`. Those are literal characters in the parquet data, not formatting artifacts of the notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Preliminary checks

    The dataset is clean at the row level: there are no missing labels, no missing texts, no exact duplicates, and no exact-text overlap between the supplied train and test splits. The classes are exactly balanced.
    """)
    return


@app.cell(hide_code=True)
def _(all_reviews, pl, test, train):
    _train_texts = train.select("text").unique()
    _test_texts = test.select("text").unique()
    _overlap = _train_texts.join(_test_texts, on="text", how="inner").height
    checks_tbl = pl.DataFrame(
        {
            "check": [
                "train rows",
                "test rows",
                "total rows",
                "null labels",
                "null texts",
                "empty/whitespace texts",
                "exact duplicate rows",
                "duplicate review texts",
                "train/test exact-text overlap",
                "unique review texts",
            ],
            "value": [
                f"{train.height:,}",
                f"{test.height:,}",
                f"{all_reviews.height:,}",
                f"{train['label'].null_count() + test['label'].null_count():,}",
                f"{train['text'].null_count() + test['text'].null_count():,}",
                f"{int((all_reviews['text'].str.strip_chars() == '').sum()):,}",
                f"{all_reviews.height - all_reviews.unique().height:,}",
                f"{all_reviews.height - all_reviews['text'].n_unique():,}",
                f"{_overlap:,}",
                f"{all_reviews['text'].n_unique():,}",
            ],
        }
    )
    checks_tbl
    return


@app.cell(hide_code=True)
def _(all_reviews, config, pl, test, train):
    _train_counts = train.group_by("label").len().rename({"len": "train"})
    _test_counts = test.group_by("label").len().rename({"len": "test"})
    label_tbl = (
        _train_counts.join(_test_counts, on="label")
        .with_columns(
            (pl.col("train") + pl.col("test")).alias("total"),
            pl.col("label")
            .replace_strict(config.LABEL_MAP, return_dtype=pl.String)
            .alias("label_name"),
        )
        .with_columns((pl.col("total") / all_reviews.height * 100).round(2).alias("share_pct"))
        .sort("label")
        .select("label", "label_name", "train", "test", "total", "share_pct")
    )
    label_tbl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The five classes are exactly 20.00% each. Accuracy and macro F1 should therefore be close, but macro F1 and per-class precision/recall are still the right checks for uneven sentiment performance. There are no row-level cleaning blockers, so the next questions are about text structure.
    """)
    return


@app.cell(hide_code=True)
def _(decode_text, pl, train):
    _parts = [
        train.filter(pl.col("label") == label).sample(n=10_000, seed=42 + label)
        for label in range(5)
    ]
    sample_tbl = (
        pl.concat(_parts)
        .with_columns(
            pl.col("text").map_elements(decode_text, return_dtype=pl.String).alias("decoded_text")
        )
        .with_columns(
            pl.col("text").str.len_chars().alias("raw_chars"),
            pl.col("decoded_text").str.len_chars().alias("decoded_chars"),
            pl.col("decoded_text").str.extract_all(r"\S+").list.len().alias("decoded_words"),
        )
    )
    return (sample_tbl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Text structure and escape sequences

    The review text contains literal JSON-style escape sequences. This is a data-encoding issue, not prose. We quantify its prevalence and then show a raw-versus-decoded example.
    """)
    return


@app.cell(hide_code=True)
def _(pl, train):
    _escape_specs = [
        ("Any backslash", "\\"),
        ("Backslash-n", r"\n"),
        ("Backslash-quote", '\\"'),
        ("Backslash-u", r"\u"),
        ("Double backslash", "\\\\"),
        ("Backslash-r", r"\r"),
        ("Backslash-t", r"\t"),
    ]
    escape_tbl = pl.DataFrame(
        [
            {
                "sequence": _label,
                "rows": train.select(
                    pl.col("text").str.contains(_pattern, literal=True).sum()
                ).item(),
            }
            for _label, _pattern in _escape_specs
        ]
    ).with_columns((pl.col("rows") / train.height * 100).round(2).alias("share_pct"))
    escape_tbl
    return


@app.cell(hide_code=True)
def _(pl, sample_tbl):
    _example = sample_tbl.filter(
        pl.col("text").str.contains(r"\n", literal=True)
        & pl.col("text").str.contains('\\"', literal=True)
        & pl.col("decoded_chars").is_between(220, 420)
    ).head(1)
    _raw_text = _example["text"][0][:300]
    _decoded_text = _example["decoded_text"][0][:300]
    escape_example_tbl = pl.DataFrame(
        {
            "version": ["raw parquet text", "decoded text"],
            "excerpt": [_raw_text, _decoded_text],
        }
    )
    escape_example_tbl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Exactly 55.58% of all reviews contain at least one literal backslash, and 50.11% contain the two-character sequence `\n`. The raw strings are valid JSON string contents: every train and test row can be decoded cleanly. Unescaping should therefore happen before tokenization. The decoded text is slightly shorter, and the decoded version should be the model input.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Review length

    Length matters in two ways: very short reviews may be ambiguous, and very long reviews will be truncated by the tokenizer. We measure both characters and whitespace tokens on the raw text, then compare decoded lengths by class on a stratified sample.
    """)
    return


@app.cell(hide_code=True)
def _(all_reviews, pl):
    _chars = all_reviews.select(pl.col("text").str.len_chars().alias("chars"))["chars"]
    _words = all_reviews.select(pl.col("text").str.extract_all(r"\S+").list.len().alias("words"))[
        "words"
    ]
    length_tbl = pl.DataFrame(
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
                "std",
            ],
            "characters": [
                float(_chars.len()),
                float(_chars.min()),
                float(_chars.quantile(0.25)),
                float(_chars.median()),
                float(_chars.mean()),
                float(_chars.quantile(0.75)),
                float(_chars.quantile(0.90)),
                float(_chars.quantile(0.95)),
                float(_chars.quantile(0.99)),
                float(_chars.max()),
                float(_chars.std()),
            ],
            "words": [
                float(_words.len()),
                float(_words.min()),
                float(_words.quantile(0.25)),
                float(_words.median()),
                float(_words.mean()),
                float(_words.quantile(0.75)),
                float(_words.quantile(0.90)),
                float(_words.quantile(0.95)),
                float(_words.quantile(0.99)),
                float(_words.max()),
                float(_words.std()),
            ],
        }
    )
    length_tbl = length_tbl.with_columns(
        pl.col("characters").round(2),
        pl.col("words").round(2),
    )
    length_tbl
    return


@app.cell(hide_code=True)
def _(all_reviews, pl):
    _chars = all_reviews.select(pl.col("text").str.len_chars().alias("chars"))["chars"]
    _words = all_reviews.select(pl.col("text").str.extract_all(r"\S+").list.len().alias("words"))[
        "words"
    ]
    length_flags_tbl = pl.DataFrame(
        {
            "check": [
                "fewer than 20 characters",
                "fewer than 5 words",
                "more than 5,000 characters",
                "more than 1,000 words",
                "contains http",
                "contains www.",
            ],
            "rows": [
                int((_chars < 20).sum()),
                int((_words < 5).sum()),
                int((_chars > 5000).sum()),
                int((_words > 1000).sum()),
                int(
                    all_reviews.select(
                        pl.col("text").str.contains("http", literal=True).sum()
                    ).item()
                ),
                int(
                    all_reviews.select(
                        pl.col("text").str.contains("www.", literal=True).sum()
                    ).item()
                ),
            ],
        }
    ).with_columns((pl.col("rows") / all_reviews.height * 100).round(2).alias("share_pct"))
    length_flags_tbl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Review length is heavily right-skewed: the median is 539 characters, but the 99th percentile is 3,292 and the maximum is 5,637. Very short reviews exist too: 0.23% are under 20 characters and 0.45% are under five words. We'll keep them in the first baseline, then test whether filtering helps. Long reviews will require tokenizer truncation, which we analyse in the next section.
    """)
    return


@app.cell(hide_code=True)
def _(config, pl, sample_tbl):
    class_length_tbl = (
        sample_tbl.group_by("label")
        .agg(
            pl.len().alias("n"),
            pl.col("decoded_chars").mean().round(2).alias("mean_chars"),
            pl.col("decoded_chars").median().round(2).alias("median_chars"),
            pl.col("decoded_words").mean().round(2).alias("mean_words"),
            pl.col("decoded_words").median().round(2).alias("median_words"),
        )
        .with_columns(
            pl.col("label")
            .replace_strict(config.LABEL_MAP, return_dtype=pl.String)
            .alias("label_name")
        )
        .sort("label")
        .select(
            "label_name",
            "n",
            "mean_chars",
            "median_chars",
            "mean_words",
            "median_words",
        )
    )
    class_length_tbl
    return


@app.cell(hide_code=True)
def _(LABEL_ORDER, THEME, config, p9, pl, sample_tbl, save_plot):
    _plot_df = sample_tbl.with_columns(
        pl.col("label")
        .replace_strict(config.LABEL_MAP, return_dtype=pl.String)
        .cast(pl.Enum(LABEL_ORDER))
        .alias("label_name")
    )
    _plot = (
        p9.ggplot(
            _plot_df,
            p9.aes(x="label_name", y="decoded_chars", fill="label_name"),
        )
        + p9.geom_violin(alpha=0.5)
        + p9.geom_boxplot(width=0.1, outlier_size=0.4, outlier_alpha=0.25)
        + p9.coord_cartesian(ylim=(0, 2000))
        + THEME
        + p9.theme(figure_size=(8, 5), legend_position="none")
        + p9.labs(
            title="Decoded review length by star rating",
            subtitle="Stratified 50k sample; y-axis clipped at 2,000 characters",
            x="",
            y="Characters",
        )
    )
    _path = save_plot(_plot, "01_eda_length_by_label.png", width=8, height=5)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Lower-star reviews are longer on average: 1-star reviews average 819.20 decoded characters, while 5-star reviews average 580.10. The boxes also overlap heavily, so length is not a clean predictor. It is still a warning: aggressive truncation will remove more evidence from negative reviews, and we should check review length as a potential shortcut rather than add it as a feature by default.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. BERT token length

    Character counts are not the same as model tokens. We measured token lengths on the 50,000-review stratified sample with `bert-base-uncased`, including special tokens, before deciding a maximum sequence length.
    """)
    return


@app.cell(hide_code=True)
def _(TOKENIZER_NAME):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.model_max_length = 1_000_000
    return (tokenizer,)


@app.cell(hide_code=True)
def _(pl, sample_tbl, tokenizer):
    _texts = sample_tbl["decoded_text"].to_list()
    _encoded = tokenizer(
        _texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    _token_lengths = [len(ids) for ids in _encoded["input_ids"]]
    sample_token_tbl = sample_tbl.with_columns(pl.Series("token_len", _token_lengths))
    return (sample_token_tbl,)


@app.cell(hide_code=True)
def _(pl, sample_token_tbl):
    _tokens = sample_token_tbl["token_len"]
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
                f"{_tokens.len():,}",
                f"{int(_tokens.min()):,}",
                f"{int(_tokens.quantile(0.25)):,}",
                f"{int(_tokens.median()):,}",
                f"{float(_tokens.mean()):.2f}",
                f"{int(_tokens.quantile(0.75)):,}",
                f"{int(_tokens.quantile(0.90)):,}",
                f"{int(_tokens.quantile(0.95)):,}",
                f"{int(_tokens.quantile(0.99)):,}",
                f"{int(_tokens.max()):,}",
                f"{int((_tokens > 128).sum()):,} ({(_tokens > 128).mean() * 100:.2f}%)",
                f"{int((_tokens > 256).sum()):,} ({(_tokens > 256).mean() * 100:.2f}%)",
                f"{int((_tokens > 512).sum()):,} ({(_tokens > 512).mean() * 100:.2f}%)",
            ],
        }
    )
    token_stats_tbl
    return


@app.cell(hide_code=True)
def _(THEME, p9, sample_token_tbl, save_plot):
    _plot = (
        p9.ggplot(sample_token_tbl, p9.aes(x="token_len"))
        + p9.geom_histogram(bins=60, fill="#4C78A8", color="white")
        + p9.geom_vline(xintercept=128, color="#F58518", linetype="dashed")
        + p9.geom_vline(xintercept=256, color="#54A24B", linetype="dashed")
        + p9.geom_vline(xintercept=512, color="#D62728", linetype="dashed")
        + THEME
        + p9.theme(figure_size=(8, 4))
        + p9.labs(
            title="BERT token length distribution",
            subtitle="Dashed lines: 128 (orange), 256 (green), and 512 (red) tokens",
            x="Tokens including special tokens",
            y="Reviews",
        )
    )
    _path = save_plot(_plot, "01_eda_token_lengths.png", width=8, height=4)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The median review is 128.00 BERT tokens and the 95th percentile is 479.00\. Exactly 20.41% exceed 256 tokens and 4.17% exceed 512. A 256-token limit is a reasonable fast-iteration baseline, while 512 is the safer final configuration because it covers 95.83% of reviews. We'll use head-only truncation first and test head + tail truncation if long negative reviews still underperform.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Class vocabulary

    To check that the labels have learnable text signal, we compare class-specific TF-IDF scores against the overall sample mean. The terms below are the ones that are most distinctive for each star rating, not simply the most frequent words overall.
    """)
    return


@app.cell(hide_code=True)
def _(TfidfVectorizer, config, np, pl, sample_tbl):
    _vectorizer = TfidfVectorizer(
        max_features=5_000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.5,
        stop_words="english",
        sublinear_tf=True,
    )
    _matrix = _vectorizer.fit_transform(sample_tbl["decoded_text"].to_list())
    _terms = _vectorizer.get_feature_names_out()
    _labels = sample_tbl["label"].to_numpy()
    _overall = np.asarray(_matrix.mean(axis=0)).ravel()
    _rows = []
    for _label in range(5):
        _scores = np.asarray(_matrix[_labels == _label].mean(axis=0)).ravel() - _overall
        _top_indices = _scores.argsort()[-12:][::-1]
        _rows.append(
            {
                "label": _label,
                "label_name": config.LABEL_MAP[_label],
                "top_terms": ", ".join(_terms[index] for index in _top_indices),
            }
        )
    top_terms_tbl = pl.DataFrame(_rows)
    top_terms_tbl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The vocabulary separates the extremes clearly: 1-star reviews are dominated by `worst`, `horrible`, `rude`, `terrible`, and `manager`, while 5-star reviews are dominated by `great`, `best`, `amazing`, `love`, and `recommend`. The middle classes share generic terms such as `good`, `ok`, and `place`, which is consistent with the adjacent-class confusion expected in a five-class problem.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Baseline sanity check

    A quick TF-IDF + logistic regression model on a stratified 50,000-review sample gives a lower-bound check on how much signal the text alone contains. This is not the final model; it tells us whether BERT fine-tuning is worth the extra cost.
    """)
    return


@app.cell(hide_code=True)
def _(
    LogisticRegression,
    TfidfVectorizer,
    accuracy_score,
    classification_report,
    config,
    confusion_matrix,
    pl,
    sample_tbl,
    train_test_split,
):
    _texts = sample_tbl["decoded_text"].to_list()
    _labels = sample_tbl["label"].to_numpy()
    _text_train, _text_test, _y_train, _y_test = train_test_split(
        _texts,
        _labels,
        test_size=0.2,
        random_state=42,
        stratify=_labels,
    )
    _vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.5,
        stop_words="english",
        sublinear_tf=True,
    )
    _x_train = _vectorizer.fit_transform(_text_train)
    _x_test = _vectorizer.transform(_text_test)
    _model = LogisticRegression(max_iter=1000)
    _model.fit(_x_train, _y_train)
    _pred = _model.predict(_x_test)
    _report = classification_report(
        _y_test,
        _pred,
        output_dict=True,
        zero_division=0,
    )
    baseline_summary_tbl = pl.DataFrame(
        {
            "metric": ["accuracy", "macro_f1", "weighted_f1"],
            "value": [
                float(accuracy_score(_y_test, _pred)),
                float(_report["macro avg"]["f1-score"]),
                float(_report["weighted avg"]["f1-score"]),
            ],
        }
    )
    per_class_tbl = pl.DataFrame(
        [
            {
                "label_name": config.LABEL_MAP[label],
                "precision": float(_report[str(label)]["precision"]),
                "recall": float(_report[str(label)]["recall"]),
                "f1": float(_report[str(label)]["f1-score"]),
                "support": int(_report[str(label)]["support"]),
            }
            for label in range(5)
        ]
    ).with_columns(
        pl.col("precision").round(2),
        pl.col("recall").round(2),
        pl.col("f1").round(2),
    )
    _confusion = confusion_matrix(_y_test, _pred, labels=list(range(5)))
    confusion_tbl = pl.DataFrame(
        [
            {
                "true_label": true_label,
                "predicted_label": predicted_label,
                "true_name": config.LABEL_MAP[true_label],
                "predicted_name": config.LABEL_MAP[predicted_label],
                "count": int(_confusion[true_label, predicted_label]),
            }
            for true_label in range(5)
            for predicted_label in range(5)
        ]
    )
    return baseline_summary_tbl, confusion_tbl, per_class_tbl


@app.cell(hide_code=True)
def _(baseline_summary_tbl, pl):
    baseline_summary_tbl.with_columns(pl.col("value").round(2))
    return


@app.cell(hide_code=True)
def _(per_class_tbl):
    per_class_tbl
    return


@app.cell(hide_code=True)
def _(baseline_summary_tbl, mo):
    _summary = {row["metric"]: row["value"] for row in baseline_summary_tbl.iter_rows(named=True)}
    mo.md(
        f"""The TF-IDF baseline reaches **{_summary["accuracy"]:.2%}** accuracy and **{_summary["macro_f1"]:.2%}** macro F1 on a 10,000-review holdout from the sample. That is well above the 20.00% random baseline and confirms a strong text signal. The remaining errors are likely concentrated between adjacent star ratings, where language overlaps."""
    )
    return


@app.cell(hide_code=True)
def _(confusion_tbl):
    confusion_tbl
    return


@app.cell(hide_code=True)
def _(LABEL_ORDER, THEME, confusion_tbl, p9, pl, save_plot):
    _plot_df = confusion_tbl.with_columns(
        pl.col("true_name").cast(pl.Enum(LABEL_ORDER)),
        pl.col("predicted_name").cast(pl.Enum(LABEL_ORDER)),
    )
    _plot = (
        p9.ggplot(
            _plot_df,
            p9.aes(x="predicted_name", y="true_name", fill="count"),
        )
        + p9.geom_tile(color="white")
        + p9.geom_text(p9.aes(label="count"), size=8)
        + p9.scale_fill_gradient(low="#F7FBFF", high="#08306B")
        + THEME
        + p9.theme(
            figure_size=(6, 5),
            panel_grid=p9.element_blank(),
            legend_position="none",
        )
        + p9.labs(
            title="TF-IDF logistic regression confusion matrix",
            subtitle="10,000-review holdout from the stratified sample",
            x="Predicted",
            y="Actual",
        )
    )
    _path = save_plot(_plot, "01_eda_tfidf_confusion.png", width=6, height=5)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The extreme classes are easiest: 1-star and 5-star reviews are usually recognised correctly. Most errors sit between neighbouring classes, which is exactly the pattern expected from a five-class star rating. This supports keeping a five-class baseline first, while treating a binary negative-versus-positive formulation as a separate decision.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. Findings and recommendations

    ### Problems and proposed fixes

    - The raw text contains literal JSON-style escape sequences in 55.58% of reviews. Decode with `json.loads` before tokenization.
    - Reviews are long-tailed: the maximum is 5,637 characters, and 4.17% exceed 512 BERT tokens. Use a tokenizer limit and a documented truncation policy.
    - Very short reviews exist, including one-character strings. Keep them in the first baseline, then test filtering as an ablation.
    - The 3-star class is semantically ambiguous. Keep it as a five-class problem for the first baseline, or explicitly document the binary mapping before switching tasks.
    - There are no missing values, exact duplicates, or exact train/test overlaps. Near-duplicate text has not been checked yet.

    ### Potential features

    - The decoded review text is the primary feature.
    - No metadata is available in this dataset. Do not invent user, business, or date features that the model will not have at inference time.
    - Review length may be informative but is also a potential shortcut. Test it only as an ablation, not as a default input to BERT.

    ### Feature engineering methods

    - Decode escapes, then normalize whitespace only if it improves validation.
    - Preserve case, punctuation, negation, and stopwords for the BERT baseline.
    - Tokenize with the final model tokenizer. Use 256 tokens for fast iteration and 512 for the final candidate unless experiments justify otherwise.
    - Use the supplied train/test split, carve a stratified validation set from train, and keep the test set untouched.
    - Start with a five-class BERT model; compare against the TF-IDF baseline and report accuracy, macro F1, and per-class precision/recall.
    """)
    return


if __name__ == "__main__":
    app.run()
