# Marimo Notebook Patterns

Compact patterns for reviewer-facing marimo notebooks. Copy only the sections
that fit the task.

## Notebook skeleton

```python
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Title

    ## Problem
    ...

    ## Dataset
    - Source: ...
    - Target: ...
    - Feature reference: ...

    ### Features
    | Column | Description | Role |
    |---|---|---|
    | ... | ... | ... |
    """)
    return


# setup, helpers, loading, sections...


if __name__ == "__main__":
    app.run()
```

## Setup cell

```python
@app.cell(hide_code=True)
def _():
    import math
    from pathlib import Path

    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")
    import plotnine as p9
    import polars as pl

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

    def save_plot(plot, filename: str, width: float = 8, height: float = 5) -> Path:
        path = IMG_DIR / filename
        p9.ggsave(plot, path, width=width, height=height, dpi=150, verbose=False)
        return path

    return IMG_DIR, THEME, mo, p9, pl, save_plot
```

## Shared dataframe cell

```python
@app.cell(hide_code=True)
def _(df, pl):
    summary_tbl = (
        df.group_by("segment")
        .agg(
            pl.len().alias("n"),
            pl.col("target").mean().alias("mean_target"),
        )
        .sort("mean_target")
    )
    summary_tbl
    return (summary_tbl,)
```

## Plot cell

```python
@app.cell(hide_code=True)
def _(THEME, p9, save_plot, summary_tbl):
    _plot = (
        p9.ggplot(summary_tbl, p9.aes(x="segment", y="mean_target"))
        + p9.geom_col(fill="#4C78A8")
        + p9.coord_flip()
        + THEME
        + p9.theme(figure_size=(8, 5))
        + p9.labs(
            title="Mean target by segment",
            x="",
            y="Mean target",
        )
    )
    _path = save_plot(_plot, "segment_target.png", width=8, height=5)
    _plot
    return
```

## Observation cell

Place directly after the related table/plot group. Keep it brief.

```python
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Low traffic has a median around 20 minutes while Jam is around 31. The
    difference is large enough to keep traffic as an ordinal feature. Weather
    shows a similar pattern, with Sunny much faster than Fog or Cloudy.
    """)
    return
```

## Data overview

```python
@app.cell(hide_code=True)
def _(raw):
    raw.glimpse()
    return
```

```python
@app.cell(hide_code=True)
def _(df):
    _columns = ["age", "rating", "target", "distance_km"]
    df.select(_columns).describe()
    return
```

```python
@app.cell(hide_code=True)
def _(raw):
    raw.head(20)
    return
```

## Data cleaning section template

```markdown
## Cleaning step: missing values

**Issue:** missing values are encoded as literal `"NaN"` strings.

**Action:** parse `"NaN"` as null and add missing indicators for fields where
missingness is informative.

**Validation:** compare missing counts before and after parsing.
```

```python
@app.cell(hide_code=True)
def _(df, pl):
    missing_tbl = (
        df.select(
            pl.col("age").null_count().alias("age_missing"),
            pl.col("rating").null_count().alias("rating_missing"),
        )
    )
    missing_tbl
    return (missing_tbl,)
```

```python
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Missingness is highest for ratings and age. Both are small enough to retain
    with indicators rather than dropping rows.
    """)
    return
```

## Feature engineering section template

Use a table to make the decision audit-friendly:

```markdown
## Feature: distance_km

- **Source:** restaurant and delivery coordinates.
- **Transformation:** Haversine distance after coordinate validation.
- **Rationale:** strong proxy for travel time.
- **Leakage risk:** none; coordinates are known at order placement.
- **Validation:** compare with and without the feature.
```

```python
@app.cell(hide_code=True)
def _(df, pl):
    feature_frame = df.with_columns(
        pl.col("distance_km").fill_null(pl.col("distance_km").median()).alias(
            "distance_km_filled"
        )
    )
    feature_frame.select("distance_km", "distance_km_filled")
    return (feature_frame,)
```

## Model training results table template

```python
@app.cell(hide_code=True)
def _(pl, results):
    results_tbl = pl.DataFrame(results).sort("validation_rmse")
    results_tbl
    return (results_tbl,)
```

```markdown
| Model | Validation RMSE | Validation MAE | Validation R² | Notes |
|---|---:|---:|---:|---|
| Baseline | ... | ... | ... | ... |
| XGBoost | ... | ... | ... | ... |
```

## Final summary template

```markdown
## Summary

### Problems and fixes

- ...

### Selected features

- ...

### Feature engineering methods

- ...
```

Do not add an “Open questions” section. Move unresolved questions to the temporary `feature_engineering_checklist.md` working file at the model project root, and do not reference that file anywhere in the repository.

## Validation checklist

- [ ] Every cell uses `@app.cell(hide_code=True)`.
- [ ] Markdown, dataframes, and plots are in separate cells.
- [ ] Dataframes and plots are displayed directly.
- [ ] No `mo.ui.table`, `mo.image`, or `mo.vstack`.
- [ ] No pandas or `to_pandas`.
- [ ] Polars dataframes are passed directly to plotnine.
- [ ] Per-plot `figure_size` is set.
- [ ] Shared objects are returned from the creating cell.
- [ ] Observations appear under the relevant table/plot group.
- [ ] Prose uses first-person plural (“we”), not “I”.
- [ ] Markdown paragraphs use continuous lines instead of hard wrapping.
- [ ] Displayed numbers are rounded to two decimal places.
- [ ] Final summary covers problems, features, and engineering methods but has no open-questions section.
- [ ] Unresolved EDA questions live only in the temporary untracked checklist.
- [ ] The temporary checklist is not referenced in code or documentation.
- [ ] `uv run marimo export html <notebook>.py -o /tmp/notebook.html` succeeds.
