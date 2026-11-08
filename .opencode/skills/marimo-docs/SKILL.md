---
name: Marimo Documentation
description: Write and maintain reviewer-facing marimo notebooks for EDA, data cleaning, feature engineering, and model training using this repository's Polars, plotnine, and cell-layout conventions. Use when creating, editing, reviewing, or restructuring marimo .py notebooks.
---

# Marimo Documentation

Use this skill for any marimo notebook that documents analysis or modeling
work. The notebook should read like a short technical report: a reviewer should
be able to follow the reasoning without reading every line of code.

## When to use

- writing or restructuring an EDA notebook,
- documenting data cleaning steps,
- documenting feature engineering decisions,
- documenting model training, evaluation, and selection,
- reviewing a marimo notebook for consistency and readability.

## Non-negotiable cell rules

1. Every cell uses:

   ```python
   @app.cell(hide_code=True)
   ```

   This includes markdown cells. Code and markdown source are hidden by default
   in edit mode and can be expanded per cell.

2. Use separate cells for:
   - markdown exposition,
   - data preparation/code,
   - dataframe display,
   - plot display.

3. One output per cell. Do not combine a markdown paragraph, a dataframe, and a
   plot in one cell.

4. Display dataframes directly as the last expression:

   ```python
   missing_tbl
   ```

   Do not use `mo.ui.table`.

5. Display plots directly as the last expression:

   ```python
   _plot
   ```

   Do not use `mo.image` in the report.

6. Do not use `mo.vstack` to combine outputs. Split the outputs into separate
   cells.

7. Do not use pandas or `to_pandas`. Prefer Polars throughout.

8. Pass Polars dataframes directly to plotnine. Do not convert them to pandas.

## Notebook structure

Use this order unless there is a strong reason to differ:

1. **Introduction**
   - Problem being solved.
   - Prediction point and target.
   - Dataset source and local snapshot.
   - Table of all features and what they mean.
   - Target feature called out explicitly.
   - Links to the data dictionary and the current feature-decision record.

2. **Data overview**
   - `raw.glimpse()`.
   - Parsed/cleaned `df.describe()` on a focused set of columns.
   - `raw.head(20)`.
   - A brief observation after the group.

3. **Preliminary checks**
   - Missing values.
   - Duplicate rows.
   - Basic target distribution.

4. **Deeper analysis**
   - Group related tables and plots.
   - Place the written observation immediately after the last cell in each
     group.

5. **Summary**
   - Problems identified and proposed fixes.
   - Potential features.
   - Potential feature engineering methods.
   - Unresolved questions are moved to the temporary checklist, not included in the report.

## Writing style

Write the report in the **first-person plural**, as a group: use “we”, “we'll”, and “our” rather than “I”, “I'll”, and “my”. For example, write “We'll start with the schema” rather than “I start with the schema”.

Keep prose brief and decision-oriented. Observations should explain:

- what is visible in the data,
- why it matters,
- what problem it creates,
- what cleaning or feature-engineering step could address it,
- whether a feature should be kept, dropped, or tested.

Place observations directly under the table/plot group they reference. Do not put them in a dedicated “what we notice” section. Do not add an observation under every cell; group related cells and comment once per group.

Use plain paragraphs, not a recurring bold heading. Write each Markdown paragraph on one continuous line; do not hard-wrap prose at 80–90 characters. Let the report renderer handle line width. Example:

```markdown
Traffic and weather both matter. Low traffic has a median around 20.00 minutes while Jam is around 31.00; Sunny weather is around 20.00 while Fog and Cloudy are around 28.00. Traffic should be ordinal, and weather should be categorical.
```

If a numeric value is the last word before a period and a line break would put the number at the start of the next line, escape the period (for example, `479\.`) so Markdown does not interpret it as an ordered-list marker. Keeping paragraphs on continuous lines is the preferred fix.

Round displayed numeric values to **two decimal places**, including numbers in tables derived from dataframes. Counts may remain integers, but percentages and floating-point statistics should use two decimals.

Avoid deep dives, long derivations, and restating the code. The notebook is a report, not a lab log.

## Marimo mechanics

- Use `app = marimo.App(width="full")`.
- Use `if __name__ == "__main__": app.run()`.
- A variable used in another cell must be returned from the cell that creates
  it:

  ```python
  missing_tbl
  return (missing_tbl,)
  ```

- Use underscore-prefixed names for cell-local variables:

  ```python
  _plot = ...
  _summary = ...
  _path = ...
  ```

- Use non-underscore names for objects shared across cells:

  ```python
  missing_tbl
  categorical_long
  valid_distances
  feature_frame
  results_tbl
  ```

- Do not recreate a dataframe in a later cell if it is already available from
  an earlier cell. Return it instead.
- Avoid multiple top-level definitions of the same non-underscore variable
  across cells; marimo will raise `MultipleDefinitionError`.
- Use `mo.md` for narrative cells.
- Use raw strings for markdown containing backslashes or LaTeX if needed.

## Tables and plots

### Dataframes

- Display a Polars dataframe directly.
- Return it if another cell needs it.
- Keep one table per cell.
- Use `pl.Enum` or `pl.Categorical` when category order matters.
- Use grouped summaries instead of dumping raw rows unless the raw rows are the
  point of the section.

### Plotnine

- Pass Polars dataframes directly.
- Set per-plot size:

  ```python
  + p9.theme(figure_size=(8, 5))
  ```

- Do not put `figure_size` in the shared global theme.
- Prefer readable sizes. A typical report plot is around `7–9 × 4–6` inches.
- Use `p9.coord_flip()` for horizontal bar charts.
- Use `pl.Enum` / `pl.Categorical` for controlled categorical order.
- Draw distribution comparisons as violin plots with a narrow box plot
  overlaid, not as standalone box plots:

  ```python
  p9.ggplot(df, p9.aes(x="group", y="value", fill="group"))
  + p9.geom_violin(alpha=0.5)
  + p9.geom_boxplot(width=0.1, outlier_size=0.4, outlier_alpha=0.25)
  ```

- When using `facet_wrap`, set `fill` to the same variable used for faceting so
  each panel keeps a consistent color. Keep `geom_violin` first and overlay the
  narrow `geom_boxplot`.
- Use `p9.theme(legend_position="none")` when the legend repeats the x-axis or
  faceting variable.
- Keep titles and subtitles specific enough to stand alone.

### Saving plots

You may save plots for your own inspection:

```python
_path = save_plot(_plot, "01_missing_values.png", width=8, height=5)
_plot
```

Do not display the saved image in the report. The cell output is the plot
object. Generated images belong in an ignored directory such as
`notebooks/imgs/`.

## EDA documentation

- Start with the problem, dataset, feature list, and target.
- Show schema, summary statistics, and first 20 rows before formal checks.
- Then check missing values and duplicates.
- Then analyze relationships, time patterns, distance/geospatial behavior,
  missingness patterns, and correlation.
- End with a summary of problems/fixes, potential features, and feature engineering.
- Keep an observation after each related group of tables/plots.

### EDA open questions

Do not include an “Open questions” section in the EDA report. Put unresolved questions in an untracked `feature_engineering_checklist.md` file at the model project root. That checklist is a temporary working document for the data scientist only; it is not a project artifact and is discarded at the end of the modeling phase. Do not link to it or mention its path in notebooks, READMEs, code, comments, commit messages, or any other documentation.

## Data cleaning documentation

A cleaning notebook should make the transformation auditable.

Required sections:

1. **Objective**
   - What is being cleaned and why.
   - Input dataset and target.

2. **Input snapshot**
   - Source, row count, column count, checksum or manifest link.
   - Known data-quality issues from the data dictionary.

3. **Issue inventory**
   - Use a table with: issue, evidence, proposed action, rationale,
     validation.

4. **Cleaning steps**
   - One logical step per section.
   - Show the issue, the transformation, and a before/after check.
   - Typical steps:
     - literal `"NaN"` to null,
     - duplicate removal,
     - type conversion,
     - time normalization,
     - category normalization,
     - outlier flags and clipping decisions,
     - coordinate validation and repair,
     - missingness indicators,
     - target checks.

5. **Target and leakage checks**
   - Confirm the target definition.
   - Explicitly identify and exclude post-outcome fields.

6. **Before/after summary**
   - Rows retained, rows removed, columns added,
   - missingness changes,
   - schema changes.

7. **Output**
   - Path to the cleaned dataset.
   - Schema of the output.
   - Link to the current feature-decision record and next notebook.

Do not just show a long code dump. Explain the decision behind each cleaning
step and show how it was validated.

## Feature engineering documentation

Required sections:

1. **Objective and target**
   - What the features need to support.
   - Prediction point and leakage constraints.

2. **Input data**
   - Cleaned dataset path and schema summary.

3. **Feature inventory**
   - Table with: feature, source columns, transformation, type, rationale,
     leakage risk, priority.

4. **Engineering steps**
   - Group related transformations.
   - Show the transformation code.
   - Show a small validation table or plot.
   - Add a brief observation after the group.

5. **Encoding decisions**
   - Ordinal vs nominal vs binary.
   - Missing indicators.
   - Rare-level grouping.
   - Native categorical handling if relevant.

6. **Leakage checks**
   - Explicitly list excluded fields.
   - Explain why each is unavailable at prediction time.

7. **Selection and ablation**
   - Which features are in the baseline.
   - Which features are optional.
   - Which ablations will decide contested features.
   - Metrics used to compare feature sets.

8. **Final feature set**
   - Ordered list of selected features.
   - Excluded features.
   - Output artifact path.
   - Link to the data dictionary and any current feature-decision record.

## Model training documentation

Required sections:

1. **Objective and metrics**
   - Task, target, prediction point,
   - primary and secondary metrics,
   - success criteria.

2. **Data split**
   - Train/validation/test strategy,
   - time-based split if relevant,
   - leakage controls.

3. **Preprocessing**
   - scikit-learn `Pipeline` / `ColumnTransformer`,
   - imputation, encoding, scaling,
   - note that preprocessing is reused at inference.

4. **Candidate models**
   - Model families and baselines,
   - hyperparameter search space,
   - tuning strategy.

5. **Results**
   - One results table with validation/test metrics,
   - runtime or model size if relevant,
   - comparison against baseline.

6. **Evaluation**
   - Predicted vs actual,
   - residual distribution,
   - error by segment (traffic, city, distance, etc.),
   - target spikes or other structure.

7. **Interpretation**
   - feature importance or permutation importance,
   - SHAP if useful,
   - caveats about correlation vs causation.

8. **Model selection**
   - chosen model and why,
   - artifact path and version,
   - model card notes.

9. **Reproducibility**
   - command to train,
   - lockfile and environment assumptions,
   - random seeds.

10. **Monitoring and retraining**
    - inputs to monitor,
    - drift signals,
    - retraining triggers.

## Validation

Before considering a notebook finished, run:

```bash
uv run marimo export html path/to/notebook.py -o /tmp/notebook.html
```

The export must complete without cell errors.

Also check:

- every cell has `hide_code=True`,
- no `mo.ui.table`, `mo.image`, or `mo.vstack`,
- no pandas or `to_pandas`,
- no duplicated top-level non-underscore variables,
- no commented-out scaffolding or backup files in the report,
- observations are placed under the relevant table/plot group,
- prose uses first-person plural (“we”), not “I”,
- Markdown paragraphs are written on continuous lines rather than hard-wrapped,
- displayed numeric values are rounded to two decimal places,
- summary includes problems, features, and engineering methods but no open-questions section,
- unresolved EDA questions live in the temporary untracked checklist,
- no notebook, README, code, comment, or other documentation references the temporary checklist.

## References

See `references/notebook-patterns.md` for compact cell and section templates.
