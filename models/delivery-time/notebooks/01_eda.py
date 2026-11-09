import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Delivery-Time EDA

    ## Problem

    The goal is to predict product delivery time for Zomato delivery orders at the point the order is placed. This is a regression problem: the output is an estimated delivery duration in minutes, not a class label. The model is intended for order-time ETA estimation, so pickup time and pickup delay are not available as inputs.

    ## Dataset

    - **Source:** [Zomato Delivery Operations Analytics Dataset](https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset)
    - **Local snapshot:** 45,584 rows and 20 columns, fetched with `uv run fetch-delivery-data`.
    - **Raw file:** `data/raw/zomato/Zomato Dataset.csv`
    - **Target:** `Time_taken (min)`
    - **Field reference:** [`../DATA_DICTIONARY.md`](../DATA_DICTIONARY.md)

    ### Features

    | Column | Description | Role |
    |---|---|---|
    | `ID` | Unique delivery identifier. | Identifier |
    | `Delivery_person_ID` | Unique courier identifier. | Identifier |
    | `Delivery_person_Age` | Courier age in years. | Feature |
    | `Delivery_person_Ratings` | Courier rating. | Feature |
    | `Restaurant_latitude` | Restaurant latitude. | Geospatial feature |
    | `Restaurant_longitude` | Restaurant longitude. | Geospatial feature |
    | `Delivery_location_latitude` | Delivery latitude. | Geospatial feature |
    | `Delivery_location_longitude` | Delivery longitude. | Geospatial feature |
    | `Order_Date` | Order date. | Date feature |
    | `Time_Orderd` | Time the order was placed (spelling as in the raw data). | Time feature |
    | `Time_Order_picked` | Time the order was picked up. | Post-order / leakage risk |
    | `Weather_conditions` | Weather at delivery time. | Categorical feature |
    | `Road_traffic_density` | Road traffic density. | Ordinal feature |
    | `Vehicle_condition` | Vehicle condition score (0–3). | Ordinal feature |
    | `Type_of_order` | Order type: Snack, Meal, Drinks, or Buffet. | Categorical feature |
    | `Type_of_vehicle` | Vehicle used for delivery. | Categorical feature |
    | `multiple_deliveries` | Number of deliveries made in the same trip. | Count feature |
    | `Festival` | Whether the delivery coincided with a festival. | Binary feature |
    | `City` | City or area type. | Categorical feature |
    | `Time_taken (min)` | Time taken for delivery in minutes. | **Target** |

    All raw columns are read as text; missing values are literal `"NaN"` strings, not empty cells.
    """)
    return


@app.cell(hide_code=True)
def _():
    import math
    import re
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")
    import plotnine as p9
    import polars as pl

    pl.Config.set_float_precision(2)

    try:
        NOTEBOOK_DIR = Path(__file__).resolve().parent
    except NameError:
        NOTEBOOK_DIR = Path.cwd() / "models" / "delivery-time" / "notebooks"

    PROJECT_DIR = NOTEBOOK_DIR.parent
    RAW_PATH = PROJECT_DIR / "data" / "raw" / "zomato" / "Zomato Dataset.csv"
    IMG_DIR = NOTEBOOK_DIR / "imgs"
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    MISSING_TOKENS = {"NaN", "nan", "", "None", None}

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
        MISSING_TOKENS,
        RAW_PATH,
        THEME,
        datetime,
        math,
        mo,
        p9,
        pl,
        re,
        save_plot,
    )


@app.cell(hide_code=True)
def _(MISSING_TOKENS, datetime, math, pl, re):
    def is_missing(value) -> bool:
        return value is None or str(value).strip() in MISSING_TOKENS

    def as_float(value) -> float | None:
        return None if is_missing(value) else float(str(value).strip())

    def as_int(value) -> int | None:
        return None if is_missing(value) else int(float(str(value).strip()))

    def parse_time_minutes(value) -> float | None:
        if is_missing(value):
            return None
        text = str(value).strip()
        match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
        if match:
            hour, minute, second = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3) or 0),
            )
            return hour * 60 + minute + second / 60
        try:
            return float(text) * 24 * 60
        except ValueError:
            return None

    def time_format(value) -> str:
        if is_missing(value):
            return "missing"
        text = str(value).strip()
        if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", text):
            return "HH:MM:SS"
        if re.fullmatch(r"\d{1,2}:\d{2}", text):
            return "HH:MM"
        if text == "1":
            return "1 (24:00)"
        try:
            float(text)
            return "fractional day"
        except ValueError:
            return "other"

    def haversine(lat1, lon1, lat2, lon2) -> float | None:
        if None in (lat1, lon1, lat2, lon2):
            return None
        radius = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        return 2 * radius * math.asin(math.sqrt(a))

    def build_analysis_frame(raw: pl.DataFrame) -> pl.DataFrame:
        rows = []
        for record in raw.iter_rows(named=True):
            rlat = as_float(record["Restaurant_latitude"])
            rlon = as_float(record["Restaurant_longitude"])
            dlat = as_float(record["Delivery_location_latitude"])
            dlon = as_float(record["Delivery_location_longitude"])
            order_min = parse_time_minutes(record["Time_Orderd"])
            picked_min = parse_time_minutes(record["Time_Order_picked"])
            prep_min = None
            if order_min is not None and picked_min is not None:
                prep_min = picked_min - order_min
                if prep_min < 0:
                    prep_min += 1440

            order_date = datetime.strptime(record["Order_Date"].strip(), "%d-%m-%Y")
            invalid_coords = (rlat == 0 and rlon == 0) or (
                dlat is not None
                and dlon is not None
                and dlat == dlon
                and 0 < dlat < 1
            )
            sign_flipped = (
                rlat is not None and dlat is not None and rlat < 0 and dlat > 0
            )
            repair_lat = abs(rlat) if sign_flipped and rlat is not None else rlat
            repair_lon = abs(rlon) if sign_flipped and rlon is not None else rlon
            distance_km = None
            if not invalid_coords:
                distance_km = haversine(repair_lat, repair_lon, dlat, dlon)
                if distance_km is not None and not (0 <= distance_km <= 200):
                    distance_km = None

            age = as_float(record["Delivery_person_Age"])
            rating = as_float(record["Delivery_person_Ratings"])
            multiple = as_int(record["multiple_deliveries"])
            city = record["City"].strip()
            if city == "Metropolitian":
                city = "Metropolitan"
            order_hour = None
            if order_min is not None:
                order_hour = int((order_min // 60) % 24)

            rows.append(
                {
                    "id": record["ID"],
                    "courier_id": record["Delivery_person_ID"],
                    "age": age,
                    "rating": rating,
                    "weather": (
                        None
                        if is_missing(record["Weather_conditions"])
                        else record["Weather_conditions"].strip()
                    ),
                    "traffic": (
                        None
                        if is_missing(record["Road_traffic_density"])
                        else record["Road_traffic_density"].strip()
                    ),
                    "vehicle_condition": as_int(record["Vehicle_condition"]),
                    "order_type": (
                        None
                        if is_missing(record["Type_of_order"])
                        else record["Type_of_order"].strip()
                    ),
                    "vehicle_type": (
                        None
                        if is_missing(record["Type_of_vehicle"])
                        else record["Type_of_vehicle"].strip()
                    ),
                    "multiple": multiple,
                    "festival": (
                        None
                        if is_missing(record["Festival"])
                        else record["Festival"].strip()
                    ),
                    "city": None if is_missing(record["City"]) else city,
                    "target": as_int(record["Time_taken (min)"]),
                    "order_min": order_min,
                    "picked_min": picked_min,
                    "prep_min": prep_min,
                    "order_hour": order_hour,
                    "day_of_week": order_date.weekday(),
                    "is_weekend": order_date.weekday() >= 5,
                    "is_rush_hour": (
                        False
                        if order_hour is None
                        else (11 <= order_hour < 14 or 17 <= order_hour < 21)
                    ),
                    "is_meal_period": (
                        False if order_hour is None else 7 <= order_hour < 22
                    ),
                    "invalid_coords": invalid_coords,
                    "sign_flipped": sign_flipped,
                    "distance_km": distance_km,
                    "age_missing": age is None,
                    "rating_missing": rating is None,
                    "time_order_missing": order_min is None,
                    "weather_missing": is_missing(record["Weather_conditions"]),
                    "traffic_missing": is_missing(record["Road_traffic_density"]),
                    "multiple_missing": multiple is None,
                    "festival_missing": is_missing(record["Festival"]),
                    "city_missing": is_missing(record["City"]),
                }
            )

        frame = pl.DataFrame(rows)
        courier_counts = (
            raw.group_by("Delivery_person_ID")
            .len()
            .rename(
                {
                    "Delivery_person_ID": "courier_id",
                    "len": "courier_delivery_count",
                }
            )
        )
        frame = frame.join(courier_counts, on="courier_id", how="left")
        frame = frame.with_columns(
            (pl.col("multiple") > 0).alias("has_multiple_deliveries"),
            pl.when(pl.col("invalid_coords"))
            .then(pl.lit("invalid"))
            .when(pl.col("sign_flipped"))
            .then(pl.lit("sign_flipped"))
            .otherwise(pl.lit("valid"))
            .alias("coord_status"),
        )
        return frame

    def long_box(frame: pl.DataFrame, features, labels) -> pl.DataFrame:
        frames = []
        for column, label in zip(features, labels):
            value_frame = (
                frame.select(["target", column])
                .rename({column: "value"})
                .with_columns(pl.lit(label).alias("feature"))
            )
            value_frame = value_frame.with_columns(
                pl.col("value").cast(pl.String).fill_null("missing")
            )
            frames.append(value_frame)
        return pl.concat(frames)

    def spearman_matrix(frame: pl.DataFrame, columns) -> pl.DataFrame:
        rows = []
        for first in columns:
            for second in columns:
                if first == second:
                    value = 1.0
                else:
                    pair = frame.select([first, second]).drop_nulls()
                    if pair.height < 3:
                        value = None
                    else:
                        value = pair.select(
                            pl.corr(pl.col(first).rank(), pl.col(second).rank())
                        ).item()
                rows.append({"var1": first, "var2": second, "corr": value})
        return pl.DataFrame(rows)

    return build_analysis_frame, long_box, spearman_matrix, time_format


@app.cell(hide_code=True)
def _(RAW_PATH, build_analysis_frame, mo, pl):
    raw = pl.read_csv(RAW_PATH, infer_schema_length=0)
    df = build_analysis_frame(raw)
    mo.md(
        f"""
    Loaded **{raw.height:,} rows × {raw.width} columns** from `{RAW_PATH}`.

    The analysis frame adds parsed dates, normalized times, derived time features, distance, coordinate-quality flags, missingness indicators, and a courier delivery count.
    """
    )
    return df, raw


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Data overview

    We start with the schema, summary statistics, and the first rows. This gives an initial sense of the data types, ranges, and obvious data-entry issues before the formal missing-value and duplicate checks.
    """)
    return


@app.cell(hide_code=True)
def _(raw):
    raw.glimpse()
    return


@app.cell(hide_code=True)
def _(df):
    _overview_columns = [
        "age",
        "rating",
        "vehicle_condition",
        "multiple",
        "target",
        "order_min",
        "prep_min",
        "distance_km",
        "order_hour",
    ]
    df.select(_overview_columns).describe()
    return


@app.cell(hide_code=True)
def _(raw):
    raw.head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The raw columns are all strings, and `"NaN"` is stored as literal text rather than a null. The snapshot contains 45,584 rows and 20 columns, with 45,584 unique delivery IDs and 1,320 unique couriers; order dates span 2022-02-11 through 2022-04-06 across 44 unique dates. The target is integer-valued from 10 to 54 minutes, and the order-time columns already show mixed representations. The coordinate columns also contain exact zero values, which will need a validity check before distance is computed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Missing values

    Missing values are stored as the literal string `"NaN"`, not as empty cells. The target has no missing values; the largest gaps are in courier ratings, courier age, and order time.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):
    _missing_spec = [
        ("Delivery_person_Age", "age"),
        ("Delivery_person_Ratings", "rating"),
        ("Time_Orderd", "order_min"),
        ("City", "city"),
        ("multiple_deliveries", "multiple"),
        ("Weather_conditions", "weather"),
        ("Road_traffic_density", "traffic"),
        ("Festival", "festival"),
    ]
    missing_tbl = pl.DataFrame(
        [
            {
                "column": _label,
                "missing": df[_column].null_count(),
                "pct": df[_column].null_count() / df.height * 100,
            }
            for _label, _column in _missing_spec
        ]
    ).sort("pct", descending=True)
    missing_tbl
    return (missing_tbl,)


@app.cell(hide_code=True)
def _(THEME, missing_tbl, p9, pl, save_plot):
    _plot_df = missing_tbl.sort("pct")
    _plot_df = _plot_df.with_columns(
        pl.col("column").cast(pl.Enum(_plot_df["column"].to_list()))
    )
    _plot = (
        p9.ggplot(_plot_df, p9.aes(x="column", y="pct"))
        + p9.geom_col(fill="#4C78A8")
        + p9.coord_flip()
        + THEME
        + p9.theme(figure_size=(8, 5))
        + p9.labs(
            title="Missing values by column",
            subtitle='Percentage of rows with the literal string "NaN"',
            x="Column name",
            y="Missing (%)",
        )
    )
    _path = save_plot(_plot, "01_missing_values.png", width=8, height=5)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Ratings, age, and order time each have roughly 3.80–4.20% missing values; city has 2.60%, multiple deliveries 2.20%, weather and traffic about 1.30%, and festival 0.50%. Across the 20 columns, 9,131 cells are missing. These are literal `"NaN"` strings, so the cleaning step should parse them as nulls and add missingness indicators where the missing pattern is informative.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Duplicate rows and target distribution

    There are no duplicate rows. The target is integer-valued from 10 to 54 minutes and shows heaping at round values.
    """)
    return


@app.cell(hide_code=True)
def _(mo, raw):
    mo.md(f"""
    Duplicate rows: **{raw.is_duplicated().sum()}**. Unique delivery IDs: **{raw['ID'].n_unique():,}**. Rows: **{raw.height:,}**.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):
    _target_summary = df.select(
        pl.col("target").min().alias("min"),
        pl.col("target").quantile(0.25).alias("p25"),
        pl.col("target").median().alias("median"),
        pl.col("target").mean().alias("mean"),
        pl.col("target").quantile(0.75).alias("p75"),
        pl.col("target").max().alias("max"),
        pl.col("target").std().alias("std"),
    )
    _target_summary
    return


@app.cell(hide_code=True)
def _(THEME, df, p9, save_plot):
    _target_counts = df.group_by("target").len().sort("target")
    _plot = (
        p9.ggplot(_target_counts, p9.aes(x="target", y="len"))
        + p9.geom_col(fill="#4C78A8", width=0.8)
        + p9.geom_vline(
            xintercept=df["target"].mean(), color="#D62728", linetype="dashed"
        )
        + p9.geom_vline(
            xintercept=df["target"].median(), color="#2CA02C", linetype="dotted"
        )
        + THEME
        + p9.theme(figure_size=(7, 5))
        + p9.labs(
            title="Target distribution: Time_taken (min)",
            subtitle=(
                f"mean={df['target'].mean():.2f}, "
                f"median={df['target'].median():.2f}, n={df.height:,}; "
                "dashed=mean, dotted=median"
            ),
            x="Minutes",
            y="Count",
        )
    )
    _path = save_plot(_plot, "02_target_distribution.png", width=7, height=5)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    There are no duplicate rows. The target is bounded between 10 and 54 minutes with a median of 26 and visible spikes at round values such as 15, 25, and 35. The spikes look like rounding or heaping rather than separate populations. Regression is still reasonable, but residuals should be checked for structure around those values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Class balance and non-standard values

    The target is continuous, so there are no target classes. The categorical checks show severe imbalance in City, vehicle type, vehicle condition 3, and Festival. The raw data also contains inconsistent time formats, a city spelling typo, out-of-range courier attributes, and invalid coordinates.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):
    _categorical_features = [
        ("weather", "weather"),
        ("traffic", "traffic"),
        ("vehicle_condition", "vehicle_condition"),
        ("order_type", "order_type"),
        ("vehicle_type", "vehicle_type"),
        ("multiple", "multiple"),
        ("festival", "festival"),
        ("city", "city"),
    ]
    _frames = []
    for _column, _label in _categorical_features:
        _counts = df.group_by(_column).len().rename({_column: "value"})
        _counts = _counts.with_columns(
            pl.lit(_label).alias("feature"),
            (pl.col("len") / df.height * 100).alias("share"),
            pl.col("value").cast(pl.String).fill_null("missing"),
        )
        _frames.append(_counts.select(["feature", "value", "len", "share"]))
    categorical_long = pl.concat(_frames).sort(
        ["feature", "share"], descending=[False, True]
    )
    categorical_long
    return (categorical_long,)


@app.cell(hide_code=True)
def _(THEME, categorical_long, p9, pl, save_plot):
    _labels = [
        "weather",
        "traffic",
        "vehicle_condition",
        "order_type",
        "vehicle_type",
        "multiple",
        "festival",
        "city",
    ]
    _plot_df = categorical_long.with_columns(
        pl.col("feature").cast(pl.Enum(_labels)),
        pl.col("value").cast(pl.Categorical),
    )
    _plot = (
        p9.ggplot(_plot_df, p9.aes(x="value", y="share"))
        + p9.geom_col(fill="#4C78A8")
        + p9.facet_wrap("feature", scales="free", ncol=4)
        + THEME
        + p9.theme(
            figure_size=(9, 6),
            axis_text_x=p9.element_text(rotation=35, hjust=1),
        )
        + p9.labs(
            title="Categorical balance",
            subtitle="Share of rows per level, including missing values",
            x="",
            y="Share (%)",
        )
    )
    _path = save_plot(_plot, "03_categorical_balance.png", width=9, height=6)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Several categories are rare: Semi-Urban accounts for only 0.36% of rows, bicycle for 0.15%, vehicle condition 3 for 1.14%, and Festival Yes for 1.97%. Those levels can carry real signal, but they are small enough that encoding choices and regularization matter.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl, raw):
    _anomaly_tbl = pl.DataFrame(
        [
            {
                "check": "City uses 'Metropolitian' spelling",
                "count": raw.filter(
                    raw["City"].str.strip_chars() == "Metropolitian"
                ).height,
                "note": "Normalize to 'Metropolitan'",
            },
            {
                "check": "Ratings above 5.0",
                "count": df.filter(pl.col("rating") > 5).height,
                "note": "Ratings range up to 6.0",
            },
            {
                "check": "Courier age below 18",
                "count": df.filter(pl.col("age") < 18).height,
                "note": "Minimum observed age is 15",
            },
            {
                "check": "Invalid paired coordinates",
                "count": df.filter(pl.col("invalid_coords")).height,
                "note": "Restaurant (0,0) and delivery lat=lon below 1.0",
            },
            {
                "check": "Sign-flipped restaurant coordinates",
                "count": df.filter(pl.col("sign_flipped")).height,
                "note": "Negative restaurant latitude while delivery latitude is positive",
            },
            {
                "check": "Literal NaN cells in Time_Orderd",
                "count": raw.filter(
                    raw["Time_Orderd"].str.strip_chars() == "NaN"
                ).height,
                "note": "Pickup time is complete",
            },
        ]
    )
    _anomaly_tbl
    return


@app.cell(hide_code=True)
def _(pl, raw, time_format):
    _time_frames = []
    for _column in ["Time_Orderd", "Time_Order_picked"]:
        _formats = raw[_column].map_elements(time_format, return_dtype=pl.String)
        _counts = _formats.value_counts(sort=True).rename({"count": "n"})
        _counts = _counts.with_columns(
            pl.lit(_column).alias("column"),
            (pl.col("n") / raw.height * 100).alias("share"),
        )
        _counts = _counts.rename({_column: "format"})
        _time_frames.append(_counts.select(["column", "format", "n", "share"]))
    time_long = pl.concat(_time_frames)
    time_long
    return (time_long,)


@app.cell(hide_code=True)
def _(THEME, p9, save_plot, time_long):
    _plot = (
        p9.ggplot(time_long, p9.aes(x="column", y="share", fill="format"))
        + p9.geom_col(position="stack")
        + THEME
        + p9.theme(figure_size=(7, 5))
        + p9.labs(
            title="Raw time-format composition",
            subtitle="The time columns mix HH:MM, fractional-day, and after-midnight formats",
            x="",
            y="Share of rows (%)",
        )
    )
    _path = save_plot(_plot, "09_time_format_composition.png", width=7, height=5)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The time columns are not consistently parsed: they mix `HH:MM`, fractional-day values, `1` meaning 24:00, and `24:xx:00` values. Other issues include the `Metropolitian` spelling, ratings up to 6.00, 38 courier ages below 18, 3,640 invalid coordinate pairs, and 431 sign-flipped restaurant coordinates. All 53 ratings above 5.00 are exactly 6.00, and 162 of the 431 sign-flipped rows also have a negative restaurant longitude while the delivery longitude is positive. The cleaning plan should normalize the time strings and city spelling, flag or clip out-of-range values, and repair or flag coordinates before computing distance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Target relationships

    Traffic, weather, city, festival, multiple deliveries, vehicle condition, and vehicle type show clear differences in delivery time.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):
    _summary = (
        df.group_by("traffic")
        .agg(
            pl.len().alias("n"),
            pl.col("target").mean().alias("mean_target"),
            pl.col("target").median().alias("median_target"),
        )
        .sort("mean_target")
    )
    _summary
    return


@app.cell(hide_code=True)
def _(df, pl):
    _summary = (
        df.group_by("weather")
        .agg(
            pl.len().alias("n"),
            pl.col("target").mean().alias("mean_target"),
            pl.col("target").median().alias("median_target"),
        )
        .sort("mean_target")
    )
    _summary
    return


@app.cell(hide_code=True)
def _(THEME, df, long_box, p9, save_plot):
    _box_df = long_box(df, ["traffic", "weather"], ["Traffic density", "Weather"])
    _plot = (
        p9.ggplot(_box_df, p9.aes(x="value", y="target", fill="feature"))
        + p9.geom_violin(alpha=0.5)
        + p9.geom_boxplot(width=0.1, outlier_size=0.4, outlier_alpha=0.25)
        + p9.facet_wrap("feature", scales="free_x", ncol=2)
        + THEME
        + p9.theme(
            figure_size=(8, 5),
            axis_text_x=p9.element_text(rotation=25, hjust=1),
            legend_position="none",
        )
        + p9.labs(
            title="Delivery time by traffic and weather",
            x="",
            y="Time taken (min)",
        )
    )
    _path = save_plot(
        _plot, "04_target_traffic_weather.png", width=8, height=5
    )
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Traffic and weather both matter. Low traffic has a median around 20 minutes while Jam is around 31; Sunny weather is around 20 while Fog and Cloudy are around 28. Traffic should be treated as ordinal, and weather should be kept as a categorical feature.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):
    _summary = (
        df.group_by("city")
        .agg(
            pl.len().alias("n"),
            pl.col("target").mean().alias("mean_target"),
            pl.col("target").median().alias("median_target"),
        )
        .sort("mean_target")
    )
    _summary
    return


@app.cell(hide_code=True)
def _(df, pl):
    _summary = (
        df.group_by("multiple")
        .agg(
            pl.len().alias("n"),
            pl.col("target").mean().alias("mean_target"),
            pl.col("target").median().alias("median_target"),
        )
        .sort("multiple")
    )
    _summary
    return


@app.cell(hide_code=True)
def _(THEME, df, long_box, p9, save_plot):
    _box_df = long_box(
        df,
        ["vehicle_type", "city", "festival", "multiple"],
        ["Vehicle type", "City", "Festival", "Multiple deliveries"],
    )
    _plot = (
        p9.ggplot(_box_df, p9.aes(x="value", y="target", fill="feature"))
        + p9.geom_violin(alpha=0.5)
        + p9.geom_boxplot(width=0.1, outlier_size=0.4, outlier_alpha=0.25)
        + p9.facet_wrap("feature", scales="free", ncol=2)
        + THEME
        + p9.theme(
            figure_size=(8, 6),
            axis_text_x=p9.element_text(rotation=25, hjust=1),
            legend_position="none",
        )
        + p9.labs(
            title="Delivery time by vehicle, city, festival, and multiple deliveries",
            x="",
            y="Time taken (min)",
        )
    )
    _path = save_plot(
        _plot, "05_target_vehicle_city_festival_multiple.png", width=8, height=6
    )
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    City, festival, and multiple deliveries show the largest effects here: Semi-Urban has a median around 49 minutes, Festival Yes around 45, and multiple deliveries 3 around 48. Vehicle type is a more modest signal. City and festival should be one-hot encoded; multiple deliveries can use both the count and a binary flag.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):
    _summary = (
        df.group_by("order_type")
        .agg(
            pl.len().alias("n"),
            pl.col("target").mean().alias("mean_target"),
            pl.col("target").median().alias("median_target"),
        )
        .sort("mean_target")
    )
    _summary
    return


@app.cell(hide_code=True)
def _(df, pl):
    _summary = (
        df.group_by("day_of_week")
        .agg(
            pl.len().alias("n"),
            pl.col("target").mean().alias("mean_target"),
            pl.col("target").median().alias("median_target"),
        )
        .sort("day_of_week")
    )
    _summary
    return


@app.cell(hide_code=True)
def _(df, pl):
    _summary = (
        df.group_by("is_weekend", "is_rush_hour", "is_meal_period")
        .agg(
            pl.len().alias("n"),
            pl.col("target").mean().alias("mean_target"),
            pl.col("target").median().alias("median_target"),
        )
        .sort("is_rush_hour", "is_meal_period")
    )
    _summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Order type is essentially flat, while day-of-week and weekend are weak. Rush-hour and meal-period flags are useful summaries of the stronger hour-of-day pattern, but they should not be expected to carry much independent signal beyond the hour itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Time-of-day pattern

    Morning deliveries have the shortest times; evening deliveries have the longest. The pattern is strong enough to retain hour-derived features.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):
    _summary = (
        df.filter(pl.col("order_hour").is_not_null())
        .group_by("order_hour")
        .agg(
            pl.len().alias("n"),
            pl.col("target").mean().alias("mean_target"),
            pl.col("target").median().alias("median_target"),
        )
        .sort("order_hour")
    )
    _summary
    return


@app.cell(hide_code=True)
def _(THEME, df, p9, pl, save_plot):
    _day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    _heat = (
        df.filter(pl.col("order_hour").is_not_null())
        .group_by(["day_of_week", "order_hour"])
        .agg(
            pl.col("target").mean().alias("mean_target"),
            pl.len().alias("n"),
        )
        .with_columns(
            pl.col("day_of_week")
            .replace_strict(list(range(7)), _day_names)
            .alias("day_name")
        )
    )
    _observed_hours = sorted(_heat["order_hour"].unique().to_list())
    _heat = _heat.with_columns(
        pl.col("order_hour").cast(pl.String).alias("order_hour_label"),
        pl.col("day_name").cast(pl.Enum(_day_names)),
    )
    _heat = _heat.with_columns(
        pl.col("order_hour_label").cast(
            pl.Enum([str(hour) for hour in _observed_hours])
        )
    )
    _plot = (
        p9.ggplot(
            _heat,
            p9.aes(x="order_hour_label", y="day_name", fill="mean_target"),
        )
        + p9.geom_tile(color="white", size=0.4)
        + p9.scale_fill_gradient(low="#F7FBFF", high="#08306B", name="Mean\ntarget")
        + THEME
        + p9.theme(figure_size=(8, 5), legend_position="right")
        + p9.labs(
            title="Mean delivery time by order hour and day of week",
            subtitle="No orders observed between 01:00 and 07:00",
            x="Order hour",
            y="",
        )
    )
    _path = save_plot(_plot, "06_target_hour_day_heatmap.png", width=8, height=5)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Morning deliveries around 08:00–10:00 are the fastest, while the 19:00–21:00 block is the slowest. The pattern is strong and non-monotonic, so hour-of-day belongs in the feature set. Raw `order_hour` is sufficient for XGBoost; cyclical sin/cos encodings are better reserved for the linear baseline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Distance and coordinate validity

    Distance has a moderate positive relationship with delivery time. The invalid-coordinate rows have a similar target distribution to valid rows, so they can stay in the dataset with a missing-distance indicator.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):
    valid_distances = df.filter(pl.col("distance_km").is_not_null())
    _summary = valid_distances.select(
        pl.len().alias("n"),
        pl.col("distance_km").min().alias("min_km"),
        pl.col("distance_km").median().alias("median_km"),
        pl.col("distance_km").mean().alias("mean_km"),
        pl.col("distance_km").max().alias("max_km"),
    )
    _summary
    return (valid_distances,)


@app.cell(hide_code=True)
def _(df, pl):
    _summary = (
        df.group_by("coord_status")
        .agg(
            pl.len().alias("n"),
            pl.col("target").mean().alias("mean_target"),
            pl.col("target").median().alias("median_target"),
            pl.col("distance_km").mean().alias("mean_distance_km"),
        )
        .sort("coord_status")
    )
    _summary
    return


@app.cell(hide_code=True)
def _(pl, valid_distances):
    distance_deciles = (
        valid_distances.with_columns(
            (pl.col("distance_km").rank("ordinal") * 10 / (pl.len() + 1))
            .floor()
            .clip(0, 9)
            .cast(pl.Int32)
            .alias("decile")
        )
        .group_by("decile")
        .agg(
            pl.len().alias("n"),
            pl.col("distance_km").mean().alias("mean_km"),
            pl.col("target").mean().alias("mean_target"),
            pl.col("target").median().alias("median_target"),
        )
        .sort("decile")
    )
    distance_deciles
    return (distance_deciles,)


@app.cell(hide_code=True)
def _(THEME, p9, pl, save_plot, valid_distances):
    _correlation = valid_distances.select(
        pl.corr(pl.col("distance_km").rank(), pl.col("target").rank()).alias(
            "spearman"
        )
    ).item()
    _sample = valid_distances.sample(fraction=0.35, seed=42)
    _sample = _sample.with_columns(pl.col("traffic").fill_null("missing"))
    _plot = (
        p9.ggplot(_sample, p9.aes(x="distance_km", y="target", color="traffic"))
        + p9.geom_point(alpha=0.2, size=0.5)
        + p9.geom_smooth(
            p9.aes(group=1), method="lm", se=False, color="black", linetype="dashed"
        )
        + THEME
        + p9.theme(figure_size=(7, 5))
        + p9.labs(
            title="Distance vs delivery time (valid coordinates)",
            subtitle=f"Spearman correlation={_correlation:.2f}; 35% sample shown",
            x="Haversine distance (km)",
            y="Time taken (min)",
        )
    )
    _path = save_plot(_plot, "07_distance_vs_target.png", width=7, height=5)
    _plot
    return


@app.cell(hide_code=True)
def _(THEME, distance_deciles, p9, pl, save_plot):
    _plot_df = distance_deciles.with_columns(
        (pl.col("decile") + 1).alias("decile_label")
    )
    _plot = (
        p9.ggplot(_plot_df, p9.aes(x="decile_label", y="mean_target"))
        + p9.geom_col(fill="#4C78A8")
        + p9.geom_text(
            p9.aes(label="mean_target"), format_string="{:.2f}", va="bottom", size=8
        )
        + THEME
        + p9.theme(figure_size=(7, 5))
        + p9.labs(
            title="Mean delivery time by distance decile",
            subtitle="The distance effect rises through the middle deciles and then plateaus",
            x="Distance decile (1=closest)",
            y="Mean time taken (min)",
        )
    )
    _path = save_plot(_plot, "14_distance_deciles.png", width=7, height=5)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Distance has a moderate positive relationship with the target, but the effect is not linear: it rises through the middle deciles and then plateaus. Raw distance is fine for XGBoost; a binned or spline version may help a linear baseline.
    """)
    return


@app.cell(hide_code=True)
def _(THEME, df, p9, save_plot):
    _plot = (
        p9.ggplot(df, p9.aes(x="coord_status", y="target"))
        + p9.geom_violin(alpha=0.5)
        + p9.geom_boxplot(width=0.1, outlier_size=0.4, outlier_alpha=0.25)
        + THEME
        + p9.theme(figure_size=(6, 4), legend_position="none")
        + p9.labs(
            title="Delivery time by coordinate validity status",
            subtitle="Invalid and sign-flipped rows resemble valid rows",
            x="Coordinate status",
            y="Time taken (min)",
        )
    )
    _path = save_plot(_plot, "08_coordinate_status_target.png", width=6, height=4)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The invalid-coordinate rows have a target distribution similar to valid rows, so dropping them would remove about 8% of the data without an obvious target-bias benefit. A better approach is to keep them, set distance to null, and include a coordinate-validity flag. The 431 sign-flipped restaurant coordinates should be tested with an `abs()` repair before distance is computed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. Missingness and target

    Festival, city, and multiple-delivery missingness are strongly associated with the target. Age, rating, order time, weather, and traffic missingness are close to missing-at-random with respect to the target.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):
    _flags = [
        ("age_missing", "Age missing"),
        ("rating_missing", "Rating missing"),
        ("time_order_missing", "Order time missing"),
        ("weather_missing", "Weather missing"),
        ("traffic_missing", "Traffic missing"),
        ("multiple_missing", "Multiple deliveries missing"),
        ("festival_missing", "Festival missing"),
        ("city_missing", "City missing"),
    ]
    _frames = []
    for _column, _label in _flags:
        _agg = df.group_by(_column).agg(
            pl.col("target").mean().alias("mean_target"),
            pl.col("target").median().alias("median_target"),
            pl.len().alias("n"),
        )
        _agg = _agg.with_columns(
            pl.lit(_label).alias("feature"),
            pl.col(_column).cast(pl.String).alias("missing"),
        )
        _frames.append(
            _agg.select(["feature", "missing", "mean_target", "median_target", "n"])
        )
    missingness_long = pl.concat(_frames)
    missingness_long
    return (missingness_long,)


@app.cell(hide_code=True)
def _(THEME, missingness_long, p9, save_plot):
    _plot = (
        p9.ggplot(
            missingness_long, p9.aes(x="feature", y="mean_target", fill="missing")
        )
        + p9.geom_col(position=p9.position_dodge(width=0.8), width=0.75)
        + THEME
        + p9.theme(
            figure_size=(8, 5),
            axis_text_x=p9.element_text(rotation=35, hjust=1),
        )
        + p9.labs(
            title="Mean target by missingness status",
            subtitle="Festival, city, and multiple-delivery missingness are strongly associated with target",
            x="",
            y="Mean time taken (min)",
        )
    )
    _path = save_plot(_plot, "10_missingness_target.png", width=8, height=5)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Missingness is not uniform in meaning. Age, rating, order time, weather, and traffic missingness look close to missing-at-random with respect to the target. City and multiple-delivery missingness are associated with lower delivery times, and festival missingness is extreme: every such row has a target between 10 and 14 minutes. That last pattern is useful but suspicious and could be an artifact of how the dataset was assembled.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 9. Correlation and courier attributes

    Distance, age, rating, multiple deliveries, and vehicle condition are the strongest numeric signals. Day-of-week is essentially uncorrelated with the target. Age is step-like, and rating is non-monotonic at the extremes.
    """)
    return


@app.cell(hide_code=True)
def _(THEME, df, p9, pl, save_plot, spearman_matrix):
    _numeric_columns = [
        "target",
        "distance_km",
        "age",
        "rating",
        "vehicle_condition",
        "multiple",
        "order_hour",
        "day_of_week",
    ]
    _corr_long = spearman_matrix(df, _numeric_columns).with_columns(
        pl.col("var1").cast(pl.Enum(_numeric_columns)),
        pl.col("var2").cast(pl.Enum(list(reversed(_numeric_columns)))),
    )
    _plot = (
        p9.ggplot(_corr_long, p9.aes(x="var1", y="var2", fill="corr"))
        + p9.geom_tile(color="white")
        + p9.geom_text(p9.aes(label="corr"), format_string="{:.2f}", size=7)
        + p9.scale_fill_gradient2(
            low="#B2182B",
            mid="white",
            high="#2166AC",
            midpoint=0,
            limits=(-1, 1),
            name="Spearman",
        )
        + THEME
        + p9.theme(
            figure_size=(7, 6),
            axis_text_x=p9.element_text(rotation=45, hjust=1),
            legend_position="right",
        )
        + p9.labs(
            title="Spearman correlation matrix",
            subtitle="Pairwise complete observations",
            x="",
            y="",
        )
    )
    _path = save_plot(_plot, "11_correlation_heatmap.png", width=7, height=6)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Distance, age, rating, multiple deliveries, and vehicle condition are the strongest numeric signals. The correlation matrix is a screening tool, not evidence of causation, but it is consistent with the group differences seen in the plots.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):
    age_agg = (
        df.filter(pl.col("age").is_not_null())
        .with_columns(((pl.col("age") // 5) * 5).clip(15, 50).alias("value"))
        .group_by("value")
        .agg(
            pl.len().alias("n"),
            pl.col("target").mean().alias("mean_target"),
        )
        .with_columns(pl.lit("Age bin (years)").alias("feature"))
    )
    age_agg
    return (age_agg,)


@app.cell(hide_code=True)
def _(df, pl):
    rating_agg = (
        df.filter(pl.col("rating").is_not_null())
        .group_by("rating")
        .agg(
            pl.len().alias("n"),
            pl.col("target").mean().alias("mean_target"),
        )
        .with_columns(
            pl.col("rating").alias("value"),
            pl.lit("Rating").alias("feature"),
        )
        .filter(pl.col("n") >= 10)
    )
    rating_agg
    return (rating_agg,)


@app.cell(hide_code=True)
def _(THEME, age_agg, p9, pl, rating_agg, save_plot):
    _combined = pl.concat(
        [
            age_agg.select(["feature", "value", "n", "mean_target"]),
            rating_agg.select(["feature", "value", "n", "mean_target"]),
        ]
    )
    _plot = (
        p9.ggplot(_combined, p9.aes(x="value", y="mean_target"))
        + p9.geom_line(color="#D62728")
        + p9.geom_point(size=2.5, color="#D62728")
        + p9.facet_wrap("feature", scales="free_x")
        + THEME
        + p9.theme(figure_size=(8, 4))
        + p9.labs(
            title="Mean delivery time by courier age and rating",
            subtitle="Age is step-like; rating is non-monotonic at the extremes",
            x="",
            y="Mean time taken (min)",
        )
    )
    _path = save_plot(_plot, "12_age_rating_vs_target.png", width=8, height=4)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Age is step-like rather than linear, and rating is non-monotonic at the extremes. A tree model can capture the raw values; a linear model should bin them. The rare out-of-range rating values are worth flagging separately rather than silently treating them as ordinary ratings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 10. Courier frequency

    Courier delivery count has essentially no target relationship. Keep it out of the core feature set unless testing courier-level target encoding.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):
    courier_counts = (
        df.group_by("courier_id")
        .len()
        .rename({"len": "deliveries"})
        .sort("deliveries")
    )
    _summary = courier_counts.select(
        pl.len().alias("couriers"),
        pl.col("deliveries").min().alias("min"),
        pl.col("deliveries").median().alias("median"),
        pl.col("deliveries").mean().alias("mean"),
        pl.col("deliveries").max().alias("max"),
    )
    _summary
    return (courier_counts,)


@app.cell(hide_code=True)
def _(THEME, courier_counts, df, p9, pl, save_plot):
    _correlation = df.select(
        pl.corr(pl.col("courier_delivery_count").rank(), pl.col("target").rank()).alias(
            "spearman"
        )
    ).item()
    _plot = (
        p9.ggplot(courier_counts, p9.aes(x="deliveries"))
        + p9.geom_histogram(bins=30, fill="#4C78A8", color="white")
        + p9.geom_vline(
            xintercept=courier_counts["deliveries"].median(),
            color="#D62728",
            linetype="dashed",
        )
        + THEME
        + p9.theme(figure_size=(7, 5))
        + p9.labs(
            title="Deliveries per courier",
            subtitle=f"Median shown as dashed line; rank correlation with target={_correlation:.2f}",
            x="Deliveries in dataset",
            y="Courier count",
        )
    )
    _path = save_plot(_plot, "13_courier_frequency.png", width=7, height=5)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Couriers appear between 5 and 67 times, but delivery frequency has essentially no relationship with the target. A raw courier count is therefore not worth carrying. Courier-level historical target encoding remains an experimental option only, and it would require out-of-fold validation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 11. Leakage check: pickup delay

    Pickup delay is 5–15 minutes and has a near-zero rank correlation with the target. This suggests `Time_taken (min)` may measure pickup-to-delivery travel time rather than order-to-delivery time. Confirm the target definition before modeling. Regardless of the definition, `Time_Order_picked` and pickup delay are not available at order placement and must not be model features.
    """)
    return


@app.cell(hide_code=True)
def _(df, pl):
    _summary = df.select(
        pl.col("prep_min").min().alias("min"),
        pl.col("prep_min").median().alias("median"),
        pl.col("prep_min").mean().alias("mean"),
        pl.col("prep_min").max().alias("max"),
    )
    _summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Pickup delay is only 5–15 minutes and has almost no rank correlation with the target. That suggests `Time_taken (min)` may measure pickup-to-delivery travel time rather than order-to-delivery time. The exact start point should be confirmed before modeling. Regardless of the answer, `Time_Order_picked` and pickup delay are unavailable at order placement and must stay out of the feature set.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Findings and recommendations

    ### Problems and proposed fixes

    - Literal `"NaN"` strings: parse as nulls and add informative missingness indicators.
    - Mixed time formats: normalize `HH:MM`, fractional-day values, `1` (24:00), and `24:xx:00` to minutes since midnight.
    - `Metropolitian` spelling: normalize to `Metropolitan`.
    - Ratings above 5.00: clip to 5.00 or retain with an out-of-range flag.
    - Courier ages below 18: flag rather than silently drop or clip.
    - Invalid coordinates: keep the 3,640 rows, add a validity flag, and leave distance missing.
    - Sign-flipped restaurant coordinates: test an `abs()` repair for the 431 affected rows.
    - Suspicious festival missingness: evaluate with and without a `festival_missing` indicator.
    - Target definition: confirm whether the target starts at order placement or pickup.
    - Duplicate rows: none found.

    ### Potential features

    - Distance: `distance_km`, `coordinates_invalid`, coordinate-status flag.
    - Categorical context: `Road_traffic_density`, `Weather_conditions`, `City`, `Festival`, `Type_of_vehicle`.
    - Ordinal: `Vehicle_condition`.
    - Count: `multiple_deliveries` and `has_multiple_deliveries`.
    - Courier: `Delivery_person_Age`, `Delivery_person_Ratings`.
    - Time: `order_hour`, `is_rush_hour`, `is_meal_period`, `time_order_missing`.
    - Missingness: `city_missing`, `multiple_missing`, and optionally `festival_missing`.
    - Weak or excluded: `Type_of_order`, `day_of_week`, `is_weekend`, `courier_delivery_count`, `Time_Order_picked`, `prep_min`.

    ### Potential feature engineering

    - Haversine distance for valid or repaired coordinates.
    - `abs()` repair for sign-flipped restaurant coordinates, with a flag.
    - Distance bins or deciles for the linear baseline; raw distance for XGBoost.
    - Cyclical hour encoding for the linear baseline; raw hour for XGBoost.
    - Ordinal encoding for traffic and vehicle condition.
    - One-hot encoding for weather, city, and vehicle type.
    - Multiple-delivery count plus a binary flag.
    - Age bins and rating bins for the linear baseline; raw values for XGBoost.
    - Rating clipping and out-of-range flags.
    - Missingness indicators for city and multiple deliveries; test festival missingness separately.
    - Courier-level target encoding only with out-of-fold validation.

    The feature engineering decisions are documented in the feature engineering notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
