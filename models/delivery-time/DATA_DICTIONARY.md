# Zomato Delivery Operations Analytics Dataset — Data Dictionary

This dictionary describes the actual CSV downloaded by
`uv run fetch-delivery-data`. Field descriptions are adapted from the Kaggle
dataset page; observed values, missing counts, and data-quality notes come from
the downloaded snapshot.

## Dataset at a glance

| Item | Value |
|---|---|
| Source | <https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset> |
| Creator | Saurabh Badole |
| Local CSV | `data/raw/zomato/Zomato Dataset.csv` |
| Fetch command | `uv run fetch-delivery-data` |
| Snapshot inspected | 2026-11-05 |
| Rows | 45,584 |
| Columns | 20 |
| Target | `Time_taken (min)` |
| Missing-value encoding | Literal string `"NaN"` in the raw CSV |
| Total missing cells | 9,131 |
| Duplicate rows | 0 |
| Unique `ID` values | 45,584 |
| Unique `Delivery_person_ID` values | 1,320 |
| Date span | 2022-02-11 to 2022-04-06 |
| Archive SHA-256 | `232d8edf7f53f64e9ad98b6ef00521fbcfe415916330bfd9f3a35c5c39823258` |

The archive checksum is recorded in `data/raw/zomato/fetch-manifest.json` after
each fetch.

All columns are stored as text in the raw CSV. The logical types below are
inferred after cleaning and parsing.

## Field dictionary

| # | Column | Role | Logical type | Description | Missing | Observed values / range |
|---:|---|---|---|---|---|---|
| 1 | `ID` | Identifier | string | Unique identifier for each delivery. | 0 | 45,584 unique values; hex-like, e.g. `0x2570`. |
| 2 | `Delivery_person_ID` | Identifier | string | Unique identifier for each delivery person. | 0 | 1,320 unique values; pattern like `DEHRES17DEL01`. |
| 3 | `Delivery_person_Age` | Feature | integer | Age of the delivery person in years. | 1,854 (4.07%) | 15–50; median 30. |
| 4 | `Delivery_person_Ratings` | Feature | float | Rating assigned to the delivery person. | 1,908 (4.19%) | 1.0–6.0; median 4.7; 53 values above 5.0. |
| 5 | `Restaurant_latitude` | Geospatial feature | float | Latitude of the restaurant. | 0 | -30.905562–30.914057; 3,640 exact zeros. |
| 6 | `Restaurant_longitude` | Geospatial feature | float | Longitude of the restaurant. | 0 | -88.366217–88.433452; 3,640 exact zeros. |
| 7 | `Delivery_location_latitude` | Geospatial feature | float | Latitude of the delivery location. | 0 | 0.01–31.054057; 3,640 rows with values below 1.0. |
| 8 | `Delivery_location_longitude` | Geospatial feature | float | Longitude of the delivery location. | 0 | 0.01–88.563452; 3,640 rows with values below 1.0. |
| 9 | `Order_Date` | Date feature | date | Date the order was placed, formatted `DD-MM-YYYY`. | 0 | 2022-02-11–2022-04-06; 44 unique dates. |
| 10 | `Time_Orderd` | Time feature | mixed time | Time the order was placed. Actual column name has a typo: `Orderd`. | 1,731 (3.80%) | Mixed `HH:MM`, fractional-day values, and `1` meaning 24:00. |
| 11 | `Time_Order_picked` | Post-order / leakage risk | mixed time | Time the order was picked up for delivery. | 0 | Mixed `HH:MM`, `HH:MM:SS`, fractional-day values, and `24:xx:00` values. |
| 12 | `Weather_conditions` | Categorical feature | string | Weather conditions at delivery time. | 616 (1.35%) | Fog, Stormy, Cloudy, Sandstorms, Windy, Sunny. |
| 13 | `Road_traffic_density` | Ordinal feature | string | Road traffic density during delivery. | 601 (1.32%) | Low, Medium, High, Jam. |
| 14 | `Vehicle_condition` | Ordinal feature | integer | Condition score of the delivery vehicle. | 0 | 0–3; condition 3 is rare (520 rows). |
| 15 | `Type_of_order` | Categorical feature | string | Type of order. | 0 | Snack, Meal, Drinks, Buffet. |
| 16 | `Type_of_vehicle` | Categorical feature | string | Vehicle used for delivery. | 0 | motorcycle, scooter, electric_scooter, bicycle. |
| 17 | `multiple_deliveries` | Count feature | integer | Number of deliveries made in the same trip. | 993 (2.18%) | 0–3; mostly 1. |
| 18 | `Festival` | Binary feature | string | Whether the delivery coincided with a festival. | 228 (0.50%) | No (44,460), Yes (896). |
| 19 | `City` | Categorical feature | string | City or area type where the delivery took place. | 1,200 (2.63%) | Metropolitian, Urban, Semi-Urban. |
| 20 | `Time_taken (min)` | Target | integer | Time taken for delivery in minutes. | 0 | 10–54; mean 26.29, median 26, standard deviation 9.38. |

## Categorical value distributions

Counts and percentages are based on all 45,584 rows.

### `Weather_conditions`

| Value | Count | Share |
|---|---:|---:|
| Fog | 7,653 | 16.79% |
| Stormy | 7,584 | 16.64% |
| Cloudy | 7,533 | 16.53% |
| Sandstorms | 7,494 | 16.44% |
| Windy | 7,422 | 16.28% |
| Sunny | 7,282 | 15.97% |
| `NaN` | 616 | 1.35% |

### `Road_traffic_density`

| Value | Count | Share |
|---|---:|---:|
| Low | 15,476 | 33.95% |
| Jam | 14,139 | 31.02% |
| Medium | 10,945 | 24.01% |
| High | 4,423 | 9.70% |
| `NaN` | 601 | 1.32% |

### `Vehicle_condition`

| Value | Count | Share |
|---|---:|---:|
| 0 | 15,005 | 32.92% |
| 1 | 15,028 | 32.97% |
| 2 | 15,031 | 32.97% |
| 3 | 520 | 1.14% |

### `Type_of_order`

| Value | Count | Share |
|---|---:|---:|
| Snack | 11,530 | 25.29% |
| Meal | 11,456 | 25.13% |
| Drinks | 11,321 | 24.84% |
| Buffet | 11,277 | 24.74% |

### `Type_of_vehicle`

| Value | Count | Share |
|---|---:|---:|
| motorcycle | 26,429 | 57.98% |
| scooter | 15,273 | 33.51% |
| electric_scooter | 3,814 | 8.37% |
| bicycle | 68 | 0.15% |

### `multiple_deliveries`

| Value | Count | Share |
|---|---:|---:|
| 0 | 14,094 | 30.92% |
| 1 | 28,151 | 61.76% |
| 2 | 1,985 | 4.35% |
| 3 | 361 | 0.79% |
| `NaN` | 993 | 2.18% |

### `Festival`

| Value | Count | Share |
|---|---:|---:|
| No | 44,460 | 97.53% |
| Yes | 896 | 1.97% |
| `NaN` | 228 | 0.50% |

### `City`

| Value | Count | Share |
|---|---:|---:|
| Metropolitian | 34,087 | 74.78% |
| Urban | 10,133 | 22.23% |
| Semi-Urban | 164 | 0.36% |
| `NaN` | 1,200 | 2.63% |

Note the dataset spelling `Metropolitian`; it should normally be
`Metropolitan`.

## Data quality notes

### Missing values

- Missing values are stored as the literal string `"NaN"`, not as empty cells.
- Polars and other readers may not treat all of these as null automatically.
- The highest missing rates are in `Delivery_person_Ratings` (4.19%),
  `Delivery_person_Age` (4.07%), and `Time_Orderd` (3.80%).
- The target `Time_taken (min)` has no missing values.

### Time fields

The time columns mix several representations:

- `HH:MM` strings, e.g. `21:55`.
- Fractional-day values, e.g. `0.833333333` for 20:00.
- `1` representing 24:00.
- `Time_Order_picked` also contains `HH:MM:SS` values with hours at 24 or above,
  e.g. `24:05:00`, representing after-midnight times.

All time values must be normalized to a common representation before deriving
order-hour, day-part, or pickup-delay features.

After normalization, pickup minus order time is between 5 and 15 minutes for
every row where both values exist. This makes pickup delay highly predictive,
but it is not available at order placement and therefore must be excluded from
an order-time ETA model.

### Date field

`Order_Date` uses day-first `DD-MM-YYYY` formatting. The observed date span is
2022-02-11 through 2022-04-06.

### Coordinates

- 3,640 rows (7.98%) have invalid paired coordinates: restaurant coordinates
  are exactly `(0, 0)`, while delivery coordinates have equal latitude and
  longitude values below `1.0`.
- 431 rows have a negative restaurant latitude while the delivery latitude is
  positive; 162 of those also have a negative restaurant longitude while the
  delivery longitude is positive. These restaurant coordinates appear
  sign-flipped relative to the delivery coordinates.
- 3,640 restaurant rows have `Restaurant_latitude == 0` and
  `Restaurant_longitude == 0`.
- 3,640 delivery rows have `Delivery_location_latitude == longitude` and both
  values below `1.0`.
- Coordinates should be validated and invalid pairs flagged or excluded before
  computing distance.

### Courier fields

- `Delivery_person_Age` ranges from 15 to 50; 38 rows are below 18.
- `Delivery_person_Ratings` ranges from 1.0 to 6.0. 53 rows are above 5.0,
  which is outside the expected rating scale.
- `Delivery_person_ID` has 1,320 unique values across 45,584 deliveries. It can
  support courier-level aggregates, but raw one-hot encoding is high-cardinality
  and should be avoided.

### Target

`Time_taken (min)` is an integer from 10 to 54 minutes. It has visible spikes at
round values such as 15, 25, and 35, which is worth examining in EDA. The exact
start point of the target—order placement or pickup—should be confirmed before
final feature selection.
