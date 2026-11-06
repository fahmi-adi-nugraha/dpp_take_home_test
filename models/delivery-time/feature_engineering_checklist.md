# Delivery-Time Feature Engineering Checklist

Use this as the working decision sheet for the classical ML project. The
recommendations are informed by
[DATA_DICTIONARY.md](DATA_DICTIONARY.md) and the analysis in
[`notebooks/01_eda.py`](notebooks/01_eda.py).

**How to use:** `[x]` marks the recommended starting point. Change boxes as
needed. `[ ]` marks optional or experimental work.

## 1. Data cleaning and normalization

- [x] Replace the literal string `"NaN"` with proper nulls before analysis.
- [x] Parse `Order_Date` as day-first `DD-MM-YYYY`.
- [x] Normalize `Time_Orderd` and `Time_Order_picked`, which mix `HH:MM`,
      fractional-day values, `1` meaning 24:00, and `24:xx:00` values.
- [x] Normalize the `City` spelling `Metropolitian` to `Metropolitan`.
- [x] Keep the target integer-valued; it has no missing values.
- [x] Confirm there are no duplicate rows; the observed count is zero.
- [x] Flag invalid coordinate pairs instead of silently using them.
- [x] Flag sign-flipped restaurant coordinates.
- [x] Test an `abs()` repair for the 431 sign-flipped restaurant coordinates
      before computing distance.
- [x] Flag ratings above 5.0 and either clip them to 5.0 or retain them with an
      out-of-range indicator.
- [x] Flag courier ages below 18 rather than silently dropping or clipping them.
- [x] Keep rows with invalid coordinates for now; their target distribution is
      similar to valid rows, and XGBoost can handle missing distance.
- [x] Add a `time_order_missing` indicator for the 1,731 missing order times.

## 2. Core feature selection

### Strong signals — recommended

- [x] `distance_km` — Haversine distance for valid or repaired coordinate pairs.
- [x] `coordinates_invalid` — flag for the 3,640 invalid paired-coordinate rows.
- [x] `Road_traffic_density` — ordinal: Low, Medium, High, Jam.
- [x] `Weather_conditions` — categorical; Sunny is notably faster, Fog/Cloudy
      slower.
- [x] `City` — categorical; keep Semi-Urban because it has a very large effect.
- [x] `Festival` — binary Yes/No.
- [x] `multiple_deliveries` — count 0–3.
- [x] `has_multiple_deliveries` — binary version of the count.
- [x] `Vehicle_condition` — ordinal 0–3.
- [x] `Delivery_person_Age` — numeric; age is step-like rather than linear.
- [x] `Delivery_person_Ratings` — numeric; rating is non-monotonic at the
      extremes.
- [x] `order_hour` — hour of order placement; numeric for XGBoost, optionally
      categorical or one-hot.
- [x] `order_hour_sin` and `order_hour_cos` — cyclical encoding for the linear
      baseline or an encoding ablation; not part of the default XGBoost set.
- [x] `is_rush_hour` — 11:00–14:00 or 17:00–21:00.
- [x] `is_meal_period` — broad breakfast/lunch/dinner window.
- [x] `city_missing` — City missingness is associated with a lower target.
- [x] `multiple_missing` — missing multiple-delivery count is associated with a
      lower target.

### Moderate or optional signals

- [ ] `Type_of_vehicle` — keep motorcycle/scooter/electric/bicycle; the effect
      is modest and bicycle is rare.
- [ ] `vehicle_condition_is_zero` — binary split of condition 0 vs 1–3; useful
      for a linear baseline.
- [ ] `weather_group` — collapse into clear, cloudy/fog, and storm/wind groups.
- [ ] `day_of_week` — weak signal; keep only if validation improves.
- [ ] `is_weekend` — no meaningful effect in the initial screen.
- [x] `age_bin` — 5-year bins for a linear model; XGBoost can use raw age.
- [ ] `rating_bin` — bin the non-monotonic rating values for a linear model.
- [ ] `distance_decile` or `distance_bin` — useful for a linear model; XGBoost
      can use raw distance.
- [ ] `restaurant_coords_sign_flipped` — data-quality flag; test whether it
      adds signal after distance repair.
- [ ] `coord_status` — diagnostic category (valid, sign_flipped, invalid);
      use `coordinates_invalid` and optionally `restaurant_coords_sign_flipped`
      in the baseline unless the categorical version improves validation.
- [ ] `festival_missing` — **high-signal but suspicious**: every missing-festival
      row has a 10–14 minute target. Include only if an ablation supports it.
- [ ] `age_missing`, `rating_missing`, `weather_missing`, `traffic_missing` —
      no meaningful target difference in the initial screen.
- [ ] `courier_delivery_count` — no meaningful target relationship; exclude from
      the core set.
- [ ] Raw `Delivery_person_ID` one-hot encoding — high-cardinality, avoid.
- [ ] Courier-level target encoding — experimental only, with out-of-fold
      cross-validation.
- [ ] Raw restaurant/delivery coordinates — only after coordinate cleaning;
      distance and city are safer baseline features.
- [x] Rounded coordinates or geohash — experimental after the baseline.

### Weak or excluded

- [ ] `Type_of_order` — no meaningful target difference; exclude from the core.
- [ ] `ID` — identifier only.
- [ ] `Time_Order_picked` — future information at order placement.
- [ ] `prep_min` — future information at order placement.
- [ ] Any feature derived from `Time_taken (min)` — direct target leakage.

## 3. Feature engineering methods

### Geospatial

- [x] Compute Haversine distance from restaurant to delivery coordinates.
- [x] Use `abs()` repair for sign-flipped restaurant coordinates and compare
      distances before and after repair.
- [x] Set distance to null when coordinate pairs remain invalid.
- [x] Add `coordinates_invalid` as a model feature.
- [x] Derive `coord_status` for EDA; keep binary coordinate flags in the
      baseline.
- [ ] Add `same_location_flag` if repaired coordinates produce zero distance.
- [ ] Add distance bins/deciles for linear or interpretable models.

### Time

- [x] Convert all time values to minutes since midnight.
- [x] Derive `order_hour` for tree models.
- [x] Derive `order_hour_sin` and `order_hour_cos` for the linear baseline or an
      encoding ablation.
- [x] Derive `is_rush_hour` and `is_meal_period`.
- [x] Add `time_order_missing`.
- [ ] Add day-of-week cyclical encodings only if `day_of_week` is retained.
- [ ] Add late-night or specific meal-window flags if EDA suggests they help.

### Categorical and ordinal encoding

- [x] Ordinal-encode `Road_traffic_density`: Low=0, Medium=1, High=2, Jam=3.
- [x] Keep `Vehicle_condition` ordinal as 0, 1, 2, 3.
- [x] One-hot encode `Weather_conditions` and `City`; one-hot
      `Type_of_vehicle` only if that feature is retained.
- [x] Encode `Festival` as a binary indicator.
- [x] Add missing indicators for `City` and `multiple_deliveries`.
- [x] Group or regularize rare levels if validation is unstable.
- [x] Use native categorical handling in XGBoost as an alternative to one-hot.

### Numeric transforms

- [x] Keep `distance_km`, `Delivery_person_Age`, and
      `Delivery_person_Ratings` numeric for XGBoost.
- [x] Add `vehicle_condition_is_zero` for a linear baseline if that split is
      retained.
- [x] Add `age_bin` and `rating_bin` for a linear baseline.
- [ ] Add `distance_decile` for a linear baseline.
- [x] Test clipping ratings above 5.0 vs retaining them with a flag.
- [x] Test whether age should be clipped to a plausible range.

### Missingness

- [x] Add `city_missing`.
- [x] Add `multiple_missing`.
- [ ] Add `festival_missing` only after a with/without ablation.
- [ ] Add indicators for age, rating, weather, and traffic only if they improve
      validation. `time_order_missing` is retained as a data-quality flag.

## 4. Modeling and validation

- [x] Use a scikit-learn `Pipeline` and `ColumnTransformer` so preprocessing is
      reused at inference time.
- [x] Start with XGBoost regression and `reg:squarederror`.
- [x] Evaluate RMSE, MAE, and R².
- [x] Use a train/validation split; keep a held-out test set for the final check.
- [ ] Add a time-based validation split as a robustness check.
- [x] Compare with and without `distance_km`.
- [x] Compare with and without invalid-coordinate rows.
- [x] Compare raw vs repaired coordinates for the 431 sign-flipped rows.
- [x] Compare count vs binary treatment of multiple deliveries.
- [x] Compare hour encodings: numeric hour, numeric hour plus rush/meal flags,
      one-hot or native categorical hour, and numeric hour plus sin/cos.
- [x] Compare with and without the high-signal missingness flags.
- [x] Inspect residuals and the target spikes at 15, 25, and 35 minutes.
- [ ] Test a log target transform.
- [ ] Test a Poisson or Tweedie objective.
- [x] Tune hyperparameters after the baseline is stable.
- [ ] Use permutation importance or SHAP to confirm feature value.

## 5. Leakage controls

- [x] Exclude `Time_Order_picked` from the order-placement feature set.
- [x] Exclude `prep_min` and all pickup-delay features.
- [x] Exclude `ID`.
- [x] Exclude raw `Delivery_person_ID` one-hot encoding.
- [x] Exclude any feature calculated from `Time_taken (min)`.
- [x] Use out-of-fold encoding if courier-level historical target features are
      ever added.
- [ ] Confirm whether `Time_taken (min)` starts at order placement or pickup.
      This is still an open target-definition question.

## 6. Recommended first baseline

A reasonable first feature set is:

```text
distance_km
coordinates_invalid
Road_traffic_density
Weather_conditions
City
Festival
multiple_deliveries
has_multiple_deliveries
Vehicle_condition
Delivery_person_Age
Delivery_person_Ratings
order_hour
is_rush_hour
is_meal_period
time_order_missing
city_missing
multiple_missing
```

Exclude `Type_of_order`, `day_of_week`, `is_weekend`,
`courier_delivery_count`, `Time_Order_picked`, and `prep_min` from the first
baseline. Add them back only if validation shows a real improvement.

## 7. Open questions

- [x] Confirm whether `Time_taken (min)` starts at order placement or pickup.

The invalid-coordinate, sign-flipped-coordinate, and `festival_missing`
decisions are already covered by the cleaning and ablation tasks in sections
1–4, so they do not need separate open questions.

Optional data-understanding checks:

- [x] Confirm the source or meaning of ratings above 5.0; current handling is to
      flag them and test clipping versus retention.
- [x] Confirm the source or meaning of courier ages below 18; current handling
      is to flag them and test clipping.

### Meaning of `Time_taken (min)`

Here's what's stated in the Kaggle page:

```
Time taken for delivery in minutes
```

This is pretty ambiguous language but to me it makes more sense to start measuring the
time taken an the time of order placement so that's what we'll do.

### Source or meaning of ratings above 5.0

Since the vast majority of the ratings fell within the 0-5 range I have to assume that any
rating above 5.0 was the result of some sort of error whether it be system error (i.e.,
the mobile app bugged out) or user error (i.e., maybe the mobile app allows users to enter
a rating on in a text field and somehow ratings above 5.0 got past the validation). I
think we stick with the current plan of flagging them and testing clipping.

### Source or meaning of courier ages below 18

It could be that somehow people who are effectively minors were able to get this job.
While technically illegal in many places, the fact is that they're still allowed to work
at this company so I say we flag them and test clipping.
