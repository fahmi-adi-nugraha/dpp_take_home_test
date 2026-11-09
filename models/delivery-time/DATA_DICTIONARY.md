# Zomato Delivery Operations Analytics Dataset — Data Dictionary

This dictionary describes the schema and field meaning for the CSV downloaded
by `uv run fetch-delivery-data`. Field descriptions are adapted from the Kaggle
dataset page.

## Dataset at a glance

| Item | Value |
|---|---|
| Source | <https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset> |
| Creator | Saurabh Badole |
| Local CSV | `data/raw/zomato/Zomato Dataset.csv` |
| Fetch command | `uv run fetch-delivery-data` |
| Format | CSV with a header row |
| Rows | 45,584 |
| Columns | 20 |
| Target | `Time_taken (min)` |
| Archive SHA-256 | `232d8edf7f53f64e9ad98b6ef00521fbcfe415916330bfd9f3a35c5c39823258` |

The archive checksum is recorded in `data/raw/zomato/fetch-manifest.json` after
each fetch. All columns are stored as text in the raw CSV; the logical types
below describe the parsed data.

## Field dictionary

| # | Column | Role | Logical type | Description |
|---:|---|---|---|---|
| 1 | `ID` | Identifier | string | Unique identifier for each delivery. |
| 2 | `Delivery_person_ID` | Identifier | string | Unique identifier for each delivery person. |
| 3 | `Delivery_person_Age` | Feature | integer | Age of the delivery person in years. |
| 4 | `Delivery_person_Ratings` | Feature | float | Rating assigned to the delivery person. |
| 5 | `Restaurant_latitude` | Geospatial feature | float | Latitude of the restaurant. |
| 6 | `Restaurant_longitude` | Geospatial feature | float | Longitude of the restaurant. |
| 7 | `Delivery_location_latitude` | Geospatial feature | float | Latitude of the delivery location. |
| 8 | `Delivery_location_longitude` | Geospatial feature | float | Longitude of the delivery location. |
| 9 | `Order_Date` | Date feature | date | Date the order was placed, formatted `DD-MM-YYYY`. |
| 10 | `Time_Orderd` | Time feature | mixed time | Time the order was placed. The raw column name contains the typo `Orderd`. |
| 11 | `Time_Order_picked` | Post-order / leakage risk | mixed time | Time the order was picked up for delivery. |
| 12 | `Weather_conditions` | Categorical feature | string | Weather conditions at delivery time. |
| 13 | `Road_traffic_density` | Ordinal feature | string | Road traffic density during delivery. |
| 14 | `Vehicle_condition` | Ordinal feature | integer | Condition score of the delivery vehicle. |
| 15 | `Type_of_order` | Categorical feature | string | Type of order: Snack, Meal, Drinks, or Buffet. |
| 16 | `Type_of_vehicle` | Categorical feature | string | Vehicle used for delivery. |
| 17 | `multiple_deliveries` | Count feature | integer | Number of deliveries made in the same trip. |
| 18 | `Festival` | Binary feature | string | Whether the delivery coincided with a festival. |
| 19 | `City` | Categorical feature | string | City or area type. |
| 20 | `Time_taken (min)` | Target | integer | Time taken for delivery in minutes. |
