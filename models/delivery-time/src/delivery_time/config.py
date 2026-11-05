"""Configuration constants for the delivery-time project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "zomato"
ARCHIVE_FILENAME = "zomato-delivery-operations-analytics-dataset.zip"
EXPECTED_CSV_FILENAME = "Zomato Dataset.csv"
MANIFEST_FILENAME = "fetch-manifest.json"

DATASET_SLUG = "saurabhbadole/zomato-delivery-operations-analytics-dataset"
DATASET_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"

TARGET_COLUMN = "Time_taken (min)"
EXPECTED_COLUMNS = (
    "ID",
    "Delivery_person_ID",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
    "Weather_conditions",
    "Road_traffic_density",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City",
    "Time_taken (min)",
)

DEFAULT_TIMEOUT_SECONDS = 60.0
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "dpp-delivery-time/0.1.0"
