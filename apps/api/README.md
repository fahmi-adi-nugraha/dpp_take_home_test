# Inference API

## Overview

Local-only FastAPI inference service for the take-home models. The delivery-time
endpoint is implemented. The sentiment route is scaffolded and returns `501`
until the Yelp BERT model is integrated.

The service has no database, authentication, or external service dependencies.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service and model readiness |
| `GET` | `/models` | Loaded model names and status |
| `POST` | `/predict/delivery-time` | Delivery-time regression |
| `POST` | `/predict/sentiment` | Reserved sentiment classification endpoint |

## Run locally

```bash
uv sync --all-packages
uv run uvicorn inference_api.main:app --reload
```

Interactive documentation is available at `http://127.0.0.1:8000/docs`.

## Delivery-time request

The endpoint accepts raw order, location, and courier inputs. The API constructs
the 18 engineered features expected by the saved XGBoost pipeline.

| Field | Type | Notes |
|---|---|---|
| `restaurant` | object or null | `latitude` and `longitude`; missing or invalid values produce a warning |
| `delivery` | object or null | `latitude` and `longitude`; missing or invalid values produce a warning |
| `order_time` | `HH:MM` or null | Order placement time; drives hour, rush-hour, and meal-period features |
| `weather` | string or null | `Cloudy`, `Fog`, `Sandstorms`, `Stormy`, `Sunny`, or `Windy` |
| `traffic` | string or null | `Low`, `Medium`, `High`, or `Jam` |
| `city` | string or null | `Metropolitan`, `Semi-Urban`, or `Urban`; `Metropolitian` is normalized |
| `festival` | string or null | `Yes` or `No` |
| `multiple_deliveries` | integer or null | Number of deliveries in the same trip |
| `vehicle_condition` | integer or null | Vehicle condition score, 0–3 |
| `courier_age` | number or null | Courier age in years |
| `courier_rating` | number or null | Courier rating; out-of-range values are passed through to the model |

Pickup time, prep time, the target, vehicle type, and order type are not accepted
because they are either leakage fields or excluded from the selected feature set.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict/delivery-time \
  -H "Content-Type: application/json" \
  -d '{
    "restaurant": {"latitude": 12.9716, "longitude": 77.5946},
    "delivery": {"latitude": 12.9352, "longitude": 77.6245},
    "order_time": "12:30",
    "weather": "Sunny",
    "traffic": "Medium",
    "city": "Urban",
    "festival": "No",
    "multiple_deliveries": 1,
    "vehicle_condition": 1,
    "courier_age": 30,
    "courier_rating": 4.7
  }'
```

### Example response

```json
{
  "prediction_minutes": 18.39,
  "model": "delivery-time-xgboost",
  "model_version": "1",
  "selected_feature_set": "plus_weather_group",
  "prediction_point": "order_placement",
  "warnings": []
}
```

If coordinates or order time are missing, the prediction still returns, and the
`warnings` list explains which inputs used the model's missing-value handling.

## Health and model status

`GET /health` returns `200` with `status: "ok"` when the delivery-time model is
loaded, or `status: "degraded"` when it is unavailable. The sentiment model is
reported as not implemented.

`GET /models` returns the same per-model readiness details as a list.

## Error behavior

- Invalid request fields return FastAPI's normal `422`.
- A missing delivery-time artifact returns `503` from the prediction endpoint.
- The sentiment endpoint returns `501`.
- The application still starts if a model artifact is missing.

## Configuration

- `DELIVERY_MODEL_PATH`: path to the classical ML artifact. Defaults to
  `models/delivery-time/artifacts/model.joblib`.
- `SENTIMENT_MODEL_PATH`: path to the NLP model directory. Reserved for the
  future sentiment integration.
- `LOG_LEVEL`: application log level, default `INFO`.

For local runs, copy `.env.example` to `.env` and load it explicitly:

```bash
uv run --env-file .env uvicorn inference_api.main:app --reload
```

## Model loading

- The delivery-time pipeline is loaded once during application startup.
- `model_metadata.json` and `feature_schema.json` beside the artifact provide
  the feature order and model metadata.
- Missing artifacts do not crash the process; readiness is exposed through
  `/health` and `/models`.

## Monitoring and retraining plan

- Log request count, latency, status codes, and prediction distributions.
- Monitor input drift against the training distribution.
- Retrain when offline metrics degrade, drift exceeds an agreed threshold, new
  labeled data arrives, or business conditions change.
- Version artifacts by Git commit or date and keep a short model card.

This is a plan only; request-level monitoring is not implemented yet.

## Adding the sentiment endpoint

1. Add a sentiment predictor in `models/yelp-sentiment/src/yelp_sentiment/inference.py`.
2. Register it in `ModelRegistry`.
3. Replace the `501` in `routers/sentiment.py` with the real prediction call.
4. Add tests under `apps/api/tests/`.

## Limitations

- Local demonstration only.
- No persistence, authentication, rate limiting, Docker, or production
  deployment configuration.
