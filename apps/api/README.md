# Inference API

## Overview

Local-only FastAPI inference service for the delivery-time and Yelp sentiment
models. The service exposes health checks, model readiness, and prediction
endpoints for both projects.

The service has no database, authentication, or external service dependencies.

## Model results

- Delivery time: held-out test RMSE **4.01**, MAE **3.16**, R² **0.82**.
- Yelp sentiment: BERT test accuracy **0.68**, macro F1 **0.67**; the TF-IDF
  logistic baseline scores **0.58** accuracy and **0.58** macro F1.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service and model readiness |
| `GET` | `/models` | Loaded model names and readiness |
| `POST` | `/predict/delivery-time` | Delivery-time regression |
| `POST` | `/predict/sentiment` | Five-class Yelp review sentiment classification |

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

## Sentiment request

The sentiment endpoint accepts one review and returns the five-class
distribution. Text is truncated to the model maximum length of 256 tokens.

```bash
curl -X POST http://127.0.0.1:8000/predict/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "The food was great and the delivery was fast."}'
```

### Example response

```json
{
  "label": 3,
  "label_name": "4 stars",
  "confidence": 0.52,
  "probabilities": {
    "1 star": 0.02,
    "2 star": 0.02,
    "3 stars": 0.06,
    "4 stars": 0.52,
    "5 stars": 0.38
  },
  "model": "bert-base-uncased",
  "model_version": "1"
}
```

## Health and model readiness

`GET /health` returns `200` with `status: "ok"` when both models are loaded, or
`status: "degraded"` when either artifact is unavailable. Each model's readiness
is reported separately.

`GET /models` returns the same per-model readiness details as a list.

## Error behavior

- Invalid request fields return FastAPI's normal `422`.
- A missing delivery-time artifact returns `503` from the delivery endpoint.
- A missing sentiment artifact returns `503` from the sentiment endpoint.
- A missing artifact does not prevent application startup.

## Configuration

- `DELIVERY_MODEL_PATH`: path to the classical ML artifact. Defaults to
  `models/delivery-time/artifacts/model.joblib`.
- `SENTIMENT_MODEL_PATH`: path to the exported PyTorch bundle. Defaults to
  `models/yelp-sentiment/artifacts/bert/model.pt`.
- `SENTIMENT_TOKENIZER_PATH`: path to the exported tokenizer directory.
  Defaults to `models/yelp-sentiment/artifacts/bert/tokenizer`.
- `LOG_LEVEL`: application log level, default `INFO`.

For local runs, copy `.env.example` to `.env` and load it explicitly:

```bash
uv run --env-file .env uvicorn inference_api.main:app --reload
```

## Model loading

- Both models are loaded once during application startup.
- The delivery-time pipeline reads `model_metadata.json` and
  `feature_schema.json` beside its artifact.
- The sentiment bundle reads `model.pt` and the sibling tokenizer directory.
  Build it with `uv run export-sentiment-model` from the Yelp project.
- Missing artifacts do not crash the process; readiness is exposed through
  `/health` and `/models`.

## Monitoring and retraining

- Log request count, latency, status codes, and prediction distributions.
- Monitor input drift against the training distribution.
- Retrain when offline metrics degrade, drift exceeds an agreed threshold, new
  labeled data arrives, or business conditions change.
- Version artifacts by Git commit or date and keep a short model card.

## Limitations

- Local demonstration only.
- No persistence, authentication, rate limiting, Docker, or production
  deployment configuration.
