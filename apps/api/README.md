# Inference API

## Overview

Local-only FastAPI inference service for both take-home models. The service is
intended for demonstration purposes and has no database, authentication, or
external service dependencies.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service and model readiness |
| `GET` | `/models` | Loaded model names and status |
| `POST` | `/predict/delivery-time` | Delivery-time regression |
| `POST` | `/predict/sentiment` | Review sentiment classification |

## Run locally

```bash
uv sync --all-packages
uv run uvicorn inference_api.main:app --reload
```

Interactive documentation is available at `http://127.0.0.1:8000/docs`.

## Example requests

These examples are illustrative until the models and API are implemented.

```bash
curl -X POST http://127.0.0.1:8000/predict/delivery-time \
  -H "Content-Type: application/json" \
  -d '{}'
```

```json
{
  "prediction": 0.0
}
```

```bash
curl -X POST http://127.0.0.1:8000/predict/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "The food was great and the delivery was fast."}'
```

```json
{
  "label": "positive",
  "score": 0.0
}
```

## Configuration

- `DELIVERY_MODEL_PATH`: path to the classical ML artifact.
- `SENTIMENT_MODEL_PATH`: path to the NLP model directory.
- `LOG_LEVEL`: application log level.

For local runs, copy `.env.example` to `.env` and load it explicitly:

```bash
uv run --env-file .env uvicorn inference_api.main:app --reload
```

## Model loading and errors

- Models are loaded once during application startup.
- Missing artifacts should not crash the whole process. Readiness should be
  reported through `/health`, and prediction endpoints should return a clear
  `503` when a model is unavailable.
- Request validation errors use FastAPI's normal `422` behavior.

## Monitoring and retraining plan

- Log request count, latency, status codes, and prediction distributions.
- Monitor input drift against the training distribution.
- Retrain when offline metrics degrade, drift exceeds an agreed threshold, new
  labeled data arrives, or business conditions change.
- Version artifacts by Git commit or date and keep a short model card.

This is a plan only; no monitoring is implemented in the scaffold.

## Adding an endpoint

1. Add request and response schemas in `schemas.py`.
2. Add a router in `routers/`.
3. Register the model loader in `registry.py`.
4. Include the router in `main.py`.
5. Add tests under `tests/`.

## Limitations

- Local demonstration only.
- No persistence, authentication, rate limiting, or production deployment
  configuration.
