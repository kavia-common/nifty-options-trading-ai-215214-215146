# nifty-options-trading-ai-215214-215146

Backend: NIFTY50 Options Trading AI (FastAPI)

This repository contains a CPU-friendly FastAPI backend that performs:
- Data ingestion via yfinance
- Feature engineering with common technical indicators
- Lightweight supervised model training and inference
- A minimal RL training/simulation stub (EMA crossover policy)
- Paper trading simulation stub
- Basic observability metrics

## Quick start

1) Create and configure environment
- Python 3.10+ recommended
- Copy environment example
  cp nifty50_trading_backend/.env.example nifty50_trading_backend/.env

2) Install dependencies
  cd nifty50_trading_backend
  pip install -r requirements.txt

3) Run the server
  uvicorn src.api.main:app --reload --port 8000

4) Generate OpenAPI schema (writes to nifty50_trading_backend/interfaces/openapi.json)
  python -m src.api.generate_openapi

Open the docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment variables

See .env.example for all keys:
- ENV, LOG_LEVEL
- DATA_DIR, FEATURES_DIR, MODEL_DIR, RL_DIR
- CORS_ALLOW_ORIGINS
- BROKER_API_KEY, BROKER_API_SECRET

On startup, storage directories will be created automatically.

## Core API endpoints

- GET /           health check
- GET /info       diagnostics and storage paths
- POST /data/ingest         Ingest OHLCV via yfinance
- POST /data/features       Compute features and target
- POST /model/train         Train lightweight model
- POST /model/predict       Run inference and save predictions
- POST /rl/train            Train RL policy (stub)
- POST /rl/simulate         Simulate policy over episodes
- POST /broker/paper        Run a paper trading simulation
- GET /broker/balance       Get current paper broker balance
- GET /broker/positions     Get current paper broker open positions
- POST /broker/order        Place a paper broker order (buy/sell)
- GET /metrics              JSON metrics
- GET /metrics/prom         Prometheus text metrics

## Tests

Run pytest from the backend folder:
  cd nifty50_trading_backend
  pytest -q

Tests use synthetic data; they do not depend on network access.

## Notes

- RL and supervised model implementations are intentionally lightweight to be CI- and CPU-friendly.
- For production-grade RL (e.g., Stable-Baselines3 PPO/A2C) and deep models (e.g., PyTorch Transformers), extend the current service layer and update requirements accordingly.