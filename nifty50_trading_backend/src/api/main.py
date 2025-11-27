from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.services.pipeline_service import PipelineService
from src.services.model_service import ModelService
from src.services.rl_service import RLService
from src.services.broker_service import BrokerService
from src.utils.observability import metrics_router
from src.utils.logging import get_logger

# Initialize settings and logger
settings = get_settings()
logger = get_logger(__name__)

openapi_tags = [
    {"name": "health", "description": "Service health and metadata"},
    {"name": "data", "description": "Data ingestion and feature pipeline operations"},
    {"name": "model", "description": "Training and inference for DL model"},
    {"name": "rl", "description": "Reinforcement Learning environment and training"},
    {"name": "broker", "description": "Paper trading and broker integration controls"},
    {"name": "metrics", "description": "Service metrics and observability"},
]


def create_app() -> FastAPI:
    """
    Factory to create and configure the FastAPI app with routers and middleware.
    """
    app = FastAPI(
        title="NIFTY50 Options Trading Backend",
        description="Backend service for data ingestion, feature engineering, transformer predictions, RL training/simulation, and paper trading.",
        version="0.1.0",
        openapi_tags=openapi_tags,
    )

    # CORS setup
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(metrics_router, prefix="/metrics", tags=["metrics"])

    # Services (singleton-ish for app)
    pipeline_service = PipelineService(settings=settings)
    model_service = ModelService(settings=settings)
    rl_service = RLService(settings=settings)
    broker_service = BrokerService(settings=settings)

    # ----- Schemas -----
    class IngestRequest(BaseModel):
        """Request for data ingestion."""
        ticker: str = Field(default="^NSEI", description="Symbol to fetch (e.g., ^NSEI for NIFTY50)")
        period: str = Field(default="1y", description="Period for yfinance (e.g., 1y, 6mo, 5y)")
        interval: str = Field(default="1d", description="Interval for yfinance (e.g., 1d, 1h, 15m)")

    class IngestResponse(BaseModel):
        """Response for data ingestion."""
        rows: int = Field(..., description="Number of rows ingested")
        path: str = Field(..., description="Storage path")

    class FeatureRequest(BaseModel):
        """Request to compute features."""
        input_path: Optional[str] = Field(default=None, description="Path to data CSV; if omitted, uses most recent")
        indicators: Optional[List[str]] = Field(default=None, description="Specific indicators to compute (default set if empty)")

    class FeatureResponse(BaseModel):
        """Response for feature pipeline."""
        rows: int = Field(..., description="Number of rows in features")
        path: str = Field(..., description="Features CSV path")

    class TrainRequest(BaseModel):
        """Training request for supervised DL model."""
        features_path: Optional[str] = Field(default=None, description="Features CSV path")
        epochs: int = Field(default=5, description="Epochs to train (keep small for demo/CI)")
        lr: float = Field(default=1e-3, description="Learning rate")
        batch_size: int = Field(default=64, description="Batch size")

    class TrainResponse(BaseModel):
        """Training response for supervised DL model."""
        model_path: str = Field(..., description="Saved model path")
        metrics: dict = Field(default_factory=dict, description="Training metrics")

    class PredictRequest(BaseModel):
        """Inference request for supervised DL model."""
        features_path: Optional[str] = Field(default=None, description="Features CSV path")
        horizon: int = Field(default=1, description="Prediction horizon steps")

    class PredictResponse(BaseModel):
        """Inference response for supervised DL model."""
        predictions_path: str = Field(..., description="Saved predictions CSV path")
        count: int = Field(..., description="Number of predictions")

    class RLTrainRequest(BaseModel):
        """Request to train RL policy."""
        timesteps: int = Field(default=1000, description="Number of training timesteps (keep small in CI)")
        algo: str = Field(default="PPO", description="RL algorithm: PPO or A2C")
        features_path: Optional[str] = Field(default=None, description="Features CSV path for env")

    class RLTrainResponse(BaseModel):
        """Response for RL training."""
        policy_path: str = Field(..., description="Saved policy path")
        algo: str = Field(..., description="Algorithm used")

    class RLSimulateRequest(BaseModel):
        """Request to simulate RL policy."""
        policy_path: Optional[str] = Field(default=None, description="Existing policy path; train quick policy if omitted")
        episodes: int = Field(default=2, description="Number of episodes to simulate")
        features_path: Optional[str] = Field(default=None, description="Features CSV path for env")

    class RLSimulateResponse(BaseModel):
        """Response for RL simulation."""
        results_path: str = Field(..., description="Simulation results CSV path")
        episodes: int = Field(..., description="Episodes simulated")

    class PaperTradeRequest(BaseModel):
        """Request to run paper trading."""
        policy_path: Optional[str] = Field(default=None, description="RL policy to use for paper trading")
        capital: float = Field(default=100000.0, description="Starting capital")
        risk_per_trade: float = Field(default=0.01, description="Fraction of capital to risk per trade")

    class PaperTradeResponse(BaseModel):
        """Response for paper trading run."""
        run_id: str = Field(..., description="Identifier for the paper trading run")
        summary_path: str = Field(..., description="Path to summary CSV/JSON")

    # ----- Routes -----
    @app.get("/", tags=["health"], summary="Health Check")
    # PUBLIC_INTERFACE
    def health_check() -> dict:
        """Health check endpoint returning basic service status."""
        return {"message": "Healthy", "version": app.version}

    @app.get("/info", tags=["health"], summary="Service Info")
    # PUBLIC_INTERFACE
    def info() -> dict:
        """Return environment and settings information useful for diagnostics."""
        return {
            "model_dir": settings.MODEL_DIR,
            "data_dir": settings.DATA_DIR,
            "features_dir": settings.FEATURES_DIR,
            "rl_dir": settings.RL_DIR,
            "env": settings.ENV,
        }

    @app.post("/data/ingest", tags=["data"], response_model=IngestResponse, summary="Ingest market data")
    # PUBLIC_INTERFACE
    def ingest(req: IngestRequest) -> IngestResponse:
        """Fetch OHLCV data via yfinance and persist as CSV."""
        try:
            path, rows = pipeline_service.ingest_data(ticker=req.ticker, period=req.period, interval=req.interval)
            return IngestResponse(rows=rows, path=path)
        except Exception as e:
            logger.exception("Ingestion failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/data/features", tags=["data"], response_model=FeatureResponse, summary="Compute features")
    # PUBLIC_INTERFACE
    def features(req: FeatureRequest) -> FeatureResponse:
        """Compute technical indicators and engineered features, save CSV."""
        try:
            path, rows = pipeline_service.compute_features(input_path=req.input_path, indicators=req.indicators or [])
            return FeatureResponse(rows=rows, path=path)
        except Exception as e:
            logger.exception("Feature pipeline failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/model/train", tags=["model"], response_model=TrainResponse, summary="Train DL model")
    # PUBLIC_INTERFACE
    def train_model(req: TrainRequest) -> TrainResponse:
        """Train a simple transformer-like model on features to predict returns."""
        try:
            model_path, metrics = model_service.train(features_path=req.features_path, epochs=req.epochs, lr=req.lr, batch_size=req.batch_size)
            return TrainResponse(model_path=model_path, metrics=metrics)
        except Exception as e:
            logger.exception("Model training failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/model/predict", tags=["model"], response_model=PredictResponse, summary="Run inference")
    # PUBLIC_INTERFACE
    def predict(req: PredictRequest) -> PredictResponse:
        """Run inference using latest or specified model on features."""
        try:
            predictions_path, count = model_service.predict(features_path=req.features_path, horizon=req.horizon)
            return PredictResponse(predictions_path=predictions_path, count=count)
        except Exception as e:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/rl/train", tags=["rl"], response_model=RLTrainResponse, summary="Train RL policy")
    # PUBLIC_INTERFACE
    def rl_train(req: RLTrainRequest) -> RLTrainResponse:
        """Train an RL agent (PPO/A2C) in a lightweight environment based on features."""
        try:
            policy_path, algo = rl_service.train(
                timesteps=req.timesteps,
                algo=req.algo,
                features_path=req.features_path,
            )
            return RLTrainResponse(policy_path=policy_path, algo=algo)
        except Exception as e:
            logger.exception("RL training failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/rl/simulate", tags=["rl"], response_model=RLSimulateResponse, summary="Simulate RL policy")
    # PUBLIC_INTERFACE
    def rl_simulate(req: RLSimulateRequest) -> RLSimulateResponse:
        """Simulate a policy over episodes and save the results."""
        try:
            results_path, episodes = rl_service.simulate(
                episodes=req.episodes,
                policy_path=req.policy_path,
                features_path=req.features_path,
            )
            return RLSimulateResponse(results_path=results_path, episodes=episodes)
        except Exception as e:
            logger.exception("RL simulation failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/broker/paper", tags=["broker"], response_model=PaperTradeResponse, summary="Run paper trading")
    # PUBLIC_INTERFACE
    def broker_paper(req: PaperTradeRequest) -> PaperTradeResponse:
        """Run a basic paper trading simulation using an RL policy outputs as signals."""
        try:
            run_id, summary_path = broker_service.paper_trade(
                policy_path=req.policy_path,
                capital=req.capital,
                risk_per_trade=req.risk_per_trade,
            )
            return PaperTradeResponse(run_id=run_id, summary_path=summary_path)
        except Exception as e:
            logger.exception("Paper trading failed")
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = create_app()
