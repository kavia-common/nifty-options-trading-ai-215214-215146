from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


# PUBLIC_INTERFACE
class IngestResponse(BaseModel):
    """Response for data ingestion."""
    rows: int = Field(..., description="Number of rows ingested")
    path: str = Field(..., description="Storage path")


# PUBLIC_INTERFACE
class FeatureResponse(BaseModel):
    """Response for feature pipeline."""
    rows: int = Field(..., description="Number of rows in features")
    path: str = Field(..., description="Features CSV path")


# PUBLIC_INTERFACE
class TrainResponse(BaseModel):
    """Training response for supervised DL model."""
    model_path: str = Field(..., description="Saved model path")
    metrics: Dict = Field(default_factory=dict, description="Training metrics")


# PUBLIC_INTERFACE
class PredictResponse(BaseModel):
    """Inference response for supervised DL model."""
    predictions_path: str = Field(..., description="Saved predictions CSV path")
    count: int = Field(..., description="Number of predictions")


# PUBLIC_INTERFACE
class RLTrainResponse(BaseModel):
    """Response for RL training."""
    policy_path: str = Field(..., description="Saved policy path")
    algo: str = Field(..., description="Algorithm used")


# PUBLIC_INTERFACE
class RLSimulateResponse(BaseModel):
    """Response for RL simulation."""
    results_path: str = Field(..., description="Simulation results CSV path")
    episodes: int = Field(..., description="Episodes simulated")


# PUBLIC_INTERFACE
class PaperTradeResponse(BaseModel):
    """Response for paper trading run."""
    run_id: str = Field(..., description="Identifier for the paper trading run")
    summary_path: str = Field(..., description="Path to summary CSV/JSON")


# PUBLIC_INTERFACE
class BalanceResponse(BaseModel):
    """Response containing current broker balance."""
    balance: float = Field(..., description="Current account balance")


# PUBLIC_INTERFACE
class PositionsResponse(BaseModel):
    """Response containing current number of positions."""
    positions: int = Field(..., description="Number of open positions")


# PUBLIC_INTERFACE
class OrderResponse(BaseModel):
    """Response for a placed order including updated account state."""
    order_id: str = Field(..., description="Order identifier")
    balance: float = Field(..., description="Updated account balance")
    positions: int = Field(..., description="Updated number of positions")
