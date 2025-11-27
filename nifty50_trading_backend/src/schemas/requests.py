from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# PUBLIC_INTERFACE
class IngestRequest(BaseModel):
    """Request for data ingestion."""
    ticker: str = Field(default="^NSEI", description="Symbol to fetch (e.g., ^NSEI for NIFTY50)")
    period: str = Field(default="1y", description="Period for yfinance (e.g., 1y, 6mo, 5y)")
    interval: str = Field(default="1d", description="Interval for yfinance (e.g., 1d, 1h, 15m)")


# PUBLIC_INTERFACE
class FeatureRequest(BaseModel):
    """Request to compute features."""
    input_path: Optional[str] = Field(default=None, description="Path to data CSV; if omitted, uses most recent")
    indicators: Optional[List[str]] = Field(default=None, description="Specific indicators to compute (default set if empty)")


# PUBLIC_INTERFACE
class TrainRequest(BaseModel):
    """Training request for supervised DL model."""
    features_path: Optional[str] = Field(default=None, description="Features CSV path")
    epochs: int = Field(default=5, description="Epochs to train (keep small for demo/CI)")
    lr: float = Field(default=1e-3, description="Learning rate")
    batch_size: int = Field(default=64, description="Batch size")


# PUBLIC_INTERFACE
class PredictRequest(BaseModel):
    """Inference request for supervised DL model."""
    features_path: Optional[str] = Field(default=None, description="Features CSV path")
    horizon: int = Field(default=1, description="Prediction horizon steps")


# PUBLIC_INTERFACE
class RLTrainRequest(BaseModel):
    """Request to train RL policy."""
    timesteps: int = Field(default=1000, description="Number of training timesteps (keep small in CI)")
    algo: str = Field(default="PPO", description="RL algorithm: PPO or A2C")
    features_path: Optional[str] = Field(default=None, description="Features CSV path for env")


# PUBLIC_INTERFACE
class RLSimulateRequest(BaseModel):
    """Request to simulate RL policy."""
    policy_path: Optional[str] = Field(default=None, description="Existing policy path; train quick policy if omitted")
    episodes: int = Field(default=2, description="Number of episodes to simulate")
    features_path: Optional[str] = Field(default=None, description="Features CSV path for env")


# PUBLIC_INTERFACE
class PaperTradeRequest(BaseModel):
    """Request to run paper trading."""
    policy_path: Optional[str] = Field(default=None, description="RL policy to use for paper trading")
    capital: float = Field(default=100000.0, description="Starting capital")
    risk_per_trade: float = Field(default=0.01, description="Fraction of capital to risk per trade")


# PUBLIC_INTERFACE
class PlaceOrderRequest(BaseModel):
    """Request to place a paper broker order."""
    side: str = Field(..., description="Order side: 'buy' or 'sell'")
    quantity: int = Field(..., description="Quantity to trade (must be > 0)")
    price: float = Field(..., description="Execution price (must be > 0)")
