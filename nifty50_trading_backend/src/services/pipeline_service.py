from __future__ import annotations

from typing import List, Optional

from src.config.settings import Settings
from src.data.ingestion import ingest_ohlcv
from src.features.pipeline import build_features
from src.storage.paths import latest_file
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PipelineService:
    """Service orchestrating data ingestion and feature engineering."""

    def __init__(self, settings: Settings):
        self.settings = settings

    # PUBLIC_INTERFACE
    def ingest_data(self, ticker: str, period: str, interval: str) -> tuple[str, int]:
        """Trigger data ingestion using yfinance."""
        path, rows = ingest_ohlcv(ticker=ticker, period=period, interval=interval, out_dir=self.settings.DATA_DIR)
        return path, rows

    # PUBLIC_INTERFACE
    def compute_features(self, input_path: Optional[str] = None, indicators: Optional[List[str]] = None) -> tuple[str, int]:
        """Compute features from input CSV and persist to features directory."""
        if input_path is None:
            input_path = latest_file(self.settings.DATA_DIR, patterns=["*.csv"])
            if input_path is None:
                raise FileNotFoundError("No input data CSV found. Ingest data or provide input_path.")
        path, rows = build_features(input_csv=input_path, output_dir=self.settings.FEATURES_DIR, indicators=indicators or [])
        return path, rows
