from __future__ import annotations

from typing import Optional

from src.config.settings import Settings
from src.models.inference import predict_supervised
from src.models.train import train_supervised
from src.storage.paths import latest_file
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelService:
    """Service orchestrating model training and inference."""

    def __init__(self, settings: Settings):
        self.settings = settings

    # PUBLIC_INTERFACE
    def train(self, features_path: Optional[str] = None, epochs: int = 5, lr: float = 1e-3, batch_size: int = 64) -> tuple[str, dict]:
        """Train the supervised model on features."""
        if features_path is None:
            features_path = latest_file(self.settings.FEATURES_DIR, patterns=["*.csv"])
            if features_path is None:
                raise FileNotFoundError("No features CSV found. Compute features first or provide features_path.")
        model_path, metrics = train_supervised(features_path=features_path, model_dir=self.settings.MODEL_DIR, epochs=epochs, lr=lr, batch_size=batch_size)
        return model_path, metrics

    # PUBLIC_INTERFACE
    def predict(self, features_path: Optional[str] = None, horizon: int = 1) -> tuple[str, int]:
        """Run inference using the latest model and features."""
        if features_path is None:
            features_path = latest_file(self.settings.FEATURES_DIR, patterns=["*.csv"])
            if features_path is None:
                raise FileNotFoundError("No features CSV found. Compute features first or provide features_path.")
        predictions_path, count = predict_supervised(features_path=features_path, model_dir=self.settings.MODEL_DIR, horizon=horizon)
        return predictions_path, count
