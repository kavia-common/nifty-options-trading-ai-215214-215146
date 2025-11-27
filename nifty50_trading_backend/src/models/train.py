from __future__ import annotations

import json
import os


from src.models.datasets import load_features_csv, split_xy
from src.models.model import LinearRegressionModel
from src.storage.paths import ensure_dir, timestamped_filename
from src.utils.logging import get_logger

logger = get_logger(__name__)


# PUBLIC_INTERFACE
def train_supervised(features_path: str, model_dir: str, epochs: int = 5, lr: float = 1e-3, batch_size: int = 64) -> tuple[str, dict]:
    """Train a simple CPU-friendly linear regression model.

    The parameters epochs, lr, and batch_size are accepted for API compatibility but are not used
    by this simple trainer.
    """
    df, target_col = load_features_csv(features_path)
    X, y, cols = split_xy(df, target_col)

    model = LinearRegressionModel(l2_reg=1e-6)
    metrics = model.fit(X, y, cols)

    ensure_dir(model_dir)
    model_path = os.path.join(model_dir, timestamped_filename("model", "json"))
    model.save(model_path)

    # Save training metrics as JSON alongside the model
    metrics_path = model_path.replace(".json", ".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Trained model saved to %s", model_path)
    return model_path, metrics
