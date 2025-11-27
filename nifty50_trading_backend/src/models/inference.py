from __future__ import annotations

import os





from src.models.datasets import load_features_csv, split_xy
from src.models.model import LinearRegressionModel
from src.storage.paths import latest_file, timestamped_filename
from src.utils.logging import get_logger

logger = get_logger(__name__)


# PUBLIC_INTERFACE
def predict_supervised(features_path: str, model_dir: str, horizon: int = 1) -> tuple[str, int]:
    """Run inference using the most recent trained model and save predictions CSV.

    Args:
        features_path: Path to features CSV.
        model_dir: Directory where models are saved.
        horizon: Prediction horizon (unused, accepted for API compatibility).

    Returns:
        (predictions_path, count): Path to saved predictions CSV and number of predictions.
    """
    df, target_col = load_features_csv(features_path)
    X, y, cols = split_xy(df, target_col)

    model_path = latest_file(model_dir, patterns=["*.json"])
    if model_path is None:
        raise FileNotFoundError("No trained model found in model directory.")

    model = LinearRegressionModel().load(model_path)
    preds = model.predict(X, cols)

    out_df = df.copy()
    out_df["prediction"] = preds

    pred_path = os.path.join(model_dir, timestamped_filename("predictions", "csv"))
    out_df.to_csv(pred_path, index=False)
    logger.info("Saved predictions to %s", pred_path)
    return pred_path, int(len(out_df))
