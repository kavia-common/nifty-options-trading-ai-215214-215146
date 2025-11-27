import os

from src.config.settings import get_settings
from src.features.pipeline import build_features
from src.models.inference import predict_supervised
from src.models.train import train_supervised


def test_train_and_predict_end_to_end():
    settings = get_settings()

    # Create a small synthetic dataset for features
    import numpy as np
    import pandas as pd

    n = 300
    rng = np.random.default_rng(42)
    logret = rng.normal(0, 0.002, size=n)
    close = 100 * np.exp(np.cumsum(logret))
    high = close * (1 + rng.uniform(0, 0.01, size=n))
    low = close * (1 - rng.uniform(0, 0.01, size=n))
    openp = close * (1 + rng.uniform(-0.005, 0.005, size=n))
    vol = rng.integers(1000, 10000, size=n)
    df = pd.DataFrame({"open": openp, "high": high, "low": low, "close": close, "volume": vol})
    raw_path = os.path.join(settings.DATA_DIR, "synthetic_model.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df.to_csv(raw_path, index=False)

    feat_path, _ = build_features(raw_path, settings.FEATURES_DIR, indicators=["rsi14", "ema12", "ema26", "macd"])

    model_path, metrics = train_supervised(feat_path, settings.MODEL_DIR, epochs=3)
    assert os.path.exists(model_path)
    assert "mse" in metrics

    pred_path, cnt = predict_supervised(feat_path, settings.MODEL_DIR)
    assert os.path.exists(pred_path)
    assert cnt > 0
