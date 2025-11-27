import os

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.features.pipeline import build_features


def make_synthetic_ohlcv_csv(path: str, n: int = 200) -> str:
    rng = np.random.default_rng(0)
    # random walk close prices
    logret = rng.normal(0, 0.002, size=n)
    close = 100 * np.exp(np.cumsum(logret))
    high = close * (1 + rng.uniform(0, 0.01, size=n))
    low = close * (1 - rng.uniform(0, 0.01, size=n))
    openp = close * (1 + rng.uniform(-0.005, 0.005, size=n))
    vol = rng.integers(1000, 10000, size=n)
    df = pd.DataFrame(
        {"open": openp, "high": high, "low": low, "close": close, "volume": vol}
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def test_build_features():
    settings = get_settings()
    raw_path = os.path.join(settings.DATA_DIR, "synthetic.csv")
    make_synthetic_ohlcv_csv(raw_path, n=250)
    out_path, rows = build_features(raw_path, settings.FEATURES_DIR, indicators=["rsi14", "ema12", "ema26", "macd"])
    assert os.path.exists(out_path)
    assert rows > 0
