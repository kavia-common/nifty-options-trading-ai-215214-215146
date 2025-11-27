import os

from src.config.settings import get_settings
from src.features.pipeline import build_features
from src.services.rl_service import RLService


def test_rl_train_and_sim():
    settings = get_settings()

    # Make small synthetic data
    import numpy as np
    import pandas as pd

    n = 250
    rng = np.random.default_rng(7)
    logret = rng.normal(0, 0.003, size=n)
    close = 100 * np.exp(np.cumsum(logret))
    high = close * (1 + rng.uniform(0, 0.01, size=n))
    low = close * (1 - rng.uniform(0, 0.01, size=n))
    openp = close * (1 + rng.uniform(-0.005, 0.005, size=n))
    vol = rng.integers(1000, 10000, size=n)
    df = pd.DataFrame({"open": openp, "high": high, "low": low, "close": close, "volume": vol})
    raw_path = os.path.join(settings.DATA_DIR, "synthetic_rl.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df.to_csv(raw_path, index=False)

    feat_path, _ = build_features(raw_path, settings.FEATURES_DIR, indicators=["ema12", "ema26"])
    rl = RLService(settings=settings)
    policy_path, algo = rl.train(timesteps=100, algo="PPO", features_path=feat_path)
    assert os.path.exists(policy_path)
    assert algo in ("PPO", "A2C")
    results_path, episodes = rl.simulate(episodes=1, policy_path=policy_path, features_path=feat_path)
    assert os.path.exists(results_path)
    assert episodes == 1
