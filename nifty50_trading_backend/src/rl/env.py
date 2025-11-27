from __future__ import annotations

import json

from dataclasses import dataclass



import pandas as pd

from src.features.indicators import ema

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CrossoverPolicy:
    """A tiny policy using EMA crossover as a trading signal."""
    short_span: int = 12
    long_span: int = 26
    threshold: float = 0.0  # small threshold to avoid churn

    # PUBLIC_INTERFACE
    def save(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(
                {"type": "CROSSOVER", "short_span": self.short_span, "long_span": self.long_span, "threshold": self.threshold},
                f,
                indent=2,
            )
        return path

    # PUBLIC_INTERFACE
    @staticmethod
    def load(path: str) -> "CrossoverPolicy":
        with open(path, "r") as f:
            d = json.load(f)
        return CrossoverPolicy(short_span=int(d.get("short_span", 12)), long_span=int(d.get("long_span", 26)), threshold=float(d.get("threshold", 0.0)))


# PUBLIC_INTERFACE
def simulate_crossover(df: pd.DataFrame, policy: CrossoverPolicy, episodes: int = 1) -> pd.DataFrame:
    """Simulate a simple long/flat strategy based on EMA crossover.

    Action: go long (1) when ema_short > ema_long + threshold, else flat (0).
    Reward: next period return * position.
    """
    data = df.copy()
    data["ema_short"] = ema(data["close"].astype(float), policy.short_span)
    data["ema_long"] = ema(data["close"].astype(float), policy.long_span)
    data["signal"] = (data["ema_short"] > data["ema_long"] + policy.threshold).astype(int)
    data["ret"] = data["close"].pct_change().shift(-1).fillna(0.0)
    data["position"] = data["signal"]
    data["reward"] = data["position"] * data["ret"]
    data["episode"] = 0  # single pass episode for now
    return data.reset_index(drop=True)
