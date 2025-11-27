from __future__ import annotations

import os
from typing import Tuple, List

import pandas as pd

from src.features.indicators import add_indicators
from src.storage.paths import ensure_dir, timestamped_filename
from src.utils.logging import get_logger

logger = get_logger(__name__)


# PUBLIC_INTERFACE
def build_features(input_csv: str, output_dir: str, indicators: List[str] | None = None) -> Tuple[str, int]:
    """Build engineered features from a raw OHLCV CSV and save to output_dir.

    The pipeline:
      - Loads CSV
      - Normalizes column names to lower_snake_case
      - Computes selected indicators
      - Computes return-based target: next period % change of close (target_return)
      - Drops rows with NaNs created by indicators/returns
      - Saves output CSV

    Args:
        input_csv: Path to OHLCV CSV.
        output_dir: Directory where features CSV is written.
        indicators: Optional list of indicator names to compute.

    Returns:
        (path, rows): Path to features CSV and number of rows.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    # Normalize column names
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    # Basic OHLCV presence; fill missing optional columns with 0 if absent
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            if col == "close":
                raise ValueError("Input CSV must contain a 'close' column.")
            df[col] = 0.0

    # Compute indicators
    df = add_indicators(df, indicators or [])

    # Compute forward return as target (predict next period return from current features)
    df["return"] = df["close"].pct_change()
    df["target_return"] = df["return"].shift(-1)

    # Drop rows with NaN from indicators and returns
    df = df.dropna().reset_index(drop=True)

    ensure_dir(output_dir)
    out_name = timestamped_filename("features", "csv")
    out_path = os.path.join(output_dir, out_name)
    df.to_csv(out_path, index=False)

    logger.info("Saved features to %s rows=%d", out_path, len(df))
    return out_path, int(len(df))
