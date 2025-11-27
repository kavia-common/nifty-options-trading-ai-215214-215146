from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# PUBLIC_INTERFACE
def load_features_csv(path: str) -> tuple[pd.DataFrame, str]:
    """Load features CSV and return dataframe and the target column name."""
    df = pd.read_csv(path)
    if "target_return" not in df.columns:
        raise ValueError("Features CSV must include a 'target_return' column as target.")
    return df, "target_return"


# PUBLIC_INTERFACE
def split_xy(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Split dataframe into X, y numpy arrays and a list of feature column names.

    Excludes non-numeric columns and the target column from features.
    """
    # Keep only numeric columns
    num_df = df.select_dtypes(include=["number"])
    if target_col not in num_df.columns:
        # Target could be numeric but dropped; re-attach from original
        num_df[target_col] = df[target_col]

    feature_cols: List[str] = [c for c in num_df.columns if c != target_col]
    X = num_df[feature_cols].to_numpy(dtype=float)
    y = num_df[target_col].to_numpy(dtype=float)
    return X, y, feature_cols
