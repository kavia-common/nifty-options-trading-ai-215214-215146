from __future__ import annotations


import pandas as pd


# PUBLIC_INTERFACE
def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average of a price series."""
    return series.ewm(span=span, adjust=False).mean()


# PUBLIC_INTERFACE
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI) calculation."""
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


# PUBLIC_INTERFACE
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD indicator: line, signal, and histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}
    )


# PUBLIC_INTERFACE
def add_indicators(df: pd.DataFrame, indicator_names: list[str]) -> pd.DataFrame:
    """Add requested indicators to the dataframe.

    Supported names (case-insensitive):
      - 'rsi', 'rsi14'
      - 'ema12', 'ema26', 'ema9', 'ema50', 'ema200'
      - 'macd' (adds macd, macd_signal, macd_hist)

    The function returns a new dataframe with added columns.
    """
    df = df.copy()
    close = df["close"].astype(float)
    normalized = [n.strip().lower() for n in indicator_names or []]

    # If no indicators provided, use sensible defaults
    if not normalized:
        normalized = ["rsi14", "ema12", "ema26", "macd"]

    for name in normalized:
        if name.startswith("rsi"):
            period = 14
            try:
                period = int(name.replace("rsi", ""))
            except Exception:
                pass
            df[f"rsi_{period}"] = rsi(close, period)
        elif name.startswith("ema"):
            try:
                span = int(name.replace("ema", ""))
            except Exception:
                span = 12
            df[f"ema_{span}"] = ema(close, span)
        elif name == "macd":
            macd_df = macd(close)
            df = pd.concat([df, macd_df], axis=1)
        else:
            # Ignore unknown indicator names gracefully
            continue

    return df
