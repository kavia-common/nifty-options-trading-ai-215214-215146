from __future__ import annotations

import os
from datetime import datetime

import yfinance as yf

from src.utils.logging import get_logger

logger = get_logger(__name__)


# PUBLIC_INTERFACE
def ingest_ohlcv(ticker: str, period: str, interval: str, out_dir: str) -> tuple[str, int]:
    """Fetch OHLCV data via yfinance and save to CSV in out_dir.

    Args:
        ticker: Symbol to download (e.g., ^NSEI for NIFTY50)
        period: Period string for yfinance (e.g., '1y', '6mo', '5y')
        interval: Interval string for yfinance (e.g., '1d', '1h', '15m')
        out_dir: Directory to save CSV

    Returns:
        tuple[path, rows]: Path to saved CSV and number of rows
    """
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)

    if df is None or df.empty:
        raise ValueError("No data returned from yfinance")

    # Ensure index is a column and standardize names
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_ticker = str(ticker).replace("^", "").replace("/", "_").replace("\\", "_")
    filename = f"{safe_ticker}_{period}_{interval}_{now}.csv"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    df.to_csv(out_path, index=False)
    logger.info("Saved data to %s rows=%d", out_path, len(df))
    return out_path, int(len(df))
