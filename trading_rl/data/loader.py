# trading_rl/data/loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    """
    Container for prepared market data.
    """
    df: pd.DataFrame          # full OHLCV+indicators dataframe, indexed by datetime
    prices: np.ndarray        # shape (T,)
    features: np.ndarray      # shape (T, F)


def load_ohlcv_csv(
    path: str | Path,
    parse_dates: str = "timestamp",
    tz: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLCV-like data from CSV.

    Expected columns:
        timestamp, open, high, low, close, volume, ...

    - Parses datetime column.
    - Sets it as index.
    - Sorts by time.
    """
    path = Path(path)
    df = pd.read_csv(path)

    if parse_dates not in df.columns:
        raise ValueError(f"Expected datetime column '{parse_dates}' in CSV, got: {df.columns.tolist()}")

    df[parse_dates] = pd.to_datetime(df[parse_dates])
    if tz is not None:
        df[parse_dates] = df[parse_dates].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")

    df = df.sort_values(parse_dates).set_index(parse_dates)
    return df


def prepare_market_arrays(
    df: pd.DataFrame,
    price_col: str = "close",
    feature_cols: Optional[Iterable[str]] = None,
    dropna: bool = True,
) -> MarketData:
    """
    Turn a dataframe into (prices, features) arrays.

    - price_col: used for trading & portfolio valuation
    - feature_cols: None → use OHLCV + any extra columns except price_col
    """
    if price_col not in df.columns:
        raise ValueError(f"price_col '{price_col}' not in df columns: {df.columns.tolist()}")

    _df = df.copy()

    if dropna:
        _df = _df.dropna(how="any")

    prices = _df[price_col].to_numpy(dtype="float32")

    if feature_cols is None:
        # Default: all numeric columns except price_col
        num_cols = _df.select_dtypes(include=["number"]).columns.tolist()
        feature_cols = [c for c in num_cols if c != price_col]

    feature_cols = list(feature_cols)
    if not feature_cols:
        # At least give the price as a feature
        features = prices.reshape(-1, 1)
    else:
        features = _df[feature_cols].to_numpy(dtype="float32")

    return MarketData(df=_df, prices=prices, features=features)
