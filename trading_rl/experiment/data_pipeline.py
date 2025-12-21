# trading_rl/experiment/data_pipeline.py
from __future__ import annotations

from typing import Tuple

import pandas as pd

from trading_rl.data.alpaca_loader import AlpacaConfig, AlpacaDataLoader
from trading_rl.data.indicators import add_talib_indicators
from trading_rl.data.loader import load_ohlcv_csv

# -----------------------------
# Time helpers
# -----------------------------


def ts_like_index(df: pd.DataFrame, s: str | None) -> pd.Timestamp | None:
    """
    Parse timestamp string and align it to df.index timezone/naiveness.
    """
    if s is None:
        return None

    ts = pd.Timestamp(s)
    tz = getattr(df.index, "tz", None)

    if tz is not None:
        # df index is tz-aware
        if ts.tzinfo is None:
            return ts.tz_localize(tz)
        return ts.tz_convert(tz)

    # df index is tz-naive
    if ts.tzinfo is not None:
        return ts.tz_convert(None).tz_localize(None)
    return ts


# -----------------------------
# Loading
# -----------------------------


def load_market_data(
    *,
    symbol: str,
    start: str,
    end: str,
    timeframe: str,
    warmup_days: int,
    csv_path: str | None = None,
    alpaca_cfg: AlpacaConfig | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV data, including warmup lookback if using Alpaca.

    Returns a df indexed by timestamp (tz-aware if source provides it),
    columns include at least: open, high, low, close, volume (and possibly vwap).
    """
    if csv_path:
        df = load_ohlcv_csv(csv_path)
        df = df.ffill().dropna()
        return df

    if alpaca_cfg is None:
        raise ValueError("alpaca_cfg must be provided when csv_path is not set")

    # warmup range starts before start
    start_ts = pd.Timestamp(start, tz="UTC")
    warmup_start = (start_ts - pd.Timedelta(days=int(warmup_days))).strftime("%Y-%m-%d")

    loader = AlpacaDataLoader(alpaca_cfg)
    df = loader.load(
        symbol=symbol,
        start=warmup_start,
        end=end,
        timeframe=timeframe,
        use_cache=use_cache,
    )

    df = df.ffill().dropna()
    return df


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering step (currently TA-Lib indicators).
    """
    df_feat = add_talib_indicators(df_raw)
    return df_feat


def clip_to_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Clip df to [start, end] using timezone-aligned timestamps.
    """
    start_ts = ts_like_index(df, start)
    end_ts = ts_like_index(df, end)
    if start_ts is None or end_ts is None:
        raise ValueError("start and end must be provided")

    out = df.loc[start_ts:end_ts]
    if len(out) == 0:
        raise ValueError(
            f"Clip produced empty df: start={start_ts}, end={end_ts}, "
            f"df.min={df.index.min()}, df.max={df.index.max()}"
        )
    return out


def split_train_eval(
    df: pd.DataFrame,
    *,
    eval_start: str | None,
    eval_end: str | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train and eval based on eval_start/eval_end.

    Train is strictly before eval_start (minus 1 second).
    Eval is [eval_start, eval_end].
    """
    if eval_start is None and eval_end is None:
        return df, df

    e_start = ts_like_index(df, eval_start) or df.index.min()
    e_end = ts_like_index(df, eval_end) or df.index.max()

    # sanity: if user asked for eval_start beyond available data
    if eval_start is not None:
        asked = ts_like_index(df, eval_start)
        if asked is not None and asked > df.index.max():
            raise ValueError(
                f"eval_start ({asked}) is after data max ({df.index.max()}). "
                "Increase end to include eval period."
            )

    eval_df = df.loc[e_start:e_end]

    train_end = e_start - pd.Timedelta(seconds=1)
    train_df = df.loc[:train_end]

    if len(eval_df) == 0:
        raise ValueError(
            f"Evaluation slice is empty: eval_start={e_start}, eval_end={e_end}, "
            f"df.min={df.index.min()}, df.max={df.index.max()}"
        )
    if len(train_df) == 0:
        raise ValueError(
            f"Training slice is empty: train_end={train_end}, "
            f"df.min={df.index.min()}, df.max={df.index.max()}"
        )

    return train_df, eval_df
