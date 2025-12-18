# trading_rl/data/alpaca_loader.py

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from dotenv import load_dotenv
import os

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.models.bars import BarSet
from alpaca.data.enums import DataFeed

import numpy as np


# ---------------------------------------------------------------------
# Dataclass configuration
# ---------------------------------------------------------------------


@dataclass
class AlpacaConfig:
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    timezone: str = "UTC"
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_dir: Optional[str] = None


# ---------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------


class AlpacaDataLoader:
    def __init__(self, cfg: AlpacaConfig, client: Optional[StockHistoricalDataClient] = None):
        load_dotenv()

        api_key = cfg.api_key or os.getenv("ALPACA_API_KEY")
        api_secret = cfg.api_secret or os.getenv("ALPACA_API_SECRET")

        self.cfg = cfg
        self.api_key = api_key
        self.api_secret = api_secret

        if client is not None:
            self.client = client
        else:
            if not api_key or not api_secret:
                raise ValueError(
                    "Alpaca API credentials missing. "
                    "Set ALPACA_API_KEY and ALPACA_API_SECRET in .env or pass them to AlpacaConfig."
                )

            self.client = StockHistoricalDataClient(api_key, api_secret)

        if cfg.cache_dir:
            Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------

    def load(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1Min",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load OHLCV data for one symbol.

        timeframe examples:
            "1Min", "5Min", "15Min", "1Hour", "1Day"
        """

        cache_path = None
        if self.cfg.cache_dir:
            fname = f"{symbol}_{start}_{end}_{timeframe}.csv".replace(":", "")
            cache_path = Path(self.cfg.cache_dir) / fname

            if use_cache and cache_path.exists():
                df = pd.read_csv(cache_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
                return df

        df = self._fetch_bars(symbol, start, end, timeframe)
        # Save cache
        if cache_path:
            df.to_csv(cache_path)

        return df

    # -------------------------------------------------------------
    # Internal function to fetch + retry Alpaca calls
    # -------------------------------------------------------------

    def _fetch_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str,
    ) -> pd.DataFrame:
        tf = self._convert_timeframe(timeframe)

        for attempt in range(self.cfg.max_retries):
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=[symbol], timeframe=tf, start=start, end=end
                )
                bars = self.client.get_stock_bars(request)
                df = bars.df  # alpaca-py returns a MultiIndex df

                # Extract single symbol without levels
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(symbol, level=0)

                df = df.copy()
                df.index = df.index.tz_convert(self.cfg.timezone)

                df = df.rename_axis("timestamp").reset_index()

                # Keep standard OHLCV columns + vwap
                df = df[["timestamp", "open", "high", "low", "close", "volume", "vwap"]]

                df = df.set_index("timestamp").sort_index()

                return df

            except Exception as e:
                print(f"[Alpaca retry {attempt + 1}/{self.cfg.max_retries}] Error: {e}")
                time.sleep(self.cfg.retry_delay)

        raise RuntimeError(f"Failed to fetch bars for {symbol} after retries.")

    # -------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------

    @staticmethod
    def _convert_timeframe(tf: str):
        tf = tf.strip().lower()

        if tf.endswith("min"):
            n = int(tf.replace("min", ""))
            return TimeFrame(n, TimeFrameUnit.Minute)

        if tf.endswith("hour"):
            n = int(tf.replace("hour", ""))
            return TimeFrame(n, TimeFrameUnit.Hour)

        if tf.endswith("day"):
            n = int(tf.replace("day", ""))
            return TimeFrame(n, TimeFrameUnit.Day)

        raise ValueError(f"Unsupported timeframe: {tf}")
