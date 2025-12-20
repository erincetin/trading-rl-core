# trading_rl/data/alpaca_loader.py

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import os

import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


@dataclass
class AlpacaConfig:
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    timezone: str = "UTC"
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_dir: Optional[str] = None


def _is_crypto_symbol(symbol: str) -> bool:
    s = symbol.upper().strip()
    # Heuristic: "BTC/USD" or "ETH/USD" etc, or "BTCUSD" style
    return ("/" in s) or (s.endswith("USD") and len(s) in (6, 7))  # BTCUSD / DOGEUSD


def _normalize_crypto_symbol(symbol: str) -> str:
    s = symbol.upper().strip()
    # If already "BTC/USD", keep it
    if "/" in s:
        return s
    # If "BTCUSD" -> "BTC/USD"
    if s.endswith("USD") and len(s) >= 6:
        base = s[:-3]
        return f"{base}/USD"
    return s


class AlpacaDataLoader:
    def __init__(self, cfg: AlpacaConfig):
        load_dotenv()

        api_key = cfg.api_key or os.getenv("ALPACA_API_KEY")
        api_secret = cfg.api_secret or os.getenv("ALPACA_API_SECRET")

        self.cfg = cfg
        self.api_key = api_key
        self.api_secret = api_secret

        # Stock client requires keys
        if not api_key or not api_secret:
            raise ValueError(
                "Alpaca API credentials missing. "
                "Set ALPACA_API_KEY and ALPACA_API_SECRET in .env or pass them to AlpacaConfig."
            )
        self.stock_client = StockHistoricalDataClient(api_key, api_secret)

        # Crypto client: keys not required for historical data (Alpaca docs/examples), but should work either way.
        self.crypto_client = CryptoHistoricalDataClient()

        if cfg.cache_dir:
            Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    def load(
        self,
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1Min",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_path = None
        if self.cfg.cache_dir:
            # sanitize filename (slashes, colons)
            safe_symbol = symbol.replace("/", "_")
            fname = f"{safe_symbol}_{start}_{end}_{timeframe}.csv".replace(":", "")
            cache_path = Path(self.cfg.cache_dir) / fname

            if use_cache and cache_path.exists():
                df = pd.read_csv(cache_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp").sort_index()
                return df

        df = self._fetch_bars(symbol, start, end, timeframe)

        if cache_path:
            df.reset_index().to_csv(cache_path, index=False)

        return df

    def _fetch_bars(
        self, symbol: str, start: str, end: str, timeframe: str
    ) -> pd.DataFrame:
        tf = self._convert_timeframe(timeframe)
        is_crypto = _is_crypto_symbol(symbol)

        for attempt in range(self.cfg.max_retries):
            try:
                if is_crypto:
                    sym = _normalize_crypto_symbol(symbol)
                    request = CryptoBarsRequest(
                        symbol_or_symbols=[sym],
                        timeframe=tf,
                        start=start,
                        end=end,
                    )
                    bars = self.crypto_client.get_crypto_bars(request)
                    df = bars.df
                    key = sym
                else:
                    request = StockBarsRequest(
                        symbol_or_symbols=[symbol],
                        timeframe=tf,
                        start=start,
                        end=end,
                    )
                    bars = self.stock_client.get_stock_bars(request)
                    df = bars.df
                    key = symbol

                # If empty, fail with clear message (prevents RangeIndex tz_convert crash)
                if df is None or len(df) == 0:
                    raise RuntimeError(
                        f"No bars returned for symbol={symbol} (resolved={key}), "
                        f"start={start}, end={end}, timeframe={timeframe}."
                    )

                # Extract single symbol from MultiIndex (alpaca-py commonly returns MultiIndex)
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(key, level=0)

                df = df.copy()

                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise RuntimeError(
                        f"Unexpected bars.df index type: {type(df.index)}; "
                        "expected DatetimeIndex. (Data may be empty or malformed.)"
                    )

                # Convert tz
                df.index = df.index.tz_convert(self.cfg.timezone)

                # Standardize columns
                df = df.rename_axis("timestamp").reset_index()
                cols = ["timestamp", "open", "high", "low", "close", "volume"]
                if "vwap" in df.columns:
                    cols.append("vwap")
                df = df[cols]

                df = df.set_index("timestamp").sort_index()
                return df

            except Exception as e:
                print(f"[Alpaca retry {attempt + 1}/{self.cfg.max_retries}] Error: {e}")
                time.sleep(self.cfg.retry_delay)

        raise RuntimeError(f"Failed to fetch bars for {symbol} after retries.")

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
