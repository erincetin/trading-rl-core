# tests/test_alpaca_loader.py
import pandas as pd
import pytest
from alpaca.data.timeframe import TimeFrameUnit

from trading_rl.data.alpaca_loader import (
    AlpacaConfig,
    AlpacaDataLoader,
    _is_crypto_symbol,
    _normalize_crypto_symbol,
)

def test_convert_timeframe():
    tf = AlpacaDataLoader._convert_timeframe("1Min")
    assert tf.amount == 1
    assert tf.unit == TimeFrameUnit.Minute


def test_convert_timeframe_invalid_raises():
    with pytest.raises(ValueError):
        AlpacaDataLoader._convert_timeframe("1Week")

def test_cache_load(tmp_path):
    cfg = AlpacaConfig(
        api_key="X", api_secret="Y",
        cache_dir=str(tmp_path)
    )
    loader = AlpacaDataLoader(cfg)
    loader._fetch_bars = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Network fetch should not occur during cache-based tests")
    )

    # Create fake cache file
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1min"),
        "open": [1,2,3], "high": [1,2,3], "low": [1,2,3],
        "close": [1,2,3], "volume": [10,20,30], "vwap": [1,2,3],
    })

    fpath = tmp_path / "AAPL_2024-01-01_2024-01-02_1Min.csv"
    df.to_csv(fpath, index=False)

    # Load from cache, no API call
    loaded = loader.load(
        "AAPL",
        "2024-01-01",
        "2024-01-02",
        timeframe="1Min",
        use_cache=True,
    )
    assert loaded.shape[0] == 3
    assert "close" in loaded.columns


def test_crypto_symbol_helpers():
    assert _is_crypto_symbol("BTCUSD") is True
    assert _is_crypto_symbol("BTC/USD") is True
    assert _is_crypto_symbol("AAPL") is False

    assert _normalize_crypto_symbol("BTCUSD") == "BTC/USD"
    assert _normalize_crypto_symbol("btc/usd") == "BTC/USD"
