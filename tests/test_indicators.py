# tests/test_indicators.py
import numpy as np
import pandas as pd
import pytest

from trading_rl.data.indicators import add_talib_indicators


def _make_price_frame(n: int = 120) -> pd.DataFrame:
    # Enough history for the longest indicators (e.g., SMA_50)
    base = np.linspace(1.0, 2.0, n)
    return pd.DataFrame(
        {
            "open": base + 0.01,
            "high": base + 0.02,
            "low": base - 0.02,
            "close": base,
            "volume": np.linspace(100, 200, n),
        }
    )


def test_add_talib_indicators_adds_expected_columns():
    df = _make_price_frame()
    out = add_talib_indicators(df)

    expected = {
        "SMA_5",
        "SMA_20",
        "SMA_50",
        "EMA_12",
        "EMA_26",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "ADX_14",
        "RSI_14",
        "Stoch_K",
        "Stoch_D",
        "ROC_10",
        "VOL_20",
        "VOL_60",
        "DD_60",
        "ATR_14",
        "BB_upper",
        "BB_middle",
        "BB_lower",
        "OBV",
        "AD",
    }

    assert len(out) > 0
    assert expected.issubset(set(out.columns))


def test_add_talib_indicators_drops_nans():
    df = _make_price_frame()
    out = add_talib_indicators(df)

    assert not out.isna().any().any()
    assert out.shape[0] < df.shape[0]  # early rows dropped for warmup


def test_add_talib_indicators_missing_columns_raises():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError):
        add_talib_indicators(df)
