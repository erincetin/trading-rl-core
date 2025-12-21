# tests/test_loader.py
import numpy as np
import pandas as pd
import pytest

from trading_rl.data.loader import load_ohlcv_csv, prepare_market_arrays

def test_prepare_market_arrays_basic():
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1min"),
        "open": [1, 2, 3, 4, 5],
        "high": [1, 2, 3, 4, 5],
        "low":  [1, 2, 3, 4, 5],
        "close": [1, 2, 3, 4, 5],
        "volume": [10, 20, 30, 40, 50],
    }
    df = pd.DataFrame(data).set_index("timestamp")

    md = prepare_market_arrays(df, price_col="close")

    assert md.prices.shape == (5,)
    assert md.features.shape[0] == 5
    assert md.df.shape[0] == 5
    assert np.allclose(md.prices, [1,2,3,4,5])

def test_prepare_market_arrays_feature_selection():
    df = pd.DataFrame({
        "close": [1,2,3],
        "feat1": [10,20,30],
        "feat2": [100,200,300],
    })

    md = prepare_market_arrays(df, price_col="close", feature_cols=["feat1"])

    assert md.features.shape == (3,1)
    assert np.allclose(md.features[:,0], [10,20,30])


def test_prepare_market_arrays_default_features_excludes_price():
    df = pd.DataFrame(
        {
            "close": [1, 2, 3],
            "volume": [10, 20, 30],
            "symbol": ["A", "A", "A"],
        }
    )

    md = prepare_market_arrays(df, price_col="close")

    assert md.features.shape == (3, 1)
    assert np.allclose(md.features[:, 0], [10, 20, 30])


def test_prepare_market_arrays_missing_price_raises():
    df = pd.DataFrame({"open": [1, 2, 3]})
    with pytest.raises(ValueError):
        prepare_market_arrays(df, price_col="close")


def test_load_ohlcv_csv_parses_and_sorts(tmp_path):
    rows = [
        {"timestamp": "2024-01-01 00:01:00", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 10},
        {"timestamp": "2024-01-01 00:00:00", "open": 2, "high": 2, "low": 2, "close": 2, "volume": 20},
    ]
    path = tmp_path / "prices.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    df = load_ohlcv_csv(path, tz="UTC")
    assert df.index.is_monotonic_increasing
    assert df.index.tz is not None
    assert df.index[0].strftime("%H:%M:%S") == "00:00:00"


def test_load_ohlcv_csv_missing_timestamp_raises(tmp_path):
    rows = [{"time": "2024-01-01", "open": 1, "close": 1}]
    path = tmp_path / "bad.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_ohlcv_csv(path)
