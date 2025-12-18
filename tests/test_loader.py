# tests/test_loader.py
import numpy as np
import pandas as pd
from trading_rl.data.loader import prepare_market_arrays

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
