# tests/test_alpaca_loader.py
import pandas as pd
from trading_rl.data.alpaca_loader import AlpacaDataLoader, AlpacaConfig
from alpaca.data.timeframe import TimeFrameUnit

class DummyClient:
    def get_stock_bars(self, request):
        raise AssertionError("Network call should not occur during cache-based tests")

def test_convert_timeframe():
    cfg = AlpacaConfig(api_key="X", api_secret="Y")
    loader = AlpacaDataLoader(cfg, client=DummyClient())

    tf = loader._convert_timeframe("1Min")
    assert tf.amount == 1
    assert tf.unit == TimeFrameUnit.Minute

def test_cache_load(tmp_path):
    cfg = AlpacaConfig(
        api_key="X", api_secret="Y",
        cache_dir=str(tmp_path)
    )
    loader = AlpacaDataLoader(cfg, client=DummyClient())

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
