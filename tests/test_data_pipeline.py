import numpy as np
import pandas as pd
import pytest

import trading_rl.experiment.data_pipeline as data_pipeline
from trading_rl.data.alpaca_loader import AlpacaConfig
from trading_rl.experiment.data_pipeline import (
    clip_to_range,
    load_market_data,
    split_train_eval,
    ts_like_index,
)


def _make_indexed_df(tz: str | None = None) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=6, freq="1min", tz=tz)
    return pd.DataFrame(
        {
            "open": np.arange(6) + 1.0,
            "high": np.arange(6) + 1.0,
            "low": np.arange(6) + 1.0,
            "close": np.arange(6) + 1.0,
            "volume": np.arange(6) + 10.0,
        },
        index=idx,
    )


def test_ts_like_index_aligns_timezone():
    df = _make_indexed_df(tz="UTC")
    ts = ts_like_index(df, "2024-01-01 00:02:00")
    assert ts is not None
    assert ts.tzinfo is not None
    assert ts.tzinfo.tzname(ts) == "UTC"

    df_naive = _make_indexed_df(tz=None)
    ts2 = ts_like_index(df_naive, "2024-01-01T00:02:00+00:00")
    assert ts2 is not None
    assert ts2.tzinfo is None


def test_clip_to_range_filters_data():
    df = _make_indexed_df(tz="UTC")
    out = clip_to_range(df, "2024-01-01 00:01:00", "2024-01-01 00:03:00")
    assert len(out) == 3
    assert out.index.min() == df.index[1]
    assert out.index.max() == df.index[3]


def test_split_train_eval_slices_correctly():
    df = _make_indexed_df(tz="UTC")
    train_df, eval_df = split_train_eval(
        df, eval_start="2024-01-01 00:03:00", eval_end="2024-01-01 00:04:00"
    )
    assert len(eval_df) == 2
    assert eval_df.index.min() == df.index[3]
    assert eval_df.index.max() == df.index[4]
    assert train_df.index.max() == df.index[2]


def test_split_train_eval_empty_eval_raises():
    df = _make_indexed_df(tz="UTC")
    with pytest.raises(ValueError):
        split_train_eval(
            df, eval_start="2024-02-01 00:00:00", eval_end="2024-02-01 00:01:00"
        )


def test_clip_to_range_missing_bounds_raises():
    df = _make_indexed_df(tz="UTC")
    with pytest.raises(ValueError):
        clip_to_range(df, None, "2024-01-01 00:03:00")


def test_load_market_data_from_csv(tmp_path):
    rows = [
        {"timestamp": "2024-01-01 00:00:00", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 10},
        {"timestamp": "2024-01-01 00:01:00", "open": 2, "high": 2, "low": 2, "close": None, "volume": 20},
        {"timestamp": "2024-01-01 00:02:00", "open": 3, "high": 3, "low": 3, "close": 3, "volume": 30},
    ]
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = load_market_data(
        symbol="AAPL",
        start="2024-01-01",
        end="2024-01-02",
        timeframe="1Min",
        warmup_days=0,
        csv_path=str(csv_path),
        alpaca_cfg=None,
    )

    assert "close" in df.columns
    assert df.index.is_monotonic_increasing
    assert len(df) == 3
    assert df["close"].iloc[1] == 1.0


def test_load_market_data_uses_warmup_days(monkeypatch):
    calls = {}

    class DummyLoader:
        def __init__(self, cfg):
            self.cfg = cfg

        def load(self, symbol, start, end, timeframe, use_cache=True):
            calls["start"] = start
            calls["end"] = end
            calls["symbol"] = symbol
            calls["timeframe"] = timeframe
            idx = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
            return pd.DataFrame(
                {
                    "open": [1, 2, 3],
                    "high": [1, 2, 3],
                    "low": [1, 2, 3],
                    "close": [1, 2, 3],
                    "volume": [10, 20, 30],
                },
                index=idx,
            )

    monkeypatch.setattr(data_pipeline, "AlpacaDataLoader", DummyLoader)

    load_market_data(
        symbol="AAPL",
        start="2024-01-10",
        end="2024-01-12",
        timeframe="1Min",
        warmup_days=3,
        csv_path=None,
        alpaca_cfg=AlpacaConfig(api_key="X", api_secret="Y"),
    )

    assert calls["start"] == "2024-01-07"
    assert calls["end"] == "2024-01-12"
    assert calls["symbol"] == "AAPL"


def test_load_market_data_requires_alpaca_cfg_when_no_csv():
    with pytest.raises(ValueError):
        load_market_data(
            symbol="AAPL",
            start="2024-01-01",
            end="2024-01-02",
            timeframe="1Min",
            warmup_days=0,
            csv_path=None,
            alpaca_cfg=None,
        )
