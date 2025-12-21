import numpy as np

from trading_rl.callbacks.eval_callback import _max_drawdown, _percentiles, _sharpe


def test_max_drawdown_basic():
    pv = np.array([1.0, 1.2, 0.9, 1.1])
    dd = _max_drawdown(pv)
    assert dd < 0.0
    assert abs(dd - (-0.25)) < 1e-6


def test_sharpe_handles_zero_variance():
    rets = np.array([0.01, 0.01, 0.01])
    assert _sharpe(rets) == 0.0


def test_percentiles_empty_returns_zeros():
    out = _percentiles([], ps=(10, 50))
    assert out["p10"] == 0.0
    assert out["p50"] == 0.0
