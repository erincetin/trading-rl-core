import numpy as np

from trading_rl.callbacks.eval_callback import (
    _max_drawdown,
    _percentiles,
    _sharpe,
    _win_rate_pct,
    WandbEvalCallback,
)


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


def test_win_rate_pct_counts_positive_steps():
    rets = np.array([-0.01, 0.0, 0.02, 0.03])
    assert abs(_win_rate_pct(rets) - 50.0) < 1e-6


def test_eval_callback_scales_buy_and_hold_curve():
    class _Cfg:
        trading_cost_pct = 0.01
        initial_cash = 1000.0

    class _BaseEnv:
        def __init__(self):
            self.prices = np.array([10.0, 11.0], dtype=np.float32)
            self.config = _Cfg()

        @property
        def unwrapped(self):
            return self

    class _Venv:
        def __init__(self):
            self.envs = [_BaseEnv()]

    class _EvalEnv:
        def __init__(self):
            self.venv = _Venv()

    cb = WandbEvalCallback(_EvalEnv(), eval_freq=1, n_eval_episodes=1)
    curves = cb.get_performance_curves()
    bh_curve = curves["buy_and_hold_curve"]
    assert bh_curve is not None
    assert abs(bh_curve[0] - 990.0) < 1e-6
