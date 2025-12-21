from types import SimpleNamespace

import pytest

from trading_rl.experiment.regimes import (
    apply_regime,
    load_regimes_from_hyperparams,
)


def _base_regime():
    return {
        "name": "r0",
        "start": "2024-01-01",
        "end": "2024-01-02",
        "eval_start": "2024-01-01",
        "eval_end": "2024-01-02",
    }


def test_load_regimes_from_hyperparams_valid():
    hp = {"regimes": [_base_regime()]}
    out = load_regimes_from_hyperparams(hp)
    assert out[0]["name"] == "r0"


def test_load_regimes_from_hyperparams_missing_keys():
    hp = {"regimes": [{"name": "bad"}]}
    with pytest.raises(ValueError):
        load_regimes_from_hyperparams(hp)


def test_apply_regime_overrides_args():
    args = SimpleNamespace(
        symbol="AAPL",
        timeframe="15Min",
        warmup_days=10,
        start="2024-01-01",
        end="2024-12-31",
        eval_start=None,
        eval_end=None,
    )

    regime = _base_regime()
    regime["symbol"] = "BTCUSD"
    regime["timeframe"] = "1Min"
    regime["warmup_days"] = 5

    out = apply_regime(args, regime)
    assert out.symbol == "BTCUSD"
    assert out.timeframe == "1Min"
    assert out.warmup_days == 5
    assert out.regime_name == "r0"
