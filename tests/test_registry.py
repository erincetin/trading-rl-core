import numpy as np
import pytest

from trading_rl.envs.trading_env import TradingEnv, TradingEnvConfig
from trading_rl.registry import (
    _build_vec_env,
    get_algo_builder,
    get_env_builder,
    maybe_wrap_vecnormalize,
)


def _make_env():
    prices = np.array([1.0, 1.1], dtype=np.float32)
    feats = np.zeros((2, 1), dtype=np.float32)
    return TradingEnv(prices, feats, TradingEnvConfig())


def test_get_algo_builder_unknown_raises():
    with pytest.raises(KeyError):
        get_algo_builder("nope")


def test_get_env_builder_unknown_raises():
    with pytest.raises(KeyError):
        get_env_builder("nope")


def test_build_vec_env_invalid_type_raises():
    with pytest.raises(ValueError):
        _build_vec_env([lambda: _make_env()], "invalid")


def test_maybe_wrap_vecnormalize_toggle():
    sb3 = pytest.importorskip("stable_baselines3")
    env = sb3.common.vec_env.DummyVecEnv([lambda: _make_env()])

    same = maybe_wrap_vecnormalize(env, enable=False)
    assert same is env

    wrapped = maybe_wrap_vecnormalize(env, enable=True, training=False)
    assert hasattr(wrapped, "training")
    assert wrapped.training is False


def test_windowed_env_builder_respects_num_envs():
    pytest.importorskip("stable_baselines3")
    prices = np.linspace(1, 2, 6).astype(np.float32)
    feats = np.zeros((6, 2), dtype=np.float32)

    env_builder = get_env_builder("windowed")
    train_env, eval_env = env_builder.factory(
        prices,
        feats,
        prices,
        feats,
        {
            "window_size": 3,
            "n_envs_train": 2,
            "n_envs_eval": 1,
            "vec_env_type": "dummy",
            "random_start": False,
        },
    )

    assert train_env.num_envs == 2
    assert eval_env.num_envs == 1

    obs = train_env.reset()
    action = np.zeros((train_env.num_envs, 1), dtype=np.float32)
    step_out = train_env.step(action)
    assert len(step_out) == 4
