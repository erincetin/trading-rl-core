# tests/test_trading_env.py
import numpy as np
import pytest

from trading_rl.envs.trading_env import TradingEnv, TradingEnvConfig

def make_simple_env():
    prices = np.array([1.0, 1.1, 1.2], dtype=np.float32)
    features = np.zeros((3, 2), dtype=np.float32)
    env = TradingEnv(prices, features, TradingEnvConfig(trading_cost_pct=0.0))
    return env

def test_env_reset():
    env = make_simple_env()
    obs, info = env.reset()

    assert obs.shape[-1] >= 2  # features + metadata
    assert info["t"] == 0
    assert info["portfolio_value"] == env.config.initial_cash

def test_env_step_increases_time():
    env = make_simple_env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step([0.0])
    assert info["t"] == 1

def test_env_reward_logic_long_position():
    prices = np.array([1.0, 1.1, 1.2], dtype=np.float32)
    feats = np.zeros((3,2), dtype=np.float32)
    
    env = TradingEnv(
        prices, feats,
        TradingEnvConfig(trading_cost_pct=0.0)
    )
    env.reset()

    obs, reward, term, trunc, info = env.step([1.0])
    
    assert reward > 0
    assert abs(reward - 0.10) < 1e-6

def test_env_terminates_at_end():
    env = make_simple_env()
    env.reset()
    env.step([0.5])  # t=1
    obs, reward, terminated, truncated, info = env.step([0.5])  # t=2 final
    assert terminated


def test_env_action_clipping_and_realized_weight():
    prices = np.array([10.0, 10.0], dtype=np.float32)
    feats = np.zeros((2, 1), dtype=np.float32)
    cfg = TradingEnvConfig(trading_cost_pct=0.0, max_position=0.5)
    env = TradingEnv(prices, feats, cfg)
    env.reset()

    _, _, _, _, info = env.step([1.0])  # clipped to 0.5

    assert abs(info["action_target_weight"] - 0.5) < 1e-6
    assert abs(info["realized_weight"] - 0.5) < 1e-6


def test_env_reward_modes_log_and_pnl():
    prices = np.array([10.0, 11.0], dtype=np.float32)
    feats = np.zeros((2, 1), dtype=np.float32)

    env_log = TradingEnv(
        prices, feats, TradingEnvConfig(trading_cost_pct=0.0, reward_mode="log")
    )
    env_log.reset()
    _, reward_log, *_ = env_log.step([1.0])
    assert abs(reward_log - np.log(1.1)) < 1e-6

    env_pnl = TradingEnv(
        prices, feats, TradingEnvConfig(trading_cost_pct=0.0, reward_mode="pnl")
    )
    env_pnl.reset()
    _, reward_pnl, *_ = env_pnl.step([1.0])
    assert abs(reward_pnl - 100_000.0) < 1e-6


def test_env_trade_cost_limits_full_investment():
    prices = np.array([10.0, 10.0], dtype=np.float32)
    feats = np.zeros((2, 1), dtype=np.float32)
    cfg = TradingEnvConfig(trading_cost_pct=0.01, initial_cash=100.0)
    env = TradingEnv(prices, feats, cfg)
    env.reset()

    _, _, _, _, info = env.step([1.0])

    assert info["trade_cost"] > 0.0
    assert abs(info["realized_weight"] - 1.0) < 1e-6
    assert info["portfolio_value_next"] < cfg.initial_cash


def test_env_history_records_steps():
    env = make_simple_env()
    env.reset()
    env.step([0.25])
    env.step([0.5])

    history = env.history
    assert len(history["t"]) == 2
    assert len(history["price_exec"]) == 2
    assert len(history["price_next"]) == 2


def test_env_observation_shape_without_metadata():
    prices = np.array([1.0, 1.0], dtype=np.float32)
    feats = np.zeros((2, 4), dtype=np.float32)
    cfg = TradingEnvConfig(
        obs_include_cash=False, obs_include_position=False, obs_include_time=False
    )
    env = TradingEnv(prices, feats, cfg)
    obs, _ = env.reset()
    assert obs.shape == (4,)


def test_env_cash_and_position_after_trade():
    prices = np.array([10.0, 10.0], dtype=np.float32)
    feats = np.zeros((2, 1), dtype=np.float32)
    cfg = TradingEnvConfig(trading_cost_pct=0.0, initial_cash=100.0)
    env = TradingEnv(prices, feats, cfg)
    env.reset()

    _, reward, _, _, info = env.step([0.5])

    assert abs(info["cash"] - 50.0) < 1e-6
    assert abs(info["position"] - 5.0) < 1e-6
    assert abs(info["portfolio_value"] - 100.0) < 1e-6
    assert abs(reward) < 1e-6


def test_env_reward_scaling_applies():
    prices = np.array([1.0, 2.0], dtype=np.float32)
    feats = np.zeros((2, 1), dtype=np.float32)
    cfg = TradingEnvConfig(trading_cost_pct=0.0, reward_scaling=10.0)
    env = TradingEnv(prices, feats, cfg)
    env.reset()

    _, reward, *_ = env.step([1.0])
    assert abs(reward - 10.0) < 1e-6


def test_env_observation_metadata_values():
    prices = np.array([1.0, 1.0], dtype=np.float32)
    feats = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    cfg = TradingEnvConfig(
        obs_include_cash=True, obs_include_position=True, obs_include_time=True
    )
    env = TradingEnv(prices, feats, cfg)
    obs, _ = env.reset()

    assert np.allclose(obs[:2], [2.0, 3.0])
    assert abs(obs[-3] - 1.0) < 1e-6  # cash_frac
    assert abs(obs[-2] - 0.0) < 1e-6  # pos_frac
    assert abs(obs[-1] - 0.0) < 1e-6  # time_frac


def test_env_action_clips_negative_to_zero():
    prices = np.array([10.0, 10.0], dtype=np.float32)
    feats = np.zeros((2, 1), dtype=np.float32)
    env = TradingEnv(prices, feats, TradingEnvConfig(trading_cost_pct=0.0))
    env.reset()

    _, _, _, _, info = env.step(-1.0)
    assert abs(info["action_target_weight"] - 0.0) < 1e-6


def test_env_unknown_reward_mode_raises():
    prices = np.array([1.0, 1.0], dtype=np.float32)
    feats = np.zeros((2, 1), dtype=np.float32)
    env = TradingEnv(
        prices, feats, TradingEnvConfig(trading_cost_pct=0.0, reward_mode="nope")
    )
    env.reset()

    with pytest.raises(ValueError):
        env.step([0.0])


def test_env_validates_input_shapes():
    prices = np.zeros((2, 1), dtype=np.float32)
    feats = np.zeros((2, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        TradingEnv(prices, feats, TradingEnvConfig())

    prices = np.zeros((2,), dtype=np.float32)
    feats = np.zeros((2,), dtype=np.float32)
    with pytest.raises(ValueError):
        TradingEnv(prices, feats, TradingEnvConfig())


def test_env_validates_length_mismatch():
    prices = np.array([1.0, 2.0], dtype=np.float32)
    feats = np.zeros((3, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        TradingEnv(prices, feats, TradingEnvConfig())


def test_env_rejects_non_positive_prices():
    prices = np.array([1.0, 0.0], dtype=np.float32)
    feats = np.zeros((2, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        TradingEnv(prices, feats, TradingEnvConfig())


def test_env_rejects_non_finite_features():
    prices = np.array([1.0, 1.0], dtype=np.float32)
    feats = np.array([[0.0], [np.nan]], dtype=np.float32)
    with pytest.raises(ValueError):
        TradingEnv(prices, feats, TradingEnvConfig())
