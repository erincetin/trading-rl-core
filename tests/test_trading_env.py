# tests/test_trading_env.py
import numpy as np
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
