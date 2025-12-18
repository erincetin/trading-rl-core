# tests/test_windowed_env.py
import numpy as np
from trading_rl.envs.windowed_wrapper import (
    WindowedTradingEnv, WindowedEnvConfig
)
from trading_rl.envs.trading_env import TradingEnvConfig

def test_windowed_env_basic():
    prices = np.linspace(1, 10, 10).astype(np.float32)     # T=10
    feats = np.zeros((10, 3), dtype=np.float32)
    win_cfg = WindowedEnvConfig(window_size=5, random_start=False)
    env_cfg = TradingEnvConfig()

    env = WindowedTradingEnv(prices, feats, env_cfg, win_cfg)

    obs, info = env.reset()
    assert info["window_start"] == 0
    assert obs.shape[-1] > 1

def test_window_random_start():
    prices = np.linspace(1, 10, 20).astype(np.float32)
    feats = np.zeros((20, 3), dtype=np.float32)
    win_cfg = WindowedEnvConfig(window_size=5, random_start=True)

    env = WindowedTradingEnv(prices, feats, TradingEnvConfig(), win_cfg)

    obs, info = env.reset()
    start = info["window_start"]
    # need window_size+1 points, so max_start = T - (window_size+1)
    assert 0 <= start <= 14

def test_window_step_maps_global_t():
    prices = np.linspace(1, 10, 10).astype(np.float32)
    feats = np.zeros((10, 2), dtype=np.float32)
    cfg = TradingEnvConfig()
    win_cfg = WindowedEnvConfig(window_size=4, random_start=False)

    env = WindowedTradingEnv(prices, feats, cfg, win_cfg)
    env.reset()  # window: [0:4]

    obs, reward, terminated, truncated, info = env.step([0.0])
    assert info["global_t"] == 1


def test_window_horizon_is_truncation_not_termination():
    prices = np.linspace(1, 10, 10).astype(np.float32)
    feats = np.zeros((10, 2), dtype=np.float32)
    win_cfg = WindowedEnvConfig(window_size=3, random_start=False)
    env = WindowedTradingEnv(prices, feats, TradingEnvConfig(), win_cfg)
    env.reset()

    # exactly window_size steps
    terminated = False
    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step([0.0])

    assert terminated is False
    assert truncated is True
