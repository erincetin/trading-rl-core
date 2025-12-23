import numpy as np
import pandas as pd
import pytest

from trading_rl.envs.trading_env import TradingEnv, TradingEnvConfig
from trading_rl.envs.windowed_wrapper import WindowedEnvConfig, WindowedTradingEnv
from trading_rl.experiment.data_pipeline import split_train_eval


def _make_prices(n=20, start=1.0, step=0.01):
    return (start + step * np.arange(n)).astype(np.float32)


def _make_features(n=20, f=3):
    return np.zeros((n, f), dtype=np.float32)


def test_windowed_env_seed_determinism():
    prices = _make_prices(50)
    feats = _make_features(50, 2)
    win_cfg = WindowedEnvConfig(window_size=10, random_start=True)
    env = WindowedTradingEnv(prices, feats, TradingEnvConfig(), win_cfg)

    _, info1 = env.reset(seed=123)
    _, info2 = env.reset(seed=123)
    assert info1["window_start"] == info2["window_start"]


def test_env_step_outputs_finite_values():
    prices = _make_prices(30)
    feats = _make_features(30, 4)
    env = TradingEnv(prices, feats, TradingEnvConfig(trading_cost_pct=0.001))
    obs, _ = env.reset(seed=0)

    for _ in range(10):
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert np.isfinite(info["portfolio_value"])
        assert np.isfinite(info["realized_weight"])
        if terminated or truncated:
            break


def test_env_accounting_invariant():
    prices = _make_prices(10, start=10.0, step=0.0)
    feats = _make_features(10, 1)
    env = TradingEnv(prices, feats, TradingEnvConfig(trading_cost_pct=0.0))
    env.reset()

    _, _, _, _, info = env.step([0.6])
    pv = info["portfolio_value"]
    cash = info["cash"]
    pos = info["position"]
    price = info["price"]
    assert abs(pv - (cash + pos * price)) < 1e-6


def test_env_realized_weight_bounds():
    prices = _make_prices(10, start=5.0, step=0.0)
    feats = _make_features(10, 1)
    cfg = TradingEnvConfig(trading_cost_pct=0.0, max_position=0.7)
    env = TradingEnv(prices, feats, cfg)
    env.reset()

    _, _, _, _, info = env.step([1.0])
    assert 0.0 <= info["realized_weight"] <= cfg.max_position + 1e-6


def test_split_train_eval_no_overlap():
    idx = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.arange(10) + 1.0,
            "high": np.arange(10) + 1.0,
            "low": np.arange(10) + 1.0,
            "close": np.arange(10) + 1.0,
            "volume": np.arange(10) + 10.0,
        },
        index=idx,
    )
    train_df, eval_df = split_train_eval(
        df, eval_start="2024-01-01 00:05:00", eval_end="2024-01-01 00:07:00"
    )
    assert train_df.index.max() < eval_df.index.min()


def test_windowed_training_does_not_exceed_eval_slice():
    prices = _make_prices(12)
    feats = _make_features(12, 1)
    win_cfg = WindowedEnvConfig(window_size=5, random_start=False)
    env = WindowedTradingEnv(prices[:6], feats[:6], TradingEnvConfig(), win_cfg)

    _, info = env.reset(seed=0)
    assert info["window_end"] <= 6


def test_env_random_policy_invariants():
    prices = _make_prices(25, start=10.0, step=0.01)
    feats = _make_features(25, 2)
    cfg = TradingEnvConfig(trading_cost_pct=0.001, max_position=1.0)
    env = TradingEnv(prices, feats, cfg)
    env.reset(seed=123)

    rng = np.random.default_rng(0)
    for _ in range(10):
        action = np.array([rng.uniform(-1.0, 2.0)], dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
        assert info["trade_cost"] >= 0.0
        assert 0.0 <= info["realized_weight"] <= cfg.max_position + 1e-6
        assert info["cash"] >= -1e-6
        assert np.isfinite(info["portfolio_value"])
        if terminated or truncated:
            break


def test_windowed_env_global_t_within_window():
    prices = _make_prices(30)
    feats = _make_features(30, 2)
    win_cfg = WindowedEnvConfig(window_size=5, random_start=True)
    env = WindowedTradingEnv(prices, feats, TradingEnvConfig(), win_cfg)

    _, info = env.reset(seed=42)
    start = info["window_start"]
    end = info["window_end"]

    for _ in range(5):
        _, _, _, truncated, info = env.step([0.0])
        gt = info["global_t"]
        assert start <= gt < end
        if truncated:
            break


def test_vecnormalize_wrap_produces_finite_values():
    sb3 = pytest.importorskip("stable_baselines3")

    prices = _make_prices(20, start=2.0, step=0.0)
    feats = _make_features(20, 1)
    env = sb3.common.vec_env.DummyVecEnv(
        [lambda: TradingEnv(prices, feats, TradingEnvConfig())]
    )
    venv = sb3.common.vec_env.VecNormalize(env, norm_obs=True, norm_reward=False)

    obs = venv.reset()
    action = np.array([[0.5]], dtype=np.float32)
    obs, rewards, dones, infos = venv.step(action)

    assert np.isfinite(obs).all()
    assert np.isfinite(rewards).all()


def test_sb3_smoke_train_short_rollout():
    sb3 = pytest.importorskip("stable_baselines3")

    prices = _make_prices(15, start=1.0, step=0.01)
    feats = _make_features(15, 2)
    env = sb3.common.vec_env.DummyVecEnv(
        [lambda: TradingEnv(prices, feats, TradingEnvConfig(trading_cost_pct=0.0))]
    )

    model = sb3.PPO("MlpPolicy", env, verbose=0, n_steps=8, batch_size=4, device="cpu")
    model.learn(total_timesteps=16)


def test_full_rollout_determinism_with_seed():
    prices = _make_prices(12, start=2.0, step=0.01)
    feats = _make_features(12, 2)
    cfg = TradingEnvConfig(trading_cost_pct=0.0, reward_mode="diff_return")

    def rollout(seed: int):
        env = TradingEnv(prices, feats, cfg)
        obs, _ = env.reset(seed=seed)
        rewards = []
        pvs = []
        for _ in range(5):
            action = np.array([0.5], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            pvs.append(float(info["portfolio_value"]))
            if terminated or truncated:
                break
        return rewards, pvs

    rewards1, pvs1 = rollout(123)
    rewards2, pvs2 = rollout(123)
    assert rewards1 == rewards2
    assert pvs1 == pvs2
