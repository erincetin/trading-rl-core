"""
Registries for algorithms and environments.

These provide string-to-builder mappings so the runner can construct
algos/envs by name and stay extensible for custom additions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from trading_rl.envs.trading_env import TradingEnv, TradingEnvConfig
from trading_rl.envs.windowed_wrapper import (
    WindowedEnvConfig,
    WindowedTradingEnv,
)

# -----------------------------
# Algo registry
# -----------------------------


@dataclass
class AlgoBuilder:
    """Metadata and builder for an algo."""

    name: str
    algo_cls: type
    factory: Callable[[VecEnv, dict], object]


def _ppo_factory(env: VecEnv, params: dict):
    policy = params.pop("policy", "MlpPolicy")
    return PPO(policy, env, **params)


def _a2c_factory(env: VecEnv, params: dict):
    policy = params.pop("policy", "MlpPolicy")
    return A2C(policy, env, **params)


def _sac_factory(env: VecEnv, params: dict):
    policy = params.pop("policy", "MlpPolicy")
    return SAC(policy, env, **params)


def _td3_factory(env: VecEnv, params: dict):
    policy = params.pop("policy", "MlpPolicy")
    return TD3(policy, env, **params)


ALGO_REGISTRY: Dict[str, AlgoBuilder] = {
    "ppo": AlgoBuilder("ppo", PPO, _ppo_factory),
    "a2c": AlgoBuilder("a2c", A2C, _a2c_factory),
    "sac": AlgoBuilder("sac", SAC, _sac_factory),
    "td3": AlgoBuilder("td3", TD3, _td3_factory),
}


def get_algo_builder(name: str) -> AlgoBuilder:
    key = name.lower()
    if key not in ALGO_REGISTRY:
        raise KeyError(f"Unknown algo '{name}'. Available: {list(ALGO_REGISTRY)}")
    return ALGO_REGISTRY[key]


# -----------------------------
# Env registry
# -----------------------------


@dataclass
class EnvBuilder:
    """Metadata and builder for an environment."""

    name: str
    factory: Callable[
        [np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict], Tuple[VecEnv, VecEnv]
    ]


def _make_windowed_env(
    train_prices: np.ndarray,
    train_features: np.ndarray,
    eval_prices: np.ndarray,
    eval_features: np.ndarray,
    cfg: dict,
):
    requested_window = int(cfg.get("window_size", min(512, len(train_prices) - 1)))
    # WindowedTradingEnv needs window_size+1 points to run window_size steps.
    window_size = min(requested_window, max(1, len(train_prices) - 1))
    window_cfg = cfg.get("window_cfg") or WindowedEnvConfig(
        window_size=window_size,
        random_start=cfg.get("random_start", True),
    )
    env_cfg = cfg.get("env_cfg") or TradingEnvConfig(
        trading_cost_pct=float(cfg.get("trading_cost_pct", 0.001)),
        reward_mode=str(cfg.get("reward_mode", "diff_return")),
        reward_scaling=float(cfg.get("reward_scaling", 1.0)),
        initial_cash=float(cfg.get("initial_cash", 1_000_000.0)),
        max_position=float(cfg.get("max_position", 1.0)),
        obs_include_cash=bool(cfg.get("obs_include_cash", True)),
        obs_include_position=bool(cfg.get("obs_include_position", True)),
        obs_include_time=bool(cfg.get("obs_include_time", True)),
    )

    if not isinstance(env_cfg, TradingEnvConfig):
        raise TypeError(f"env_cfg must be TradingEnvConfig, got {type(env_cfg)}")

    if not isinstance(window_cfg, WindowedEnvConfig):
        raise TypeError(f"window_cfg must be WindowedEnvConfig, got {type(window_cfg)}")

    def train_fn():
        return WindowedTradingEnv(
            prices=train_prices,
            features=train_features,
            env_config=env_cfg,
            window_cfg=window_cfg,
        )

    def eval_fn():
        return WindowedTradingEnv(
            prices=eval_prices,
            features=eval_features,
            env_config=env_cfg,
            window_cfg=WindowedEnvConfig(
                window_size=window_size,
                random_start=False,  # deterministic eval
            ),
        )

    train_env = DummyVecEnv([train_fn])
    eval_env = DummyVecEnv([eval_fn])
    return train_env, eval_env


def _make_vanilla_env(
    train_prices: np.ndarray,
    train_features: np.ndarray,
    eval_prices: np.ndarray,
    eval_features: np.ndarray,
    cfg: dict,
):
    env_cfg = cfg.get("env_cfg") or TradingEnvConfig(
        trading_cost_pct=float(cfg.get("trading_cost_pct", 0.001)),
        reward_mode=str(cfg.get("reward_mode", "diff_return")),
        reward_scaling=float(cfg.get("reward_scaling", 1.0)),
        initial_cash=float(cfg.get("initial_cash", 1_000_000.0)),
        max_position=float(cfg.get("max_position", 1.0)),
        obs_include_cash=bool(cfg.get("obs_include_cash", True)),
        obs_include_position=bool(cfg.get("obs_include_position", True)),
        obs_include_time=bool(cfg.get("obs_include_time", True)),
    )

    def build_train():
        return TradingEnv(prices=train_prices, features=train_features, config=env_cfg)

    def build_eval():
        return TradingEnv(prices=eval_prices, features=eval_features, config=env_cfg)

    train_env = DummyVecEnv([build_train])
    eval_env = DummyVecEnv([build_eval])
    return train_env, eval_env


ENV_REGISTRY: Dict[str, EnvBuilder] = {
    "windowed": EnvBuilder("windowed", _make_windowed_env),
    "vanilla": EnvBuilder("vanilla", _make_vanilla_env),
}


def get_env_builder(name: str) -> EnvBuilder:
    key = name.lower()
    if key not in ENV_REGISTRY:
        raise KeyError(f"Unknown env '{name}'. Available: {list(ENV_REGISTRY)}")
    return ENV_REGISTRY[key]


# -----------------------------
# Helpers
# -----------------------------


def maybe_wrap_vecnormalize(
    env: VecEnv,
    enable: bool,
    stats_path: str | None = None,
    training: bool = True,
    norm_obs: bool = True,
    norm_reward: bool = True,
    clip_obs: float = 10.0,
):
    """
    Create or load VecNormalize wrapper.
    """
    if not enable:
        return env

    if stats_path:
        try:
            loaded = VecNormalize.load(stats_path, env)
            loaded.training = training
            return loaded
        except FileNotFoundError:
            pass

    vn = VecNormalize(
        env, norm_obs=norm_obs, norm_reward=norm_reward, clip_obs=clip_obs
    )
    vn.training = training
    return vn
