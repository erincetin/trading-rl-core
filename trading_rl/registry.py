# trading_rl\registry.py
"""
Registries for algorithms and environments.

These provide string-to-builder mappings so the runner can construct
algos/envs by name and stay extensible for custom additions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

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


def _ppo_factory(env: VecEnv, params: dict) -> PPO:
    policy = params.pop("policy", "MlpPolicy")
    return PPO(policy, env, **params)


def _a2c_factory(env: VecEnv, params: dict) -> A2C:
    policy = params.pop("policy", "MlpPolicy")
    return A2C(policy, env, **params)


def _sac_factory(env: VecEnv, params: dict) -> SAC:
    policy = params.pop("policy", "MlpPolicy")
    return SAC(policy, env, **params)


def _td3_factory(env: VecEnv, params: dict) -> TD3:
    policy = params.pop("policy", "MlpPolicy")

    exploration_sigma = float(params.pop("exploration_noise", 0.1))

    if "action_noise" not in params:
        n_actions = int(np.prod(env.action_space.shape))
        params["action_noise"] = NormalActionNoise(
            mean=np.zeros(n_actions, dtype=np.float32),
            sigma=exploration_sigma * np.ones(n_actions, dtype=np.float32),
        )

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
    window_size = min(requested_window, max(1, len(train_prices) - 1))

    # NEW: vectorization knobs
    n_envs_train = int(cfg.get("n_envs_train", 1))
    n_envs_eval = int(cfg.get("n_envs_eval", 1))
    vec_env_type = str(cfg.get("vec_env_type", "dummy"))

    window_cfg_train = cfg.get("window_cfg") or WindowedEnvConfig(
        window_size=window_size,
        random_start=bool(cfg.get("random_start", True)),
    )
    env_cfg = cfg.get("env_cfg") or TradingEnvConfig(
        trading_cost_pct=float(cfg.get("trading_cost_pct", 0.001)),
        reward_mode=str(cfg.get("reward_mode", "log_return")),
        reward_scaling=float(cfg.get("reward_scaling", 1.0)),
        initial_cash=float(cfg.get("initial_cash", 1_000_000.0)),
        max_position=float(cfg.get("max_position", 1.0)),
        obs_include_cash=bool(cfg.get("obs_include_cash", True)),
        obs_include_position=bool(cfg.get("obs_include_position", True)),
        obs_include_time=bool(cfg.get("obs_include_time", True)),
        obs_include_pnl=bool(cfg.get("obs_include_pnl", True)),
        obs_lookback=int(cfg.get("obs_lookback", 1)),
    )

    if not isinstance(env_cfg, TradingEnvConfig):
        raise TypeError(f"env_cfg must be TradingEnvConfig, got {type(env_cfg)}")
    if not isinstance(window_cfg_train, WindowedEnvConfig):
        raise TypeError(
            f"window_cfg must be WindowedEnvConfig, got {type(window_cfg_train)}"
        )

    def make_train_env():
        return WindowedTradingEnv(
            prices=train_prices,
            features=train_features,
            env_config=env_cfg,
            window_cfg=window_cfg_train,
        )

    # Eval on the full series (unwindowed) to measure end-to-end performance.
    def make_eval_env():
        return TradingEnv(prices=eval_prices, features=eval_features, config=env_cfg)

    train_fns = [lambda: make_train_env() for _ in range(n_envs_train)]
    eval_fns = [lambda: make_eval_env() for _ in range(n_envs_eval)]

    train_env = _build_vec_env(train_fns, vec_env_type)
    eval_env = _build_vec_env(eval_fns, vec_env_type if n_envs_eval > 1 else "dummy")

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
        reward_mode=str(cfg.get("reward_mode", "log_return")),
        reward_scaling=float(cfg.get("reward_scaling", 1.0)),
        initial_cash=float(cfg.get("initial_cash", 1_000_000.0)),
        max_position=float(cfg.get("max_position", 1.0)),
        obs_include_cash=bool(cfg.get("obs_include_cash", True)),
        obs_include_position=bool(cfg.get("obs_include_position", True)),
        obs_include_time=bool(cfg.get("obs_include_time", True)),
        obs_include_pnl=bool(cfg.get("obs_include_pnl", True)),
        obs_lookback=int(cfg.get("obs_lookback", 1)),
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


def _build_vec_env(env_fns, vec_env_type: str):
    vec_env_type = (vec_env_type or "dummy").lower()
    if vec_env_type == "dummy":
        return DummyVecEnv(env_fns)

    if vec_env_type == "subproc":
        # On Windows, SubprocVecEnv uses spawn; keep this only if needed.
        # It should work as long as you run from a __main__-guarded entrypoint (you do).
        return SubprocVecEnv(env_fns, start_method="spawn")

    raise ValueError(
        f"Unknown vec_env_type='{vec_env_type}'. Use 'dummy' or 'subproc'."
    )


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
