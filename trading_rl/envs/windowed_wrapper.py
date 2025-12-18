# trading_rl/envs/windowed_wrapper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from .trading_env import TradingEnv, TradingEnvConfig


@dataclass
class WindowedEnvConfig:
    window_size: int
    random_start: bool = True


class WindowedTradingEnv(gym.Env):
    """
    Wraps TradingEnv to train on random contiguous windows of the full series.

    - At reset, picks a start index `s`
    - Underlying env sees prices[s : s+window_size+1], features[s : s+window_size+1]
      so the agent can take exactly `window_size` actions (needs next price for reward).
    - Evaluation can still be done on a non-windowed TradingEnv
    """

    metadata = TradingEnv.metadata

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        env_config: Optional[TradingEnvConfig] = None,
        window_cfg: Optional[WindowedEnvConfig] = None,
    ):
        super().__init__()
        if prices.ndim != 1 or features.ndim != 2:
            raise ValueError(
                "full_prices shape (T,), full_features shape (T, F) required"
            )
        if len(prices) != len(features):
            raise ValueError("full_prices and full_features length mismatch")

        self.prices = prices
        self.features = features
        self.T = len(prices)

        self.env_config = env_config or TradingEnvConfig()
        self.window_config = window_cfg or WindowedEnvConfig(
            window_size=self.T, random_start=False
        )

        if self.window_config.window_size < 1:
            raise ValueError("window_size must be >= 1")
        # We need window_size+1 prices to run window_size steps.
        if self.window_config.window_size + 1 > self.T:
            raise ValueError(
                "window_size too large for series length; require window_size+1 <= T"
            )

        # Placeholder; will be created on reset
        self._env: Optional[TradingEnv] = None
        self._start_idx: int = 0
        self._end_idx: int = 0

        # Expose spaces from a dummy env
        dummy_env = TradingEnv(
            prices=self.prices[: self.window_config.window_size + 1],
            features=self.features[: self.window_config.window_size + 1],
            config=self.env_config,
        )
        self.action_space = dummy_env.action_space
        self.observation_space = dummy_env.observation_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Choose window start
        if self.window_config.random_start:
            # Need window_size+1 points to run window_size steps
            max_start = self.T - (self.window_config.window_size + 1)
            if max_start < 0:
                raise ValueError("window_size larger than data length")
            self._start_idx = int(self.np_random.integers(0, max_start + 1))
        else:
            self._start_idx = 0

        self._end_idx = self._start_idx + (self.window_config.window_size + 1)

        prices_slice = self.prices[self._start_idx : self._end_idx]
        features_slice = self.features[self._start_idx : self._end_idx]

        self._env = TradingEnv(
            prices=prices_slice,
            features=features_slice,
            config=self.env_config,
        )

        obs, info = self._env.reset(seed=seed, options=options)
        # Optionally augment info with global indices
        info["window_start"] = self._start_idx
        info["window_end"] = self._end_idx
        return obs, info

    def step(self, action):
        if self._env is None:
            raise RuntimeError("env.step() called before reset()")

        obs, reward, terminated, truncated, info = self._env.step(action)

        # TimeLimit-style horizon handling:
        # reaching the end of the window is a truncation, not a terminal event.
        if terminated and not truncated:
            terminated = False
            truncated = True
            info["TimeLimit.truncated"] = True

        # Map underlying timestep to global timestep
        local_t = info.get("t", 0)
        global_t = self._start_idx + local_t
        info["global_t"] = global_t
        info["window_start"] = self._start_idx
        info["window_end"] = self._end_idx

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._env is not None:
            self._env.render()
