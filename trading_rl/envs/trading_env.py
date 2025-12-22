# trading_rl/envs/trading_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class TradingEnvConfig:
    initial_cash: float = 1_000_000.0
    trading_cost_pct: float = 0.001  # 0.1%
    max_position: float = 1.0  # long-only, max 100% of portfolio in asset
    reward_scaling: float = 1.0  # optionally scale rewards
    obs_include_cash: bool = True
    obs_include_position: bool = True
    obs_include_time: bool = True
    obs_include_pnl: bool = True
    reward_mode: str = "log_return"


class TradingEnv(gym.Env):
    """
    Single-asset, long-only, continuous action trading environment.

    - prices: shape (T,)
    - features: shape (T, F)
    - action: target weight in [0, 1] of portfolio in the asset
    - reward: fractional portfolio change between t and t+1
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        config: Optional[TradingEnvConfig] = None,
    ) -> None:
        super().__init__()

        if prices.ndim != 1:
            raise ValueError(f"prices must be shape (T,), got {prices.shape}")
        if features.ndim != 2:
            raise ValueError(f"features must be shape (T, F), got {features.shape}")
        if len(prices) != len(features):
            raise ValueError("prices and features must have same length")
        if not np.isfinite(prices).all():
            raise ValueError("prices must be finite")
        if not np.isfinite(features).all():
            raise ValueError("features must be finite")
        if (prices <= 0).any():
            raise ValueError("prices must be > 0")

        self.prices = prices.astype("float32")
        self.features = features.astype("float32")
        self.T, self.F = self.features.shape
        self.config = config or TradingEnvConfig()

        # Action: target allocation to asset in [0, max_position] (long-only)
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([self.config.max_position], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: features + (optional) [cash_frac, position_frac, time_frac]
        obs_dim = self.F
        if self.config.obs_include_cash:
            obs_dim += 1
        if self.config.obs_include_position:
            obs_dim += 1
        if self.config.obs_include_time:
            obs_dim += 1
        if self.config.obs_include_pnl:
            obs_dim += 1

        # Reasonable bounds; features can be roughly standardized by user
        high = np.ones(obs_dim, dtype=np.float32) * 1e6
        low = -high

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Internal state
        self._t: int = 0
        self._cash: float = self.config.initial_cash
        self._position: float = 0.0  # number of shares
        self._avg_entry_price: float = 0.0
        self._portfolio_value: float = self.config.initial_cash

        # History for analysis
        self._history: Dict[str, list] = {
            "t": [],
            "price": [],
            "cash": [],
            "position": [],
            "portfolio_value": [],
            "action": [],
            "reward": [],
            "price_exec": [],
            "price_next": [],
            "trade_cost": [],
            "realized_weight": [],
        }

    # ---- Gymnasium API ----

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._t = 0
        self._cash = self.config.initial_cash
        self._position = 0.0
        self._avg_entry_price = 0.0
        self._portfolio_value = self.config.initial_cash

        # Clear history
        for k in self._history:
            self._history[k].clear()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        """
        One-step portfolio update:

        1. Compute current portfolio value at t.
        2. Interpret action as target weight in asset.
        3. Trade from current position to target position at price_t, paying cost.
        4. Move to t+1, revalue portfolio, compute reward.
        """
        # Clip action to valid range
        if isinstance(action, (list, tuple, np.ndarray)):
            a = float(
                np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
            )
        else:
            a = float(
                np.clip(action, self.action_space.low[0], self.action_space.high[0])
            )

        price_t = float(self.prices[self._t])

        # Portfolio value before trade (at price_t)
        prev_portfolio_value = self._cash + self._position * price_t

        # Target dollar exposure in asset
        target_asset_value = a * prev_portfolio_value

        current_asset_value = self._position * price_t
        trade_value = target_asset_value - current_asset_value

        executed_trade_value = 0.0
        if abs(trade_value) > 1e-8:
            if trade_value > 0:
                old_pos = self._position
                max_buy_value = min(trade_value, max(0.0, self._cash))
                trade_cost = abs(max_buy_value) * self.config.trading_cost_pct
                max_buy_value = min(max_buy_value, max(0.0, self._cash - trade_cost))
                executed_trade_value = max_buy_value
                delta_shares = max_buy_value / price_t
                self._position += delta_shares
                if self._position > 0.0:
                    if old_pos > 0.0 and self._avg_entry_price > 0.0:
                        self._avg_entry_price = (
                            self._avg_entry_price * old_pos + price_t * delta_shares
                        ) / self._position
                    else:
                        self._avg_entry_price = price_t
                self._cash -= max_buy_value + trade_cost
            else:
                old_pos = self._position
                max_sell_value = min(-trade_value, current_asset_value)
                trade_cost = abs(max_sell_value) * self.config.trading_cost_pct
                executed_trade_value = -max_sell_value
                delta_shares = max_sell_value / price_t
                self._position -= delta_shares
                if self._position <= 0.0:
                    self._position = 0.0
                    self._avg_entry_price = 0.0
                self._cash += max_sell_value - trade_cost
        else:
            trade_cost = 0.0

        # Move time forward
        terminated = False
        truncated = False
        if self._t >= self.T - 2:
            # After this step, we'll value at final price and terminate
            next_t = self.T - 1
            terminated = True
        else:
            next_t = self._t + 1

        self._t = next_t
        price_next = float(self.prices[self._t])

        # Portfolio value after price moves
        portfolio_value = self._cash + self._position * price_next
        self._portfolio_value = portfolio_value

        asset_value_next = self._position * price_next
        realized_weight_next = asset_value_next / max(self._portfolio_value, 1e-8)

        # Reward
        pv0 = max(prev_portfolio_value, 1e-8)
        pv1 = max(portfolio_value, 1e-8)

        mode = (self.config.reward_mode or "diff_return").lower()
        if mode in {"diff_return", "simple_return"}:
            reward = (pv1 - pv0) / pv0
        elif mode in {"log_return", "log"}:
            if pv0 <= 0 or pv1 <= 0:
                raise ValueError("portfolio_value must be > 0 for log_return")
            reward = float(np.log(pv1 / pv0))
        elif mode in {"excess_return", "excess"}:
            agent_ret = (pv1 - pv0) / pv0
            price_ret = (price_next / price_t) - 1.0
            reward = agent_ret - price_ret
        elif mode in {"pnl", "delta_pv"}:
            reward = pv1 - pv0
        else:
            raise ValueError(f"Unknown reward_mode: {self.config.reward_mode}")

        reward = float(reward) * float(self.config.reward_scaling)

        # Bookkeeping
        self._record_step(
            action=a,
            price_exec=price_t,
            price_next=price_next,
            reward=reward,
            trade_cost=trade_cost,
            realized_weight=realized_weight_next,
        )

        obs = self._get_obs()
        info = self._get_info()
        info.update(
            {
                "trade_cost": float(trade_cost),
                "trade_value_target": float(trade_value),
                "trade_value_executed": float(executed_trade_value),
                "action_target_weight": float(a),
                "realized_weight": float(realized_weight_next),
                "portfolio_value_prev": float(prev_portfolio_value),
                "portfolio_value_next": float(self._portfolio_value),
                "price_exec": float(price_t),
                "price_next": float(price_next),
            }
        )
        return obs, reward, terminated, truncated, info

    # ---- Helpers ----

    def _get_obs(self) -> np.ndarray:
        feats = self.features[self._t]  # shape (F,)
        obs_list = [feats]

        if self.config.obs_include_cash:
            cash_frac = self._cash / max(self._portfolio_value, 1e-8)
            obs_list.append(np.array([cash_frac], dtype=np.float32))

        if self.config.obs_include_position:
            price_t = float(self.prices[self._t])
            asset_value = self._position * price_t
            pos_frac = asset_value / max(self._portfolio_value, 1e-8)
            obs_list.append(np.array([pos_frac], dtype=np.float32))

        if self.config.obs_include_time:
            time_frac = self._t / max(self.T - 1, 1)
            obs_list.append(np.array([time_frac], dtype=np.float32))

        if self.config.obs_include_pnl:
            price_t = float(self.prices[self._t])
            unrealized = (price_t - self._avg_entry_price) * self._position
            pnl_frac = unrealized / max(self._portfolio_value, 1e-8)
            obs_list.append(np.array([pnl_frac], dtype=np.float32))

        return np.concatenate(obs_list).astype(np.float32)

    def _get_info(self) -> Dict:
        return {
            "t": self._t,
            "price": float(self.prices[self._t]),
            "cash": self._cash,
            "position": self._position,
            "portfolio_value": self._portfolio_value,
        }

    def _record_step(
        self,
        action: float,
        price_exec: float,
        price_next: float,
        reward: float,
        trade_cost: float,
        realized_weight: float,
    ) -> None:
        self._history["t"].append(self._t)
        self._history["price"].append(price_next)
        self._history["price_exec"].append(price_exec)
        self._history["price_next"].append(price_next)
        self._history["trade_cost"].append(trade_cost)
        self._history["realized_weight"].append(realized_weight)
        self._history["cash"].append(self._cash)
        self._history["position"].append(self._position)
        self._history["portfolio_value"].append(self._portfolio_value)
        self._history["action"].append(action)
        self._history["reward"].append(reward)

    # ---- Convenience getters ----

    @property
    def history(self) -> Dict[str, np.ndarray]:
        return {k: np.array(v) for k, v in self._history.items()}

    def render(self):
        # Minimal human-readable render, can be extended later
        print(
            f"t={self._t} "
            f"price={self.prices[self._t]:.2f} "
            f"cash={self._cash:.2f} "
            f"pos={self._position:.4f} "
            f"PV={self._portfolio_value:.2f}"
        )
