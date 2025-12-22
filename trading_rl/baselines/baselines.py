import numpy as np
import pandas as pd


def _validate_prices(prices) -> np.ndarray:
    arr = np.asarray(prices, dtype=float)
    if arr.size == 0:
        raise ValueError("prices must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError("prices must be finite")
    if (arr <= 0).any():
        raise ValueError("prices must be > 0")
    return arr


def compute_buy_and_hold(prices, cost: float = 0.0, include_exit_cost: bool = True):
    prices = _validate_prices(prices)
    cost = float(cost)
    if cost < 0:
        raise ValueError("cost must be >= 0")
    pv = prices / prices[0]
    pv = pv * (1.0 - cost)  # enter once
    if include_exit_cost and pv.size > 0:
        pv = pv.copy()
        pv[-1] = pv[-1] * (1.0 - cost)
    return pv.tolist()


def compute_sma_crossover(prices, fast=20, slow=50, cost=0.001):
    """
    Long-only SMA crossover strategy:
       - long when SMA_fast > SMA_slow
       - flat otherwise
    Includes trading costs.
    """
    prices = pd.Series(_validate_prices(prices))

    sma_fast = prices.rolling(fast).mean().shift(1)
    sma_slow = prices.rolling(slow).mean().shift(1)

    # Generate raw signals
    long_signal = (sma_fast > sma_slow).astype(float)

    pv = [1.0]
    prev_pos = 0.0

    for i in range(1, len(prices)):
        # return from holding the asset
        ret = (prices[i] / prices[i - 1]) - 1

        # cost when position changes
        trade_cost = cost * abs(long_signal[i] - prev_pos)

        # update PV
        pv.append(pv[-1] * (1 + long_signal[i] * ret - trade_cost))

        prev_pos = long_signal[i]

    return pv
