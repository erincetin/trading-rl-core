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


def compute_buy_and_hold(prices):
    prices = _validate_prices(prices)
    pv = prices / prices[0]
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
