import numpy as np
import pytest

from trading_rl.baselines.baselines import compute_buy_and_hold, compute_sma_crossover


def test_buy_and_hold_normalizes_to_first_price():
    prices = np.array([10.0, 12.0, 11.0], dtype=float)
    curve = compute_buy_and_hold(prices)
    assert curve[0] == 1.0
    assert abs(curve[1] - 1.2) < 1e-6


def test_sma_crossover_returns_curve_length_matches_prices():
    prices = np.linspace(1, 5, 5)
    curve = compute_sma_crossover(prices, fast=2, slow=3, cost=0.0)
    assert len(curve) == len(prices)
    assert curve[0] == 1.0


def test_baselines_reject_non_positive_prices():
    prices = np.array([1.0, 0.0, 2.0])
    with pytest.raises(ValueError):
        compute_buy_and_hold(prices)
    with pytest.raises(ValueError):
        compute_sma_crossover(prices)
