import numpy as np

from scripts.runner import expand_matrix, parse_list
from trading_rl.registry import get_algo_builder, get_env_builder, maybe_wrap_vecnormalize


def test_expand_matrix_cartesian():
    regimes = [{"name": "r0", "start": "2024-01-01", "end": "2024-01-02"}]
    combos = expand_matrix(regimes, ["ppo", "a2c"], ["vanilla", "windowed"], [0, 1])
    assert len(combos) == 2 * 2 * 2
    assert combos[0]["algo"] == "ppo"
    assert combos[0]["env"] == "vanilla"
    assert combos[0]["seed"] == 0
    assert combos[0]["regime"]["name"] == "r0"


def test_registry_builds_env_and_algo():
    prices = np.linspace(1, 10, 20).astype(np.float32)
    feats = np.zeros((20, 3), dtype=np.float32)

    env_builder = get_env_builder("vanilla")
    train_env, eval_env = env_builder.factory(prices, feats, prices, feats, {})

    algo_builder = get_algo_builder("ppo")
    model = algo_builder.factory(train_env, {"verbose": 0})

    assert model is not None
    assert train_env.observation_space.shape is not None

    wrapped = maybe_wrap_vecnormalize(train_env, enable=True, training=True)
    assert wrapped is not None


def test_parse_list_handles_string_and_iterable():
    assert parse_list("ppo,a2c") == ["ppo", "a2c"]
    assert parse_list(["sac", "td3"]) == ["sac", "td3"]
