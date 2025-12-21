from types import SimpleNamespace

import numpy as np
import pytest

from scripts.runner import (
    _apply_run_cfg,
    _resolve_run_cfg,
    expand_matrix,
    parse_args,
    parse_list,
)
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


def test_resolve_run_cfg_merges_algo_override():
    hp = {
        "run": {"total_timesteps": 1000, "eval_freq": 10},
        "sac": {"run": {"total_timesteps": 2000}},
    }
    cfg = _resolve_run_cfg(hp, "sac")
    assert cfg["total_timesteps"] == 2000
    assert cfg["eval_freq"] == 10


def test_apply_run_cfg_per_algo_overrides_values():
    args = SimpleNamespace(
        total_timesteps=1000,
        eval_freq=10,
        eval_episodes=1,
        normalize=None,
        vecnorm_path=None,
        resume=None,
        checkpoint=None,
        wandb_log_freq=1000,
        sb3_log_interval=None,
        output_dir="models",
    )
    _apply_run_cfg(args, {"total_timesteps": 2000}, per_algo=True)
    assert args.total_timesteps == 2000


def test_parse_args_requires_envs(monkeypatch, tmp_path):
    cfg = "run:\n  seeds: [1]\n"
    path = tmp_path / "hp.yaml"
    path.write_text(cfg, encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["runner.py", "--algos", "ppo", "--hyperparams", str(path)],
    )

    with pytest.raises(ValueError, match="envs not set"):
        parse_args()


def test_parse_args_requires_seeds_or_repeats(monkeypatch, tmp_path):
    cfg = "run:\n  envs: [windowed]\n"
    path = tmp_path / "hp.yaml"
    path.write_text(cfg, encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["runner.py", "--algos", "ppo", "--hyperparams", str(path)],
    )

    with pytest.raises(ValueError, match="seeds not set"):
        parse_args()
