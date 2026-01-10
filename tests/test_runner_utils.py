from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.runner import (
    _apply_run_cfg,
    _resolve_run_cfg,
    _run_combo,
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


def test_expand_matrix_supports_timesteps():
    regimes = [{"name": "r0", "start": "2024-01-01", "end": "2024-01-02"}]
    combos = expand_matrix(
        regimes, ["ppo"], ["vanilla"], [0], timesteps=[100, 200]
    )
    assert len(combos) == 2
    assert combos[0]["total_timesteps"] == 100
    assert combos[1]["total_timesteps"] == 200


def test_run_combo_overrides_total_timesteps_after_run_cfg(monkeypatch):
    captured = {}

    args = SimpleNamespace(
        symbol="AAPL",
        timeframe="1Min",
        start="2024-01-01",
        end="2024-01-02",
        eval_start="2024-01-01",
        eval_end="2024-01-02",
        warmup_days=0,
        csv_path=None,
        api_key="X",
        api_secret="Y",
        cache_dir="data_cache",
        hyperparams_data={},
        normalize=None,
        vecnorm_path=None,
        resume=False,
        checkpoint=None,
        wandb_log_freq=1000,
        sb3_log_interval=None,
        output_dir="models",
        project="proj",
        group=None,
        run_name=None,
        seed=0,
    )

    def _fake_apply_regime(base_args, _regime):
        return base_args

    def _fake_resolve_run_cfg(_hp, _algo):
        return {"total_timesteps": [100, 200]}

    def _fake_build_experiment_config(*, args, **_kwargs):
        return SimpleNamespace(total_timesteps=args.total_timesteps)

    def _fake_train_once(*, exp, **_kwargs):
        captured["total_timesteps"] = exp.total_timesteps

    def _fake_load_market_data(**_kwargs):
        idx = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        return pd.DataFrame(
            {
                "open": [1.0, 1.0, 1.0],
                "high": [1.0, 1.0, 1.0],
                "low": [1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0],
                "volume": [1.0, 1.0, 1.0],
            },
            index=idx,
        )

    def _fake_split_train_eval(df, **_kwargs):
        return df, df

    def _fake_prepare_market_arrays(df):
        arr = df["close"].to_numpy(dtype=np.float32)
        feats = np.zeros((len(df), 1), dtype=np.float32)
        return SimpleNamespace(prices=arr, features=feats)

    monkeypatch.setattr("scripts.runner.apply_regime", _fake_apply_regime)
    monkeypatch.setattr("scripts.runner._resolve_run_cfg", _fake_resolve_run_cfg)
    monkeypatch.setattr(
        "scripts.runner.build_experiment_config", _fake_build_experiment_config
    )
    monkeypatch.setattr("scripts.runner.train_once", _fake_train_once)
    monkeypatch.setattr("scripts.runner.load_market_data", _fake_load_market_data)
    monkeypatch.setattr("scripts.runner.build_features_cached", lambda df, **_k: df)
    monkeypatch.setattr("scripts.runner.split_train_eval", _fake_split_train_eval)
    monkeypatch.setattr("scripts.runner.prepare_market_arrays", _fake_prepare_market_arrays)

    combo = {
        "regime": {"name": "r0", "start": "2024-01-01", "end": "2024-01-02"},
        "algo": "ppo",
        "env": "vanilla",
        "seed": 0,
        "total_timesteps": 500,
    }

    _run_combo(args, combo)
    assert captured["total_timesteps"] == 500


def test_registry_builds_env_and_algo():
    prices = np.linspace(1, 10, 20).astype(np.float32)
    feats = np.zeros((20, 3), dtype=np.float32)

    env_builder = get_env_builder("vanilla")
    train_env, eval_env = env_builder.factory(prices, feats, prices, feats, {})

    algo_builder = get_algo_builder("ppo")
    model = algo_builder.factory(train_env, {"verbose": 0, "device": "cpu"})

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


def test_total_timesteps_precedence_cli_over_per_algo_over_global():
    args = SimpleNamespace(total_timesteps=10, _total_timesteps_from_cli=True)
    _apply_run_cfg(args, {"total_timesteps": 20}, per_algo=False)
    _apply_run_cfg(args, {"total_timesteps": 30}, per_algo=True)
    assert args.total_timesteps == 10

    args = SimpleNamespace(total_timesteps=None, _total_timesteps_from_cli=False)
    _apply_run_cfg(args, {"total_timesteps": 20}, per_algo=False)
    _apply_run_cfg(args, {"total_timesteps": 30}, per_algo=True)
    assert args.total_timesteps == 30


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
