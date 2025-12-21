import pytest

from trading_rl.config.hyperparams import (
    env_cfg,
    load_hyperparams,
    merged_algo_params,
    resolve_policy_kwargs,
    vecnormalize_cfg,
)


def test_resolve_policy_kwargs_activation_fn():
    torch = pytest.importorskip("torch")

    out = resolve_policy_kwargs({"activation_fn": "relu", "net_arch": [64, 64]})
    assert out["activation_fn"] is torch.nn.ReLU
    assert out["net_arch"] == [64, 64]


def test_merged_algo_params_strips_non_sb3_blocks():
    hp = {
        "shared": {"learning_rate": 0.001},
        "ppo": {"env": {"window_size": 10}, "vecnormalize": {"enable": True}},
    }
    params = merged_algo_params(hp, "ppo", seed=123)
    assert params["learning_rate"] == 0.001
    assert params["seed"] == 123
    assert "env" not in params
    assert "vecnormalize" not in params


def test_env_cfg_merges_algo_override():
    hp = {
        "env": {"trading_cost_pct": 0.001, "nested": {"a": 1, "b": 2}},
        "ppo": {"env": {"nested": {"b": 3}}},
    }
    cfg = env_cfg(hp, "ppo")
    assert cfg["trading_cost_pct"] == 0.001
    assert cfg["nested"]["a"] == 1
    assert cfg["nested"]["b"] == 3


def test_vecnormalize_cfg_strips_enable_flag():
    hp = {"vecnormalize": {"enable": True, "clip_obs": 5.0}}
    cfg = vecnormalize_cfg(hp)
    assert "enable" not in cfg
    assert cfg["clip_obs"] == 5.0


def test_load_hyperparams_experiments_schema(tmp_path):
    content = """
experiments:
  ppo:
    algo:
      learning_rate: 0.01
      verbose: 1
    env:
      window_size: 5
env:
  trading_cost_pct: 0.1
vecnormalize:
  enable: true
regimes:
  - name: r0
    start: "2024-01-01"
    end: "2024-01-02"
    eval_start: "2024-01-01"
    eval_end: "2024-01-02"
"""
    path = tmp_path / "hp.yaml"
    path.write_text(content, encoding="utf-8")

    hp = load_hyperparams(str(path))

    assert "experiments" not in hp
    assert hp["ppo"]["learning_rate"] == 0.01
    assert hp["ppo"]["env"]["window_size"] == 5
    assert hp["env"]["trading_cost_pct"] == 0.1
    assert hp["vecnormalize"]["enable"] is True
    assert hp["regimes"][0]["name"] == "r0"


def test_load_hyperparams_invalid_experiments_type(tmp_path):
    content = "experiments: []"
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_hyperparams(str(path))


def test_load_hyperparams_invalid_experiment_entry(tmp_path):
    content = """
experiments:
  ppo: []
"""
    path = tmp_path / "bad_entry.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_hyperparams(str(path))


def test_experiments_schema_flows_into_env_and_algo_params(tmp_path):
    content = """
experiments:
  ppo:
    algo:
      learning_rate: 0.01
    env:
      window_size: 7
shared:
  gamma: 0.99
env:
  trading_cost_pct: 0.1
"""
    path = tmp_path / "hp.yaml"
    path.write_text(content, encoding="utf-8")

    hp = load_hyperparams(str(path))

    cfg = env_cfg(hp, "ppo")
    assert cfg["trading_cost_pct"] == 0.1
    assert cfg["window_size"] == 7

    params = merged_algo_params(hp, "ppo", seed=5)
    assert params["learning_rate"] == 0.01
    assert params["gamma"] == 0.99
    assert params["seed"] == 5
    assert "env" not in params
