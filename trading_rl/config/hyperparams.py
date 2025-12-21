from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

DEFAULT_HYPERPARAMS_PATH = Path(__file__).with_name("sb3_finance_hyperparams.yaml")


def _normalize_hyperparams(hp: dict) -> dict:
    if not isinstance(hp, dict):
        return {}

    experiments = hp.get("experiments")
    if experiments is None:
        return hp
    if not isinstance(experiments, dict):
        raise ValueError("hyperparams 'experiments' must be a mapping")

    out = {k: v for k, v in hp.items() if k != "experiments"}

    for name, cfg in experiments.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"experiment '{name}' must be a mapping")
        algo_cfg = dict(cfg.get("algo", {}) or {})

        env_cfg = cfg.get("env", None)
        if env_cfg is not None:
            if not isinstance(env_cfg, dict):
                raise ValueError(f"experiment '{name}'.env must be a mapping")
            algo_cfg["env"] = env_cfg

        vec_cfg = cfg.get("vecnormalize", None)
        if vec_cfg is not None:
            if not isinstance(vec_cfg, dict):
                raise ValueError(f"experiment '{name}'.vecnormalize must be a mapping")
            algo_cfg["vecnormalize"] = vec_cfg

        run_cfg = cfg.get("run", None)
        if run_cfg is not None:
            if not isinstance(run_cfg, dict):
                raise ValueError(f"experiment '{name}'.run must be a mapping")
            algo_cfg["run"] = run_cfg

        out[name] = algo_cfg

    return out


def load_hyperparams(path: str) -> dict:
    hp_path = Path(path)
    if not hp_path.exists():
        raise FileNotFoundError(f"Hyperparams file not found: {hp_path.resolve()}")
    with hp_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _normalize_hyperparams(data)


def _resolve_activation_fn(name: Any):
    if name is None:
        return None
    if not isinstance(name, str):
        return name

    import torch.nn as nn

    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leakyrelu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "selu": nn.SELU,
    }
    key = name.strip().lower()
    if key in mapping:
        return mapping[key]
    if hasattr(nn, name):
        return getattr(nn, name)
    raise ValueError(f"Unknown activation_fn '{name}'. Supported: {sorted(mapping)}")


def resolve_policy_kwargs(policy_kwargs: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not policy_kwargs:
        return {}
    out = dict(policy_kwargs)
    if "activation_fn" in out:
        out["activation_fn"] = _resolve_activation_fn(out["activation_fn"])
    return out


def merged_algo_params(hp: Mapping[str, Any], algo: str, seed: int) -> Dict[str, Any]:
    shared = dict(hp.get("shared", {}) or {})
    algo_cfg = dict(hp.get(algo, {}) or {})

    # --- strip non-SB3 blocks (these must NOT go into PPO/A2C/SAC/TD3 kwargs) ---
    # per-algo env override lives under "<algo>: env: ..."
    algo_cfg.pop("env", None)
    # defensive: if you ever add per-algo vecnormalize overrides
    algo_cfg.pop("vecnormalize", None)
    # run config belongs to runner, not SB3
    algo_cfg.pop("run", None)
    # --------------------------------------------------------------------------

    if "policy_kwargs" in shared:
        shared["policy_kwargs"] = resolve_policy_kwargs(shared.get("policy_kwargs"))
    if "policy_kwargs" in algo_cfg:
        algo_cfg["policy_kwargs"] = resolve_policy_kwargs(algo_cfg.get("policy_kwargs"))

    merged = {**shared, **algo_cfg}
    merged["seed"] = int(seed)
    return merged


def _merge_dicts(base: dict, override: dict) -> dict:
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)  # recursive merge (nice for nested)
        else:
            out[k] = v
    return out


def env_cfg(hp: dict, algo: str | None = None) -> dict:
    base = hp.get("env", {}) or {}
    if algo is None:
        return dict(base)

    algo_block = hp.get(algo.lower(), {}) or {}
    override = algo_block.get("env", {}) or {}
    return _merge_dicts(base, override)


def vecnormalize_cfg(hp: dict, algo: str | None = None) -> dict:
    base = dict(hp.get("vecnormalize", {}) or {})
    if algo is None:
        cfg = base
    else:
        algo_block = hp.get(algo.lower(), {}) or {}
        override = algo_block.get("vecnormalize", {}) or {}
        cfg = _merge_dicts(base, override)

    cfg.pop("enable", None)  # <-- CRITICAL
    return cfg
