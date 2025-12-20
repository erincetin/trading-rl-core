from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


DEFAULT_HYPERPARAMS_PATH = Path(__file__).with_name("sb3_finance_hyperparams.yaml")


def load_hyperparams(path: str | Path | None = None) -> Dict[str, Any]:
    hp_path = Path(path) if path is not None else DEFAULT_HYPERPARAMS_PATH
    with hp_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Hyperparams file must be a mapping at top-level: {hp_path}")
    return data


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


def vecnormalize_cfg(hp: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(hp.get("vecnormalize", {}) or {})
