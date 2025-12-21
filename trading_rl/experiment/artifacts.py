from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
from stable_baselines3.common.vec_env import VecEnv

import wandb
from trading_rl.experiment.config import ExperimentConfig


def write_dataset_manifest(
    df: pd.DataFrame,
    run_dir: Path,
    exp: ExperimentConfig,
    split: str,
) -> Path:
    path = run_dir / f"dataset_manifest_{split}.json"
    manifest = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "symbol": exp.symbol,
        "timeframe": exp.timeframe,
        "split": split,
        "regime": exp.regime_name,
        "algo": exp.algo,
        "env": exp.env_name,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def save_checkpoint(
    run_dir: Path,
    model,
    vec_env: VecEnv,
    exp: ExperimentConfig,
) -> Tuple[Path, Optional[Path], Path]:
    run_dir.mkdir(parents=True, exist_ok=True)

    model_base = run_dir / "model"
    model.save(model_base)
    model_path = model_base.with_suffix(".zip")

    vecnorm_path: Optional[Path] = None
    if hasattr(vec_env, "save"):
        vecnorm_path = run_dir / "vecnormalize.pkl"
        vec_env.save(vecnorm_path)

    config_path = run_dir / "experiment_config.json"
    config_path.write_text(json.dumps(exp.to_dict(), indent=2), encoding="utf-8")

    return model_path, vecnorm_path, config_path


def log_wandb_artifact(
    run,
    exp: ExperimentConfig,
    model_path: Path,
    vecnorm_path: Optional[Path],
    manifests: Iterable[Path],
    config_path: Path,
) -> None:
    artifact = wandb.Artifact(
        name=exp.name,
        type="model",
        metadata={
            "algo": exp.algo,
            "env": exp.env_name,
            "regime": exp.regime_name,
            "seed": int(exp.seed),
        },
    )

    if model_path.exists():
        artifact.add_file(model_path)
    if vecnorm_path and vecnorm_path.exists():
        artifact.add_file(vecnorm_path)
    for m in manifests:
        if m.exists():
            artifact.add_file(m)
    if config_path.exists():
        artifact.add_file(config_path)

    run.log_artifact(artifact)
