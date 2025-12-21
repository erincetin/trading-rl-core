from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict

from trading_rl.config.hyperparams import (
    env_cfg,
    merged_algo_params,
    vecnormalize_cfg,
)
from trading_rl.experiment.serialization import make_json_safe


@dataclass(frozen=True)
class ExperimentConfig:
    # identity
    name: str
    algo: str
    env_name: str
    seed: int
    regime_name: str

    # data
    symbol: str
    timeframe: str
    start: str
    end: str
    eval_start: str
    eval_end: str
    warmup_days: int

    # training
    total_timesteps: int
    eval_freq: int
    eval_episodes: int

    # algorithm / env config
    algo_params: Dict[str, Any]
    env_params: Dict[str, Any]
    vecnormalize_params: Dict[str, Any]

    # logging
    project: str
    group: str | None
    run_name: str | None

    # infra / runtime  ✅ NEW
    normalize: bool | None = None
    vecnorm_path: str | None = None
    wandb_log_freq: int = 1000
    tensorboard_root: str = "runs"
    resume: bool = False
    checkpoint: str | None = None
    sb3_log_interval: int | None = None
    output_dir: str = "models"

    def sb3_params(self) -> Dict[str, Any]:
        """Params passed to SB3 constructor."""
        return deepcopy(self.algo_params)

    def env_cfg(self) -> Dict[str, Any]:
        """Params passed to env builder."""
        return deepcopy(self.env_params)

    def vecnorm_cfg(self) -> Dict[str, Any]:
        return deepcopy(self.vecnormalize_params)

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        return make_json_safe(raw)


def build_experiment_config(
    *,
    args,
    hyperparams: dict,
    regime: dict,
    algo: str,
    env_name: str,
    seed: int,
) -> ExperimentConfig:
    # resolve params
    algo_params = merged_algo_params(hyperparams, algo, seed)
    env_params = env_cfg(hyperparams, algo)
    vecnorm_params = vecnormalize_cfg(hyperparams, algo)
    vecnorm_enabled = bool((hyperparams.get("vecnormalize") or {}).get("enable", False))
    algo_vecnorm = (
        (hyperparams.get(algo.lower(), {}) or {}).get("vecnormalize", {}) or {}
    )
    if "enable" in algo_vecnorm:
        vecnorm_enabled = bool(algo_vecnorm.get("enable"))
    normalize = args.normalize if args.normalize is not None else vecnorm_enabled

    regime_name = regime["name"]

    exp_name = f"{algo}-{env_name}-{regime_name}-seed{seed}"

    group = args.group
    if group:
        group = f"{group}-{algo}-{env_name}-{regime_name}"
    else:
        group = f"{algo}-{env_name}-{regime_name}"

    return ExperimentConfig(
        # identity
        name=exp_name,
        algo=algo,
        env_name=env_name,
        seed=seed,
        regime_name=regime_name,
        # data
        symbol=regime.get("symbol", args.symbol),
        timeframe=regime.get("timeframe", args.timeframe),
        start=regime["start"],
        end=regime["end"],
        eval_start=regime["eval_start"],
        eval_end=regime["eval_end"],
        warmup_days=int(regime.get("warmup_days", args.warmup_days)),
        # training
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        # configs
        algo_params=algo_params,
        env_params=env_params,
        vecnormalize_params=vecnorm_params,
        # logging
        project=args.project,
        group=group,
        run_name=args.run_name,
        normalize=normalize,
        vecnorm_path=args.vecnorm_path,
        wandb_log_freq=args.wandb_log_freq,
        tensorboard_root="runs",
        resume=args.resume,
        checkpoint=args.checkpoint,
        sb3_log_interval=args.sb3_log_interval,
        output_dir=args.output_dir,
    )
