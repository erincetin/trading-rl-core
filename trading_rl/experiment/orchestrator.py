# trading_rl/experiment/orchestrator.py
from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecEnv

import wandb
from trading_rl.callbacks.eval_callback import WandbEvalCallback
from trading_rl.callbacks.wandb_callback import WandbCallback
from trading_rl.experiment.artifacts import (
    log_wandb_artifact,
    save_checkpoint,
    write_dataset_manifest,
)
from trading_rl.experiment.config import ExperimentConfig
from trading_rl.registry import (
    get_algo_builder,
    get_env_builder,
    maybe_wrap_vecnormalize,
)


def _small_hp_for_wandb(hp: dict, algo: str) -> dict:
    """
    Keep W&B config small and JSON-serializable.
    """
    return {
        "algo": hp.get(algo, {}) or {},
        "env": hp.get("env", {}) or {},
        "vecnormalize": hp.get("vecnormalize", {}) or {},
    }


def _try_sync_vecnormalize_stats(train_env: VecEnv, eval_env: VecEnv) -> None:
    """
    If both are VecNormalize, sync running stats from train -> eval.
    Works whether eval_env is VecNormalize or plain VecEnv.
    """
    try:
        from stable_baselines3.common.vec_env import VecNormalize as VN

        if isinstance(train_env, VN) and isinstance(eval_env, VN):
            eval_env.obs_rms = train_env.obs_rms
            eval_env.ret_rms = train_env.ret_rms
    except Exception:
        return


def _safe_seed_env(env: VecEnv, seed: int) -> None:
    try:
        env.seed(seed)
    except Exception:
        pass


def train_once(
    *,
    exp: ExperimentConfig,
    md_train,
    md_eval,
    df_train,
    df_eval,
) -> None:
    """
    Orchestrates one experiment run.
    """

    algo = exp.algo
    env_name = exp.env_name
    seed = exp.seed

    normalize = (
        exp.normalize
        if exp.normalize is not None
        else bool(exp.vecnormalize_params.get("enable", False))
    )

    # -------------------------
    # W&B
    # -------------------------

    run = wandb.init(
        project=exp.project,
        group=exp.group,
        name=exp.run_name or exp.name,
        config=exp.to_dict(),
        sync_tensorboard=True,
        save_code=True,
    )

    # -------------------------
    # Build envs
    # -------------------------
    algo_builder = get_algo_builder(algo)
    env_builder = get_env_builder(env_name)

    train_env, eval_env = env_builder.factory(
        md_train.prices,
        md_train.features,
        md_eval.prices,
        md_eval.features,
        exp.env_cfg(),
    )

    train_env = maybe_wrap_vecnormalize(
        train_env,
        enable=normalize,
        stats_path=exp.vecnorm_path,
        training=True,
        **exp.vecnorm_cfg(),
    )
    eval_env = maybe_wrap_vecnormalize(
        eval_env,
        enable=normalize,
        stats_path=exp.vecnorm_path,
        training=False,
        **exp.vecnorm_cfg(),
    )

    _try_sync_vecnormalize_stats(train_env, eval_env)
    _safe_seed_env(train_env, seed)
    _safe_seed_env(eval_env, seed)

    # -------------------------
    # Callbacks
    # -------------------------
    callbacks = CallbackList(
        [
            WandbCallback(log_freq=exp.wandb_log_freq, verbose=0),
            WandbEvalCallback(
                eval_env,
                eval_freq=exp.eval_freq,
                n_eval_episodes=exp.eval_episodes,
            ),
        ]
    )

    # -------------------------
    # Model
    # -------------------------
    model_params = exp.sb3_params()
    model_params["tensorboard_log"] = str(Path(exp.tensorboard_root) / run.id)

    if exp.resume and exp.checkpoint:
        model = algo_builder.algo_cls.load(exp.checkpoint, env=train_env)
        reset_steps = False
    else:
        model = algo_builder.factory(train_env, model_params)
        reset_steps = True

    learn_kwargs = dict(
        total_timesteps=exp.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_steps,
    )

    if algo in {"sac", "td3"}:
        learn_kwargs["log_interval"] = 1
    elif exp.sb3_log_interval is not None:
        learn_kwargs["log_interval"] = exp.sb3_log_interval

    model.learn(**learn_kwargs)

    # -------------------------
    # Artifacts
    # -------------------------
    run_dir = Path(exp.output_dir) / run.id

    model_path, vecnorm_path, config_path = save_checkpoint(
        run_dir=run_dir,
        model=model,
        vec_env=train_env,
        exp=exp,
    )

    manifest_train = write_dataset_manifest(df_train, run_dir, exp, "train")
    manifest_eval = write_dataset_manifest(df_eval, run_dir, exp, "eval")

    log_wandb_artifact(
        run=run,
        exp=exp,
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        manifests=[manifest_train, manifest_eval],
        config_path=config_path,
    )

    wandb.finish()
