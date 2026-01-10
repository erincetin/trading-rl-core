# trading_rl/experiment/orchestrator.py
from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np

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


def _plot_performance_summary(run_dir: Path, eval_cb: WandbEvalCallback) -> None:
    data = eval_cb.get_performance_curves()
    last_curve = data.get("last_curve")
    best_curve = data.get("best_curve")
    bh_curve = data.get("buy_and_hold_curve")
    if last_curve is None and best_curve is None:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "performance_summary.png"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)

    def _plot_curve(curve, label, color):
        if curve is None or len(curve) == 0:
            return
        x = np.arange(len(curve))
        ax.plot(x, curve, label=label, linewidth=2, color=color)

    def _ret_pct(curve):
        if curve is None or len(curve) < 2:
            return 0.0
        return 100.0 * float(curve[-1] / max(curve[0], 1e-12) - 1.0)

    last_label = None
    if last_curve is not None:
        last_label = f"final eval ({_ret_pct(last_curve):.2f}%)"
    best_label = None
    if best_curve is not None:
        best_label = f"best eval ({_ret_pct(best_curve):.2f}%)"
    bh_label = None
    if bh_curve is not None:
        bh_label = f"buy & hold ({_ret_pct(bh_curve):.2f}%)"

    _plot_curve(last_curve, last_label, color="#1f77b4")
    _plot_curve(best_curve, best_label, color="#2ca02c")
    _plot_curve(bh_curve, bh_label, color="#7f7f7f")

    ax.set_title("Final vs Best Eval Performance (vs Buy & Hold)")
    ax.set_xlabel("step")
    ax.set_ylabel("portfolio value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    try:
        wandb.log({"eval/performance_summary": wandb.Image(str(out_path))})
    except Exception:
        pass


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

    if exp.normalize is None:
        normalize = bool(exp.vecnormalize_params.get("enable", False))
    else:
        normalize = bool(exp.normalize)

    # -------------------------
    # W&B
    # -------------------------

    wandb_kwargs = {
        "project": exp.project,
        "entity": exp.entity,
        "group": exp.group,
        "name": exp.run_name or exp.name,
        "config": exp.to_dict(),
        "sync_tensorboard": True,
        "save_code": True,
    }
    if exp.wandb_sync_on_end:
        wandb_kwargs["mode"] = "offline"

    run = wandb.init(**wandb_kwargs)

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
    eval_cb = WandbEvalCallback(
        eval_env,
        eval_freq=exp.eval_freq,
        n_eval_episodes=exp.eval_episodes,
        log_eval_curves=exp.wandb_log_eval_curves,
        log_baseline_curves=exp.wandb_log_baseline_curves,
        log_action_hist=exp.wandb_log_action_hist,
        log_debug=exp.wandb_log_debug,
        wandb_curve_max_points=exp.wandb_curve_max_points,
        wandb_action_hist_freq=exp.wandb_action_hist_freq,
        wandb_hist_max_points=exp.wandb_hist_max_points,
    )
    callbacks = CallbackList(
        [
            WandbCallback(log_freq=exp.wandb_log_freq, verbose=0),
            eval_cb,
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

    eval_cb.run_final_eval()

    # -------------------------
    # Artifacts
    # -------------------------
    run_dir = Path(exp.output_dir) / run.id

    _plot_performance_summary(run_dir, eval_cb)

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
    if exp.wandb_sync_on_end:
        try:
            run_root = str(Path(run.dir).parent)
            cmd = ["wandb", "sync", run_root, "--project", exp.project]
            if exp.entity:
                cmd.extend(["--entity", exp.entity])
            subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            pass
