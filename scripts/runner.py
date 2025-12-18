# scripts/runner.py
"""
Matrix runner for RL finance experiments.

Features:
- Expand algos × envs × seeds and run sequentially.
- W&B logging + artifacts (model, vecnormalize stats, config).
- Optional resume from an existing checkpoint (local path).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import wandb
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecEnv

from trading_rl.baselines.baselines import compute_buy_and_hold, compute_sma_crossover
from trading_rl.callbacks.eval_callback import WandbEvalCallback
from trading_rl.callbacks.wandb_callback import WandbCallback
from trading_rl.config.hyperparams import (
    env_cfg,
    load_hyperparams,
    merged_algo_params,
    vecnormalize_cfg,
)
from trading_rl.data.alpaca_loader import AlpacaConfig, AlpacaDataLoader
from trading_rl.data.loader import load_ohlcv_csv, prepare_market_arrays
from trading_rl.registry import (
    get_algo_builder,
    get_env_builder,
    maybe_wrap_vecnormalize,
)
from trading_rl.data.indicators import add_talib_indicators


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def parse_list(arg: str | Iterable[str]) -> List[str]:
    if isinstance(arg, str):
        return [x for x in arg.split(",") if x]
    return list(arg)


def expand_matrix(algos: Sequence[str], envs: Sequence[str], seeds: Sequence[int]):
    combos = []
    for algo in algos:
        for env in envs:
            for seed in seeds:
                combos.append({"algo": algo, "env": env, "seed": int(seed)})
    return combos


def load_market_data(args) -> pd.DataFrame:
    if args.csv_path:
        df = load_ohlcv_csv(args.csv_path)
    else:
        cfg = AlpacaConfig(
            api_key=args.api_key,
            api_secret=args.api_secret,
            cache_dir=args.cache_dir,
        )

        start_ts = pd.Timestamp(args.start, tz="UTC")
        warmup_start = (start_ts - pd.Timedelta(days=int(args.warmup_days))).strftime(
            "%Y-%m-%d"
        )

        loader = AlpacaDataLoader(cfg)
        df = loader.load(
            symbol=args.symbol,
            start=warmup_start,
            end=args.end,
            timeframe=args.timeframe,
            use_cache=True,
        )

    df = df.ffill().dropna()
    return df


def _ts_like_index(df: pd.DataFrame, s: str | None) -> pd.Timestamp | None:
    if s is None:
        return None
    ts = pd.Timestamp(s)
    tz = getattr(df.index, "tz", None)

    # If index is tz-aware, localize/convert ts to match it
    if tz is not None:
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
    else:
        # If index is tz-naive, drop tz from ts if it has one
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None).tz_localize(None)

    return ts


def split_train_eval(df: pd.DataFrame, args):
    if args.eval_start is None and args.eval_end is None:
        return df, df

    eval_start = _ts_like_index(df, args.eval_start) or df.index.min()
    eval_end = _ts_like_index(df, args.eval_end) or df.index.max()

    if args.eval_start and _ts_like_index(df, args.eval_start) > df.index.max():
        raise ValueError(
            f"eval_start ({args.eval_start}) is after data max ({df.index.max()}). "
            "Increase --end to include eval period."
        )

    eval_df = df.loc[eval_start:eval_end]

    train_end = eval_start - pd.Timedelta(seconds=1)
    train_df = df.loc[:train_end]

    if len(eval_df) == 0:
        raise ValueError("Evaluation slice is empty; adjust eval_start/eval_end.")
    if len(train_df) == 0:
        raise ValueError("Training slice is empty; adjust eval_start/eval_end.")

    return train_df, eval_df


def _write_manifest(df: pd.DataFrame, path: Path, args, split: str) -> Path:
    manifest = {
        "rows": len(df),
        "columns": list(df.columns),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "source": args.csv_path or "alpaca",
        "split": split,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def _save_checkpoint(run_dir: Path, model, vec_env: VecEnv, meta: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    base_path = run_dir / "model"
    model.save(base_path)
    model_path = base_path.with_suffix(".zip")

    stats_path = None
    if hasattr(vec_env, "save"):
        stats_path = run_dir / "vecnormalize.pkl"
        vec_env.save(stats_path)

    cfg_path = run_dir / "config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return model_path, stats_path


def _log_artifact(
    run,
    name: str,
    model_path: Path,
    stats_path: Path | None,
    manifests: list[Path],
    config_path: Path,
):
    artifact = wandb.Artifact(name=name, type="model")
    if model_path.exists():
        artifact.add_file(model_path)
    if stats_path and stats_path.exists():
        artifact.add_file(stats_path)
    for m in manifests:
        if m.exists():
            artifact.add_file(m)
    artifact.add_file(config_path)
    run.log_artifact(artifact)


# ---------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------


def train_once(
    algo: str, env_name: str, seed: int, md_train, md_eval, df_train, df_eval, args
):
    hp = args.hyperparams_data or {}
    algo_params = merged_algo_params(hp, algo, seed)
    env_params = env_cfg(hp)
    vn_params = vecnormalize_cfg(hp)

    # VecNormalize: allow YAML to set a default, CLI can override.
    if args.normalize is None:
        normalize = bool(vn_params.get("enable", False))
    else:
        normalize = bool(args.normalize)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group,
        config={
            "algo": algo,
            "env": env_name,
            "seed": seed,
            "total_timesteps": args.total_timesteps,
            "eval_freq": args.eval_freq,
            "eval_episodes": args.eval_episodes,
            "normalize": normalize,
        },
        name=args.run_name or f"{algo}-{env_name}-seed{seed}",
        resume="allow" if args.resume else None,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )

    algo_builder = get_algo_builder(algo)
    env_builder = get_env_builder(env_name)

    train_env, eval_env = env_builder.factory(
        md_train.prices, md_train.features, md_eval.prices, md_eval.features, env_params
    )
    train_env = maybe_wrap_vecnormalize(
        train_env,
        normalize,
        args.vecnorm_path,
        training=True,
        norm_obs=bool(vn_params.get("norm_obs", True)),
        norm_reward=bool(vn_params.get("norm_reward", True)),
        clip_obs=float(vn_params.get("clip_obs", 10.0)),
    )
    eval_env_wrapped = maybe_wrap_vecnormalize(
        eval_env,
        normalize,
        args.vecnorm_path,
        training=False,
        norm_obs=bool(vn_params.get("norm_obs", True)),
        norm_reward=bool(vn_params.get("norm_reward", True)),
        clip_obs=float(vn_params.get("clip_obs", 10.0)),
    )

    # sync stats if both are VecNormalize instances
    try:
        from stable_baselines3.common.vec_env import VecNormalize as VN

        if isinstance(train_env, VN) and isinstance(eval_env_wrapped, VN):
            eval_env_wrapped.obs_rms = train_env.obs_rms
            eval_env_wrapped.ret_rms = train_env.ret_rms
    except Exception:
        pass

    try:
        train_env.seed(seed)
        eval_env_wrapped.seed(seed)
    except Exception:
        pass

    # callbacks
    callbacks = CallbackList(
        [
            WandbCallback(log_freq=args.wandb_log_freq, verbose=0),
            WandbEvalCallback(
                eval_env_wrapped,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_episodes,
            ),
        ]
    )

    model_params = dict(algo_params)
    model_params["tensorboard_log"] = str(Path("runs") / run.id)

    if args.resume and args.checkpoint:
        model = algo_builder.algo_cls.load(args.checkpoint, env=train_env)
        reset_steps = False
    else:
        model = algo_builder.factory(train_env, model_params)
        reset_steps = True

    is_off_policy = algo.lower() in {"sac", "td3"}
    if is_off_policy:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_steps,
            log_interval=1,
        )
    elif args.sb3_log_interval is not None:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_steps,
            log_interval=args.sb3_log_interval,
        )
    else:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_steps,
        )

    run_dir = Path(args.output_dir) / run.id
    meta = {
        "algo": algo,
        "env": env_name,
        "seed": seed,
        "total_timesteps": args.total_timesteps,
    }
    model_path, stats_path = _save_checkpoint(run_dir, model, train_env, meta)

    manifest_train = _write_manifest(
        df_train, run_dir / "dataset_manifest_train.json", args, "train"
    )
    manifest_eval = _write_manifest(
        df_eval, run_dir / "dataset_manifest_eval.json", args, "eval"
    )
    config_path = run_dir / "config.json"
    _log_artifact(
        run,
        f"checkpoint-{run.id}",
        model_path,
        stats_path,
        [manifest_train, manifest_eval],
        config_path,
    )

    # quick eval summary
    prices = eval_env_wrapped.envs[0].unwrapped.prices
    wandb.log(
        {
            "baseline/buy_and_hold_return_pct": float(
                (compute_buy_and_hold(prices)[-1] - 1) * 100
            ),
            "baseline/sma_return_pct": float(
                (compute_sma_crossover(prices)[-1] - 1) * 100
            ),
        }
    )

    wandb.finish()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RL trading experiments (matrix over algos/envs/seeds)."
    )
    parser.add_argument(
        "--hyperparams",
        type=str,
        default=str(Path("trading_rl") / "config" / "sb3_finance_hyperparams.yaml"),
        help="Path to SB3 hyperparams YAML.",
    )
    parser.add_argument(
        "--algos", type=str, default="ppo", help="Comma-separated algo names."
    )
    parser.add_argument(
        "--envs", type=str, default="windowed", help="Comma-separated env names."
    )
    parser.add_argument(
        "--seeds", type=str, default="42", help="Comma-separated seeds."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="If set, seeds will be range(repeats).",
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=1)
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument(
        "--normalize", action="store_true", default=None, help="Enable VecNormalize."
    )
    norm_group.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Disable VecNormalize.",
    )
    parser.add_argument(
        "--vecnorm-path",
        type=str,
        default=None,
        help="Path to VecNormalize stats for resume/eval.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume an existing checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint for resume.",
    )
    parser.add_argument("--project", type=str, default="trading-rl")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-log-freq",
        type=int,
        default=1000,
        help="Log train metrics to W&B every N steps.",
    )
    parser.add_argument(
        "--sb3-log-interval",
        type=int,
        default=None,
        help="SB3 logger dump interval in episodes (off-policy algos force 1).",
    )
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--cache-dir", type=str, default="data_cache")
    parser.add_argument(
        "--csv-path", type=str, default=None, help="Local CSV with OHLCV data."
    )
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--timeframe", type=str, default="15Min")
    parser.add_argument(
        "--eval-start",
        type=str,
        default=None,
        help="Datetime (inclusive) for eval slice start.",
    )
    parser.add_argument(
        "--eval-end",
        type=str,
        default=None,
        help="Datetime (inclusive) for eval slice end.",
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="Alpaca API key (fallback to env)."
    )
    parser.add_argument(
        "--api-secret",
        type=str,
        default=None,
        help="Alpaca API secret (fallback to env).",
    )
    parser.add_argument(
        "--warmup-days",
        type=int,
        default=30,
        help="Extra days to fetch before start for indicator warmup.",
    )

    args = parser.parse_args()

    algos = parse_list(args.algos)
    envs = parse_list(args.envs)
    seeds = (
        list(range(args.repeats))
        if args.repeats is not None
        else [int(s) for s in parse_list(args.seeds)]
    )

    args.algos = algos
    args.envs = envs
    args.seeds = seeds

    args.hyperparams_data = (
        load_hyperparams(args.hyperparams) if args.hyperparams else {}
    )
    return args


def main():
    args = parse_args()

    df_raw = load_market_data(args)

    df_feat = add_talib_indicators(df_raw)

    start_ts = _ts_like_index(df_feat, args.start)
    end_ts = _ts_like_index(df_feat, args.end)
    df_feat = df_feat.loc[start_ts:end_ts]

    print("index tz:", df_feat.index.tz)
    print("min:", df_feat.index.min(), "max:", df_feat.index.max())
    print("start_ts:", start_ts, "end_ts:", end_ts)
    print("rows after clip:", len(df_feat))

    train_df, eval_df = split_train_eval(df_feat, args)

    window_size = int(args.hyperparams_data.get("env", {}).get("window_size", 512))
    min_len = window_size + 2
    assert len(train_df) >= min_len, f"train_df too short: {len(train_df)} < {min_len}"
    assert len(eval_df) >= min_len, f"eval_df too short: {len(eval_df)} < {min_len}"

    md_train = prepare_market_arrays(train_df)
    md_eval = prepare_market_arrays(eval_df)

    combos = expand_matrix(args.algos, args.envs, args.seeds)

    for combo in combos:
        train_once(
            combo["algo"],
            combo["env"],
            combo["seed"],
            md_train,
            md_eval,
            train_df,
            eval_df,
            args,
        )


if __name__ == "__main__":
    main()
