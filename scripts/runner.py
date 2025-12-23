"""
Matrix runner for RL finance experiments.

Behavior:
- Regimes are loaded from --regimes (YAML/JSON list) if provided.
- Else, regimes are loaded from hyperparams YAML (key: "regimes") if present.
- Else, a single implicit regime is built from CLI fields.
- Matrix expands over regimes × algos × envs × seeds.
- --regime filters to a single regime by name.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Sequence

from trading_rl.config.hyperparams import load_hyperparams
from trading_rl.data.alpaca_loader import AlpacaConfig
from trading_rl.data.indicators import add_talib_indicators
from trading_rl.data.loader import prepare_market_arrays
from trading_rl.experiment.config import build_experiment_config
from trading_rl.experiment.data_pipeline import (
    load_market_data,
    split_train_eval,
    ts_like_index,
)
from trading_rl.experiment.orchestrator import train_once
from trading_rl.experiment.regimes import (
    apply_regime,
    load_regimes_file,
    load_regimes_from_hyperparams,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def parse_list(arg: str | Iterable[str]) -> List[str]:
    if isinstance(arg, str):
        return [x for x in arg.split(",") if x]
    return list(arg)


def _merge_dicts(base: dict, override: dict) -> dict:
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_run_cfg(hp: dict, algo: str) -> dict:
    base = hp.get("run", {}) or {}
    algo_block = hp.get(algo.lower(), {}) or {}
    override = algo_block.get("run", {}) or {}
    return _merge_dicts(base, override)


def _apply_run_cfg(args, run_cfg: dict, *, per_algo: bool) -> None:
    if not run_cfg:
        return

    if per_algo:
        keys = [
            "total_timesteps",
            "eval_freq",
            "eval_episodes",
            "normalize",
            "vecnorm_path",
            "resume",
            "checkpoint",
            "wandb_log_freq",
            "sb3_log_interval",
            "output_dir",
        ]
    else:
        keys = [
            "envs",
            "seeds",
            "repeats",
            "total_timesteps",
            "eval_freq",
            "eval_episodes",
            "normalize",
            "vecnorm_path",
            "resume",
            "checkpoint",
            "wandb_log_freq",
            "sb3_log_interval",
            "output_dir",
            "cache_dir",
            "csv_path",
        ]

    for key in keys:
        if key not in run_cfg:
            continue
        if per_algo:
            setattr(args, key, run_cfg[key])
        elif getattr(args, key) is None:
            setattr(args, key, run_cfg[key])


def _run_combo(args, combo: dict) -> None:
    # apply regime -> returns args-like object with overrides + regime_name
    rargs = apply_regime(args, combo["regime"])
    run_cfg = _resolve_run_cfg(args.hyperparams_data or {}, combo["algo"])
    _apply_run_cfg(rargs, run_cfg, per_algo=True)

    # data
    df_raw = load_market_data(
        symbol=rargs.symbol,
        start=rargs.start,
        end=rargs.end,
        timeframe=rargs.timeframe,
        warmup_days=rargs.warmup_days,
        csv_path=rargs.csv_path,
        alpaca_cfg=AlpacaConfig(
            api_key=rargs.api_key,
            api_secret=rargs.api_secret,
            cache_dir=rargs.cache_dir,
        ),
    )
    df_feat = add_talib_indicators(df_raw)

    start_ts = ts_like_index(df_feat, rargs.start)
    end_ts = ts_like_index(df_feat, rargs.end)
    df_feat = df_feat.loc[start_ts:end_ts]

    train_df, eval_df = split_train_eval(
        df_feat,
        eval_start=rargs.eval_start,
        eval_end=rargs.eval_end,
    )

    # minimum length sanity checks
    env_hp = (rargs.hyperparams_data or {}).get("env", {}) or {}
    window_size = int(env_hp.get("window_size", 512))
    if combo["env"] == "windowed":
        min_train_len = window_size + 1
        min_eval_len = 2
    else:
        min_train_len = 2
        min_eval_len = 2

    if len(train_df) < min_train_len or len(eval_df) < min_eval_len:
        raise ValueError(
            f"[{getattr(rargs, 'regime_name', 'default')}] too short: "
            f"train={len(train_df)} (need>={min_train_len}), "
            f"eval={len(eval_df)} (need>={min_eval_len})"
        )

    md_train = prepare_market_arrays(train_df)
    md_eval = prepare_market_arrays(eval_df)

    exp = build_experiment_config(
        args=rargs,
        hyperparams=rargs.hyperparams_data,
        regime=combo["regime"],
        algo=combo["algo"],
        env_name=combo["env"],
        seed=combo["seed"],
    )

    train_once(
        exp=exp,
        md_train=md_train,
        md_eval=md_eval,
        df_train=train_df,
        df_eval=eval_df,
    )


def _run_combo_worker(args_dict: dict, combo: dict) -> None:
    args = SimpleNamespace(**args_dict)
    _run_combo(args, combo)


def expand_matrix(
    regimes: Sequence[dict],
    algos: Sequence[str],
    envs: Sequence[str],
    seeds: Sequence[int],
):
    combos = []
    for regime in regimes:
        for algo in algos:
            for env in envs:
                for seed in seeds:
                    combos.append(
                        {"regime": regime, "algo": algo, "env": env, "seed": int(seed)}
                    )
    return combos


def _default_regimes_path() -> str | None:
    # scripts/config.yaml (same directory as this runner)
    p = Path(__file__).resolve().with_name("config.yaml")
    return str(p) if p.exists() else None


def _implicit_regime_from_cli(args) -> dict:
    # minimal shape expected by apply_regime + validator in regimes.py
    return {
        "name": "default",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "warmup_days": args.warmup_days,
        "start": args.start,
        "end": args.end,
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RL trading experiments (matrix over regimes/algos/envs/seeds)."
    )
    parser.add_argument(
        "--hyperparams",
        type=str,
        default=str(Path("scripts") / "config.yaml"),
        help="Path to SB3 hyperparams YAML (may include 'regimes').",
    )

    parser.add_argument(
        "--algos", type=str, default="ppo", help="Comma-separated algo names."
    )
    parser.add_argument(
        "--envs", type=str, default=None, help="Comma-separated env names."
    )
    parser.add_argument(
        "--seeds", type=str, default=None, help="Comma-separated seeds."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="If set, seeds will be range(repeats).",
    )

    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)

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
        "--resume",
        action="store_true",
        default=None,
        help="Resume an existing checkpoint.",
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
        default=None,
        help="Log train metrics to W&B every N steps.",
    )
    parser.add_argument(
        "--sb3-log-interval",
        type=int,
        default=None,
        help="SB3 logger dump interval (episodes).",
    )

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--csv-path", type=str, default=None, help="Local CSV with OHLCV data."
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Max number of experiments to run in parallel.",
    )

    # Single-regime fields (used only if no regimes are supplied)
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
        help="Extra days to fetch before start for warmup.",
    )

    parser.add_argument(
        "--regimes",
        type=str,
        default=None,
        help="Optional: YAML/JSON regimes list. Overrides hyperparams regimes.",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        help="Optional: run only a single regime by name.",
    )

    args = parser.parse_args()

    args.hyperparams_data = (
        load_hyperparams(args.hyperparams) if args.hyperparams else {}
    )

    args.algos = parse_list(args.algos)
    _apply_run_cfg(args, args.hyperparams_data.get("run", {}) or {}, per_algo=False)

    if args.envs is None:
        raise ValueError("envs not set. Add run.envs in config.yaml or pass --envs.")
    args.envs = parse_list(args.envs)

    if args.repeats is None and args.seeds is None:
        raise ValueError(
            "seeds not set. Add run.seeds or run.repeats in config.yaml "
            "or pass --seeds/--repeats."
        )
    args.seeds = (
        list(range(int(args.repeats)))
        if args.repeats is not None
        else [int(s) for s in parse_list(args.seeds)]
    )

    if args.total_timesteps is None:
        args.total_timesteps = 100_000
    if args.eval_freq is None:
        args.eval_freq = 10_000
    if args.eval_episodes is None:
        args.eval_episodes = 1
    if args.wandb_log_freq is None:
        args.wandb_log_freq = 1000
    if args.output_dir is None:
        args.output_dir = "models"
    if args.cache_dir is None:
        args.cache_dir = "data_cache"

    return args


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    args = parse_args()

    # regimes priority:
    # 1) --regimes file (if provided / default file exists)
    # 2) hyperparams 'regimes'
    # 3) implicit CLI regime
    if args.regimes:
        regimes = load_regimes_file(args.regimes)
    else:
        regimes = load_regimes_from_hyperparams(args.hyperparams_data)

    if not regimes:
        regimes = [_implicit_regime_from_cli(args)]

    if args.regime:
        wanted = args.regime
        filtered = [r for r in regimes if r.get("name") == wanted]
        if not filtered:
            available = [r.get("name") for r in regimes]
            raise ValueError(f"--regime '{wanted}' not found. Available: {available}")
        regimes = filtered

    combos = expand_matrix(regimes, args.algos, args.envs, args.seeds)

    if args.parallel and args.parallel > 1:
        args_dict = vars(args)
        with ProcessPoolExecutor(max_workers=int(args.parallel)) as executor:
            futures = [executor.submit(_run_combo_worker, args_dict, c) for c in combos]
            for fut in as_completed(futures):
                fut.result()
    else:
        for combo in combos:
            _run_combo(args, combo)


if __name__ == "__main__":
    main()
