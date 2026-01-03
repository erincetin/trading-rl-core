from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np

from trading_rl.baselines.baselines import compute_buy_and_hold
from trading_rl.config.hyperparams import env_cfg, load_hyperparams
from trading_rl.data.alpaca_loader import AlpacaConfig
from trading_rl.data.indicators import add_talib_indicators
from trading_rl.data.loader import prepare_market_arrays
from trading_rl.experiment.data_pipeline import (
    load_market_data,
    split_train_eval,
    ts_like_index,
)
from trading_rl.registry import get_env_builder


def _parse_list(arg: str | Iterable[str]) -> list[str]:
    if isinstance(arg, str):
        return [x for x in arg.split(",") if x]
    return list(arg)


def _select_regime(hp: dict, name: str | None) -> dict:
    regimes = hp.get("regimes") or []
    if not regimes:
        raise ValueError("No regimes found in hyperparams.")
    if name is None:
        return regimes[0]
    for regime in regimes:
        if regime.get("name") == name:
            return regime
    names = [r.get("name") for r in regimes]
    raise ValueError(f"Regime '{name}' not found. Available: {names}")


def _weight_to_action(weight: float, *, max_position: float, action_transform: str) -> float:
    w = float(np.clip(weight, 0.0, max_position))
    if action_transform == "symmetric":
        if max_position <= 0:
            raise ValueError("max_position must be > 0 for symmetric transform")
        return 2.0 * (w / max_position) - 1.0
    return w


def _rollout(
    eval_env,
    *,
    weight: float,
    rebalance: bool,
    max_position: float,
    action_transform: str,
    max_steps: int | None,
    initial_cash: float,
):
    out = eval_env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, infos = out
    else:
        obs, infos = out, None

    if infos is None:
        infos = [{} for _ in range(eval_env.num_envs)]
    elif isinstance(infos, dict):
        infos = [infos]

    pv0 = float(infos[0].get("portfolio_value", initial_cash))
    pv_curve = [pv0]

    action_raw = _weight_to_action(
        weight, max_position=max_position, action_transform=action_transform
    )
    done0 = False
    step_idx = 0

    while not done0:
        action = np.full(
            (eval_env.num_envs, 1), action_raw, dtype=np.float32
        )
        step_out = eval_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            truncs = None
        else:
            obs, rewards, dones, truncs, infos = step_out

        info0 = infos[0]
        pv_next = info0.get("portfolio_value_next", info0.get("portfolio_value", pv_curve[-1]))
        pv_curve.append(float(pv_next))

        if not rebalance:
            realized_weight = float(info0.get("realized_weight", 0.0))
            action_raw = _weight_to_action(
                realized_weight, max_position=max_position, action_transform=action_transform
            )

        if truncs is None:
            done0 = bool(dones[0])
        else:
            done0 = bool(dones[0] or truncs[0])

        step_idx += 1
        if max_steps is not None and step_idx >= max_steps:
            break

    return pv_curve


def _summarize(name: str, pv_curve: list[float]) -> float:
    if not pv_curve:
        raise ValueError(f"{name} produced empty PV curve.")
    ret = pv_curve[-1] / max(pv_curve[0], 1e-12) - 1.0
    print(f"{name}: final PV={pv_curve[-1]:.2f}  return={ret*100:.2f}%")
    return ret * 100.0


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run a sanity sweep of fixed policies.")
    parser.add_argument(
        "--hyperparams",
        type=str,
        default=str("scripts/config.yaml"),
        help="Path to hyperparams YAML.",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        help="Regime name from config.yaml (default: first regime).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        help="Algo name to pick env overrides (default: ppo).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="windowed",
        help="Env name: windowed or vanilla.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Use local CSV; if missing, fetch from Alpaca and write it.",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Optional path to save Alpaca data as CSV.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Alpaca cache when fetching data.",
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-secret", type=str, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on eval steps.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to W&B (uses WANDB_MODE if set).",
    )
    parser.add_argument("--project", type=str, default="trading-rl")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    hp = load_hyperparams(args.hyperparams)
    regime = _select_regime(hp, args.regime)

    run_cfg = hp.get("run", {}) or {}
    csv_path = args.csv_path or run_cfg.get("csv_path")
    cache_dir = run_cfg.get("cache_dir")

    env_settings = env_cfg(hp, args.algo)
    env_settings = dict(env_settings)
    env_settings.setdefault("random_start", False)

    use_cache = not args.no_cache
    csv_path_str = str(csv_path) if csv_path else None
    if csv_path_str:
        try:
            df_raw = load_market_data(
                symbol=regime.get("symbol", "UNKNOWN"),
                start=regime["start"],
                end=regime["end"],
                timeframe=regime.get("timeframe", "1Min"),
                warmup_days=int(regime.get("warmup_days", 30)),
                csv_path=csv_path_str,
                alpaca_cfg=None,
            )
        except FileNotFoundError:
            df_raw = load_market_data(
                symbol=regime.get("symbol", "UNKNOWN"),
                start=regime["start"],
                end=regime["end"],
                timeframe=regime.get("timeframe", "1Min"),
                warmup_days=int(regime.get("warmup_days", 30)),
                csv_path=None,
                alpaca_cfg=AlpacaConfig(
                    api_key=args.api_key,
                    api_secret=args.api_secret,
                    cache_dir=cache_dir,
                ),
                use_cache=use_cache,
            )
            df_raw.reset_index().to_csv(csv_path_str, index=False)
    else:
        df_raw = load_market_data(
            symbol=regime.get("symbol", "UNKNOWN"),
            start=regime["start"],
            end=regime["end"],
            timeframe=regime.get("timeframe", "1Min"),
            warmup_days=int(regime.get("warmup_days", 30)),
            csv_path=None,
            alpaca_cfg=AlpacaConfig(
                api_key=args.api_key,
                api_secret=args.api_secret,
                cache_dir=cache_dir,
            ),
            use_cache=use_cache,
        )
        if args.save_csv:
            df_raw.reset_index().to_csv(args.save_csv, index=False)
    df_feat = add_talib_indicators(df_raw)
    start_ts = ts_like_index(df_feat, regime["start"])
    end_ts = ts_like_index(df_feat, regime["end"])
    df_feat = df_feat.loc[start_ts:end_ts]

    train_df, eval_df = split_train_eval(
        df_feat,
        eval_start=regime["eval_start"],
        eval_end=regime["eval_end"],
    )

    md_train = prepare_market_arrays(train_df)
    md_eval = prepare_market_arrays(eval_df)

    env_builder = get_env_builder(args.env)
    _, eval_env = env_builder.factory(
        md_train.prices,
        md_train.features,
        md_eval.prices,
        md_eval.features,
        env_settings,
    )

    max_position = float(env_settings.get("max_position", 1.0))
    action_transform = str(env_settings.get("action_transform", "identity")).lower()
    initial_cash = float(env_settings.get("initial_cash", 1_000_000.0))
    trade_cost = float(env_settings.get("trading_cost_pct", 0.001))

    pv_cash = _rollout(
        eval_env,
        weight=0.0,
        rebalance=True,
        max_position=max_position,
        action_transform=action_transform,
        max_steps=args.max_steps,
        initial_cash=initial_cash,
    )
    pv_full = _rollout(
        eval_env,
        weight=max_position,
        rebalance=True,
        max_position=max_position,
        action_transform=action_transform,
        max_steps=args.max_steps,
        initial_cash=initial_cash,
    )
    pv_bh = _rollout(
        eval_env,
        weight=max_position,
        rebalance=False,
        max_position=max_position,
        action_transform=action_transform,
        max_steps=args.max_steps,
        initial_cash=initial_cash,
    )

    print(f"Regime: {regime.get('name')}")
    print(f"Eval rows: {len(eval_df)}")
    print(f"Action transform: {action_transform} (max_position={max_position})")

    cash_ret = _summarize("cash 0%", pv_cash)
    full_ret = _summarize("target 100% (rebalance)", pv_full)
    bh_ret = _summarize("buy&hold (no rebalance)", pv_bh)

    bh_curve = compute_buy_and_hold(
        md_eval.prices, cost=trade_cost, include_exit_cost=False
    )
    bh_baseline = (bh_curve[-1] - 1.0) * 100.0
    print(f"baseline buy&hold return={bh_baseline:.2f}%")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=args.run_name or f"sanity-{regime.get('name', 'regime')}",
            config={
                "regime": regime.get("name"),
                "algo_env": args.algo,
                "env": args.env,
                "action_transform": action_transform,
                "max_position": max_position,
                "trade_cost_pct": trade_cost,
            },
        )
        wandb.log(
            {
                "sanity/cash_return_pct": cash_ret,
                "sanity/target_full_return_pct": full_ret,
                "sanity/buy_hold_return_pct": bh_ret,
                "baseline/buy_and_hold_return_pct": bh_baseline,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
