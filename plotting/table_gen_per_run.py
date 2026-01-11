from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import pandas as pd
import wandb


# ----------------------------
# Edit these
# ----------------------------
ENTITY = "burakerinc-metu"
PROJECT = "trading-rl-windowed-long-3"
RUN_FILTER: Dict[str, Any] = {}  # e.g. {"state": "finished"} or {"config.algo": "ppo"}


# ----------------------------
# Regime name mapping
# ----------------------------
REGIME_RENAME = {
    "btc_2023-24_trainH1_evalQ3": "BTCUSD 1Hour",
    "btc_2022-23_trainM1-2_evalM3": "BTCUSD 15Min",
}


# ----------------------------
# Heuristics (no B&H collapse)
# ----------------------------
@dataclass(frozen=True)
class Thresholds:
    eps_win_pct: float = 0.25      # beats B&H if excess >= eps_win_pct (pp)
    do_nothing_turnover: float = 0.05
    low_trades: int = 5


TH = Thresholds()


def pick(summary: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        if k in summary and summary[k] is not None:
            v = summary[k]
            if isinstance(v, (int, float)):
                return float(v)
    return None


def pick_config(config: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in config:
            return config[k]
    return None


def _compute_excess(eval_ret: Optional[float], bh_ret: Optional[float]) -> Optional[float]:
    if eval_ret is None or bh_ret is None:
        return None
    return float(eval_ret) - float(bh_ret)


def classify_run(
    excess: Optional[float],
    trades: Optional[float],
    turnover: Optional[float],
    th: Thresholds = TH,
) -> str:
    if excess is None:
        return "missing_metrics"

    tr = float(trades or 0.0)
    to = float(turnover or 0.0)

    do_nothing = (tr <= th.low_trades) and (to <= th.do_nothing_turnover)
    beats = excess >= th.eps_win_pct

    if do_nothing:
        return "do_nothing"
    if beats:
        return "beats_buy_and_hold"
    return "active_not_better"


def get_best_metrics_from_history(
    run: wandb.apis.public.Run,
    history_samples: int = 20000,
) -> Dict[str, Optional[float]]:
    # Candidate keys (history may or may not include these)
    CAND_BH = ["baseline/buy_and_hold_return_pct", "eval/buy_and_hold_return_pct"]
    CAND_RET = ["eval/mean_return_pct", "eval/median_return_pct"]
    CAND_TRADES = ["eval/mean_trades_count", "eval/median_trades_count"]
    CAND_TURN = ["eval/mean_turnover", "eval/median_turnover"]

    try:
        h = run.history(samples=history_samples)
    except Exception:
        h = None

    if h is None or len(h) == 0:
        return {
            "best_source": "no_history",
            "best_eval_return_pct": None,
            "best_buy_hold_return_pct": None,
            "best_excess_return_pct": None,
            "best_trades": None,
            "best_turnover": None,
        }

    def first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    bh_col = first_col(h, CAND_BH)
    ret_col = first_col(h, CAND_RET)

    # If eval return is not logged as a series, we can't compute best from history
    if ret_col is None:
        return {
            "best_source": "no_eval_logged",
            "best_eval_return_pct": None,
            "best_buy_hold_return_pct": None,
            "best_excess_return_pct": None,
            "best_trades": None,
            "best_turnover": None,
        }

    h2 = h[pd.notna(h[ret_col])].copy()
    if len(h2) == 0:
        return {
            "best_source": "no_eval_rows",
            "best_eval_return_pct": None,
            "best_buy_hold_return_pct": None,
            "best_excess_return_pct": None,
            "best_trades": None,
            "best_turnover": None,
        }

    # Pick row by max excess if bh exists in history, else by max eval return
    if bh_col is not None and h2[bh_col].notna().any():
        h2["__excess"] = pd.to_numeric(h2[ret_col], errors="coerce") - pd.to_numeric(h2[bh_col], errors="coerce")
        idx = h2["__excess"].astype(float).idxmax()
    else:
        idx = pd.to_numeric(h2[ret_col], errors="coerce").astype(float).idxmax()

    row = h2.loc[idx]

    def row_pick(row_: pd.Series, candidates: List[str]) -> Optional[float]:
        for c in candidates:
            if c in row_.index and pd.notna(row_[c]):
                try:
                    return float(row_[c])
                except Exception:
                    return None
        return None

    best_bh = row_pick(row, CAND_BH)
    best_ret = row_pick(row, CAND_RET)
    best_excess = _compute_excess(best_ret, best_bh)

    return {
        "best_source": "history",
        "best_eval_return_pct": best_ret,
        "best_buy_hold_return_pct": best_bh,
        "best_excess_return_pct": best_excess,
        "best_trades": row_pick(row, CAND_TRADES),
        "best_turnover": row_pick(row, CAND_TURN),
    }


def fetch_runs_no_grouping(entity: str, project: str, run_filter: Dict[str, Any]) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=run_filter)

    rows = []
    for run in runs:
        s = dict(run.summary)
        c = dict(run.config)

        # Final metrics from summary
        final_bh = pick(s, "baseline/buy_and_hold_return_pct", "eval/buy_and_hold_return_pct")
        final_eval = pick(s, "eval/mean_return_pct", "eval/median_return_pct")
        final_excess = _compute_excess(final_eval, final_bh)
        final_trades = pick(s, "eval/mean_trades_count", "eval/median_trades_count")
        final_turnover = pick(s, "eval/mean_turnover", "eval/median_turnover")

        # Best metrics from history (fallback to final if missing)
        best = get_best_metrics_from_history(run)
        if best["best_excess_return_pct"] is None and final_excess is not None:
            best = {
                **best,
                "best_source": (best.get("best_source") or "unknown") + "->final_fallback",
                "best_eval_return_pct": final_eval,
                "best_buy_hold_return_pct": final_bh,
                "best_excess_return_pct": final_excess,
                "best_trades": final_trades,
                "best_turnover": final_turnover,
            }

        # Try to pull algo/regime/seed (adjust keys if your config differs)
        algo = pick_config(c, "algo", "algo_name", "algorithm", "model")
        regime = pick_config(c, "regime", "regime_name", "regime_id")
        seed = pick_config(c, "seed", "random_seed")

        # Rename regime for paper table
        regime_display = REGIME_RENAME.get(str(regime), str(regime) if regime is not None else None)

        label_best = classify_run(best["best_excess_return_pct"], best["best_trades"], best["best_turnover"])
        label_final = classify_run(final_excess, final_trades, final_turnover)

        rows.append(
            {
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "url": run.url,

                "regime": regime_display,
                "algo": algo,
                "seed": seed,

                "best_source": best.get("best_source"),
                "best_eval_return_pct": best.get("best_eval_return_pct"),
                "best_buy_hold_return_pct": best.get("best_buy_hold_return_pct"),
                "best_excess_return_pp": best.get("best_excess_return_pct"),
                "best_trades": best.get("best_trades"),
                "best_turnover": best.get("best_turnover"),
                "best_label": label_best,

                "final_eval_return_pct": final_eval,
                "final_buy_hold_return_pct": final_bh,
                "final_excess_return_pp": final_excess,
                "final_trades": final_trades,
                "final_turnover": final_turnover,
                "final_label": label_final,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["regime", "algo", "seed", "best_excess_return_pp"],
            ascending=[True, True, True, False],
            na_position="last",
        )
    return df


def to_latex_per_run_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "% (empty)\n"

    d = df.copy().head(max_rows)

    def fmt(x, nd=2):
        if pd.isna(x):
            return "--"
        return f"{float(x):.{nd}f}"

    d["Best excess (pp)"] = d["best_excess_return_pp"].map(lambda x: fmt(x, 2))
    d["Final excess (pp)"] = d["final_excess_return_pp"].map(lambda x: fmt(x, 2))
    d["Trades"] = d["best_trades"].map(lambda x: "--" if pd.isna(x) else f"{float(x):.1f}")
    d["Turnover"] = d["best_turnover"].map(lambda x: "--" if pd.isna(x) else f"{float(x):.2f}")
    d["Beat B\\&H"] = d["best_label"].map(lambda s: "yes" if s == "beats_buy_and_hold" else "no")

    out = d[["regime", "algo", "seed", "Best excess (pp)", "Final excess (pp)", "Trades", "Turnover", "Beat B\\&H"]]

    return out.to_latex(
        index=False,
        escape=False,
        longtable=False,
        column_format="llrccccr",
    )


if __name__ == "__main__":
    df = fetch_runs_no_grouping(ENTITY, PROJECT, RUN_FILTER)

    print("\nPer-run (first 30):")
    print(df.head(30).to_string(index=False))

    df.to_csv("wandb_runs_per_run.csv", index=False)
    print("\nWrote wandb_runs_per_run.csv")

    latex = to_latex_per_run_table(df, max_rows=40)
    with open("wandb_runs_per_run_table.tex", "w", encoding="utf-8") as f:
        f.write(latex)
    print("Wrote wandb_runs_per_run_table.tex")
