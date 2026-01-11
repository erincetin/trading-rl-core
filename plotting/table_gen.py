from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import wandb


# ----------------------------
# Edit these
# ----------------------------
ENTITY = "burakerinc-metu"
PROJECT = "trading-rl-windowed-long-3"
RUN_FILTER: Dict[str, Any] = {}  # e.g. {"config.algo": "a2c"} or {"tags": "windowed"}

# If your runs are huge, cap history rows per run (None = all)
HISTORY_SAMPLES: Optional[int] = None  # e.g. 2000


# ----------------------------
# Collapse / win heuristics
# ----------------------------
@dataclass(frozen=True)
class Thresholds:
    eps_pct: float = 0.50          # "close to buy&hold" in pct-points
    eps_win_pct: float = 0.50      # "beats buy&hold" threshold
    do_nothing_turnover: float = 0.05
    bh_turnover: float = 0.80
    low_trades: int = 10


TH = Thresholds()


# ----------------------------
# Helpers
# ----------------------------
def pick(summary: Dict[str, Any], *keys: str) -> Optional[float]:
    """Return the first present numeric metric from summary for the given keys."""
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
    bh_collapse = (abs(excess) <= th.eps_pct) and (tr <= 2) and (to >= th.bh_turnover)
    beats = excess >= th.eps_win_pct

    if do_nothing:
        return "do_nothing"
    if bh_collapse:
        return "buy_and_hold_collapse"
    if beats:
        return "beats_buy_and_hold"
    return "active_not_better"


def _compute_excess(eval_ret: Optional[float], bh: Optional[float]) -> Optional[float]:
    if eval_ret is None or bh is None:
        return None
    return float(eval_ret) - float(bh)


def get_final_metrics_from_summary(run: wandb.apis.public.Run) -> Dict[str, Optional[float]]:
    s = dict(run.summary)

    bh = pick(s, "baseline/buy_and_hold_return_pct", "eval/buy_and_hold_return_pct")
    eval_ret = pick(s, "eval/mean_return_pct", "eval/median_return_pct")
    excess = _compute_excess(eval_ret, bh)

    trades = pick(s, "eval/mean_trades_count", "eval/median_trades_count")
    turnover = pick(s, "eval/mean_turnover", "eval/median_turnover")
    mdd = pick(s, "eval/mean_max_drawdown_pct", "eval/median_max_drawdown_pct")
    sharpe = pick(s, "eval/mean_sharpe", "eval/median_sharpe")

    return {
        "final_buy_hold_return_pct": bh,
        "final_eval_return_pct": eval_ret,
        "final_excess_return_pct": excess,
        "final_max_drawdown_pct": mdd,
        "final_sharpe": sharpe,
        "final_trades": trades,
        "final_turnover": turnover,
    }

def get_best_metrics_from_history(
    run: wandb.apis.public.Run,
    history_samples: Optional[int] = 20000,
) -> Dict[str, Optional[float]]:
    # Candidate keys (try these in order; adjust if your project uses different names)
    CAND_BH = ["baseline/buy_and_hold_return_pct", "eval/buy_and_hold_return_pct"]
    CAND_RET = ["eval/mean_return_pct", "eval/median_return_pct"]
    CAND_SHARPE = ["eval/mean_sharpe", "eval/median_sharpe"]
    CAND_MDD = ["eval/mean_max_drawdown_pct", "eval/median_max_drawdown_pct"]
    CAND_TRADES = ["eval/mean_trades_count", "eval/median_trades_count"]
    CAND_TURN = ["eval/mean_turnover", "eval/median_turnover"]

    # Pull history without specifying keys first, so we can discover real column names
    try:
        h = run.history(samples=history_samples)
    except Exception:
        h = None

    if h is None or len(h) == 0:
        return {
            "best_source": "no_history",
            "best_step": None,
            "best_global_step": None,
            "best_buy_hold_return_pct": None,
            "best_eval_return_pct": None,
            "best_excess_return_pct": None,
            "best_max_drawdown_pct": None,
            "best_sharpe": None,
            "best_trades": None,
            "best_turnover": None,
        }

    # Helper: pick the first existing column name from candidates
    def first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    bh_col = first_col(h, CAND_BH)
    ret_col = first_col(h, CAND_RET)

    # If eval return was never logged as a timeseries, we cannot compute "best" from history
    if ret_col is None:
        return {
            "best_source": "no_eval_logged",
            "best_step": None,
            "best_global_step": None,
            "best_buy_hold_return_pct": None,
            "best_eval_return_pct": None,
            "best_excess_return_pct": None,
            "best_max_drawdown_pct": None,
            "best_sharpe": None,
            "best_trades": None,
            "best_turnover": None,
        }

    # Filter rows that actually have eval values
    h2 = h[pd.notna(h[ret_col])].copy()
    if len(h2) == 0:
        return {
            "best_source": "no_eval_rows",
            "best_step": None,
            "best_global_step": None,
            "best_buy_hold_return_pct": None,
            "best_eval_return_pct": None,
            "best_excess_return_pct": None,
            "best_max_drawdown_pct": None,
            "best_sharpe": None,
            "best_trades": None,
            "best_turnover": None,
        }

    # Compute excess if buy&hold exists in history; otherwise maximize eval return
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
        "best_step": row_pick(row, ["_step"]),
        "best_global_step": row_pick(row, ["global_step"]),
        "best_buy_hold_return_pct": best_bh,
        "best_eval_return_pct": best_ret,
        "best_excess_return_pct": best_excess,
        "best_max_drawdown_pct": row_pick(row, CAND_MDD),
        "best_sharpe": row_pick(row, CAND_SHARPE),
        "best_trades": row_pick(row, CAND_TRADES),
        "best_turnover": row_pick(row, CAND_TURN),
    }

def fetch_runs(entity: str, project: str, run_filter: Dict[str, Any]) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=run_filter)

    rows = []
    for run in runs:
        c = dict(run.config)

        final = get_final_metrics_from_summary(run)
        best = get_best_metrics_from_history(run, history_samples=HISTORY_SAMPLES)

        # Prefer best-vs-final classification based on BEST excess (that matches your selection logic).
        label = classify_run(best.get("best_excess_return_pct"), best.get("best_trades"), best.get("best_turnover"))

        # Extract algo/regime/seed (adjust keys to your logging)
        algo = pick_config(c, "algo", "algo_name", "algorithm", "model", "run_name")
        regime = pick_config(c, "regime", "regime_name", "regime_id")
        seed = pick_config(c, "seed", "random_seed")

        # Deltas (best - final)
        excess_delta = None
        if best.get("best_excess_return_pct") is not None and final.get("final_excess_return_pct") is not None:
            excess_delta = float(best["best_excess_return_pct"]) - float(final["final_excess_return_pct"])

        rows.append(
            {
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "url": run.url,
                "algo": algo,
                "regime": regime,
                "seed": seed,
                **final,
                **best,
                "excess_delta_pp": excess_delta,
                "label_best": label,
                # Optional: also classify FINAL behavior
                "label_final": classify_run(final.get("final_excess_return_pct"), final.get("final_trades"), final.get("final_turnover")),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            by=["regime", "algo", "seed", "best_excess_return_pct"],
            ascending=[True, True, True, False],
            na_position="last",
        )
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across seeds per (regime, algo)."""
    if df.empty:
        return df

    grouped = df.groupby(["regime", "algo"], dropna=False)

    agg = grouped.agg(
        n_runs=("run_id", "count"),

        # Best
        best_excess_mean=("best_excess_return_pct", "mean"),
        best_excess_std=("best_excess_return_pct", "std"),
        best_sharpe_mean=("best_sharpe", "mean"),
        best_mdd_mean=("best_max_drawdown_pct", "mean"),
        best_trades_mean=("best_trades", "mean"),
        best_turnover_mean=("best_turnover", "mean"),

        # Final
        final_excess_mean=("final_excess_return_pct", "mean"),
        final_excess_std=("final_excess_return_pct", "std"),
        final_sharpe_mean=("final_sharpe", "mean"),
        final_mdd_mean=("final_max_drawdown_pct", "mean"),
        final_trades_mean=("final_trades", "mean"),
        final_turnover_mean=("final_turnover", "mean"),

        # Deltas
        excess_delta_mean=("excess_delta_pp", "mean"),
        excess_delta_std=("excess_delta_pp", "std"),

        # Collapse / win counts based on BEST label (matches "best checkpoint" narrative)
        beats_bh=("label_best", lambda s: (s == "beats_buy_and_hold").sum()),
        bh_collapse=("label_best", lambda s: (s == "buy_and_hold_collapse").sum()),
        do_nothing=("label_best", lambda s: (s == "do_nothing").sum()),
    ).reset_index()

    for col in ["beats_bh", "bh_collapse", "do_nothing"]:
        agg[col + "_rate"] = agg[col] / agg["n_runs"]

    agg = agg.sort_values(by="best_excess_mean", ascending=False, na_position="last")
    return agg

def to_latex_results_table(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "% (empty summary)\n"

    d = summary.copy()

    # Columns we need from summarize()
    cols = [
        "regime", "algo",
        "best_excess_mean", "best_excess_std",
        "final_excess_mean", "final_excess_std",
        "best_trades_mean", "best_turnover_mean",
        "beats_bh_rate", "bh_collapse_rate",
    ]
    d = d[cols]

    def fmt_mean_std(mean, std) -> str:
        if pd.isna(mean):
            return "--"
        if pd.isna(std):
            return f"{mean:.2f}"
        return f"{mean:.2f} $\\pm$ {std:.2f}"

    d["Best excess (pp)"] = [
        fmt_mean_std(m, s) for m, s in zip(d["best_excess_mean"], d["best_excess_std"])
    ]
    d["Final excess (pp)"] = [
        fmt_mean_std(m, s) for m, s in zip(d["final_excess_mean"], d["final_excess_std"])
    ]

    d["Trades"] = d["best_trades_mean"].map(lambda x: "--" if pd.isna(x) else f"{x:.1f}")
    d["Turnover"] = d["best_turnover_mean"].map(lambda x: "--" if pd.isna(x) else f"{x:.2f}")

    d["Beat B\\&H"] = d["beats_bh_rate"].map(lambda x: "--" if pd.isna(x) else f"{100*x:.0f}\\%")
    d["B\\&H collapse"] = d["bh_collapse_rate"].map(lambda x: "--" if pd.isna(x) else f"{100*x:.0f}\\%")

    out = d[
        ["regime", "algo", "Best excess (pp)", "Final excess (pp)", "Trades", "Turnover", "Beat B\\&H", "B\\&H collapse"]
    ]

    latex = out.to_latex(
        index=False,
        escape=False,
        longtable=False,
        caption=None,
        label=None,
        column_format="llcccccc",
    )
    return latex



if __name__ == "__main__":
    df = fetch_runs(ENTITY, PROJECT, RUN_FILTER)
    print("\nPer-run table (top 30 rows):")
    print(df.head(30).to_string(index=False))

    summary = summarize(df)
    print("\nAggregated by (regime, algo):")
    print(summary.to_string(index=False))

    # Optional exports
    df.to_csv("wandb_runs_table.csv", index=False)
    summary.to_csv("wandb_runs_summary.csv", index=False)

    # Optional: LaTeX
    with open("wandb_results_table.tex", "w", encoding="utf-8") as f:
        f.write(to_latex_results_table(summary))

    print("\nWrote wandb_runs_table.csv, wandb_runs_summary.csv, wandb_results_table.tex")
