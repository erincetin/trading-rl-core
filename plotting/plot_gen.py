from __future__ import annotations

import os
import math
import pandas as pd
import matplotlib.pyplot as plt

SUMMARY_CSV = "wandb_runs_summary.csv"

# Output files
FIG_MAIN = "fig_excess_return_grouped.png"
FIG_PER_REGIME_PREFIX = "fig_excess_return_"  # + <regime>.png
TEX_TABLE = "table_results_summary.tex"

# Plot settings
DPI = 200  # paper-friendly PNG
BAR_WIDTH = 0.35


def latex_escape(s: str) -> str:
    # minimal escaping for table cells
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic sanity
    required = {"regime", "algo", "excess_mean", "excess_std", "beats_bh_rate", "bh_collapse_rate", "do_nothing_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    # Keep consistent ordering
    algo_order = ["a2c", "ppo", "sac", "td3"]
    # Normalize algo strings (optional)
    df["algo"] = df["algo"].astype(str)
    df["algo_norm"] = df["algo"].str.lower()

    # If your algo names are longer (e.g., "PPO"), map them
    mapping = {"ppo": "PPO", "a2c": "A2C", "sac": "SAC", "td3": "TD3"}
    df["algo_label"] = df["algo_norm"].map(mapping).fillna(df["algo"])

    # Sort regimes stable
    df["regime"] = df["regime"].astype(str)

    # Create categorical ordering for algos
    df["algo_norm"] = pd.Categorical(df["algo_norm"], categories=algo_order, ordered=True)
    df = df.sort_values(["regime", "algo_norm"])
    return df


def plot_grouped_excess(df: pd.DataFrame, out_path: str) -> None:
    regimes = list(df["regime"].unique())
    algos = list(df.sort_values("algo_norm")["algo_label"].unique())

    # Build matrices: shape (n_regimes, n_algos)
    # We'll align by algo_norm ordering within each regime
    pivot_mean = df.pivot(index="regime", columns="algo_label", values="excess_mean").reindex(index=regimes, columns=algos)
    pivot_std = df.pivot(index="regime", columns="algo_label", values="excess_std").reindex(index=regimes, columns=algos)

    x = list(range(len(algos)))
    fig, ax = plt.subplots(figsize=(10, 4.8))

    # offsets for grouped bars
    n = len(regimes)
    # Center groups around x; offsets in [-..., +...]
    offsets = [(i - (n - 1) / 2) * BAR_WIDTH for i in range(n)]

    for i, reg in enumerate(regimes):
        means = pivot_mean.loc[reg].tolist()
        stds = pivot_std.loc[reg].tolist()
        ax.bar(
            [xi + offsets[i] for xi in x],
            means,
            width=BAR_WIDTH,
            yerr=stds,
            capsize=3,
            label=reg,
        )

    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.set_ylabel("Excess return vs Buy-and-Hold (pct points)")
    ax.set_title("Excess return (mean ± std over seeds), grouped by regime")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_per_regime(df: pd.DataFrame, out_prefix: str) -> None:
    regimes = list(df["regime"].unique())
    for reg in regimes:
        sub = df[df["regime"] == reg].copy()
        sub = sub.sort_values("algo_norm")
        algos = sub["algo_label"].tolist()
        means = sub["excess_mean"].tolist()
        stds = sub["excess_std"].tolist()

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        x = list(range(len(algos)))
        ax.bar(x, means, yerr=stds, capsize=3)
        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.set_ylabel("Excess return vs Buy-and-Hold (pct points)")
        ax.set_title(f"Excess return (mean ± std over seeds): {reg}")
        fig.tight_layout()

        safe = reg.replace("/", "_").replace(" ", "_")
        out_path = f"{out_prefix}{safe}.png"
        fig.savefig(out_path, dpi=DPI)
        plt.close(fig)


def write_latex_table(df: pd.DataFrame, out_path: str) -> None:
    # Compact table: one row per (regime, algo)
    # Columns: excess_mean±std, beats, bh collapse, do nothing
    rows = []
    df2 = df.sort_values(["regime", "algo_norm"])
    for _, r in df2.iterrows():
        excess = r["excess_mean"]
        std = r["excess_std"]
        beats = r["beats_bh_rate"]
        bhc = r["bh_collapse_rate"]
        dn = r["do_nothing_rate"]

        # Pretty formatting
        excess_cell = f"{excess:.2f} $\\pm$ {0.0 if pd.isna(std) else std:.2f}"
        beats_cell = f"{100*beats:.0f}\\%"
        bhc_cell = f"{100*bhc:.0f}\\%"
        dn_cell = f"{100*dn:.0f}\\%"

        rows.append(
            (
                latex_escape(str(r["regime"])),
                latex_escape(str(r["algo_label"])),
                excess_cell,
                beats_cell,
                bhc_cell,
                dn_cell,
            )
        )

    # Emit as a LaTeX tabular you can \input{}
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated from wandb_runs_summary.csv\n")
        f.write("\\begin{tabular}{llcccc}\n")
        f.write("\\toprule\n")
        f.write("Regime & Algo & Excess (pp) & Beat B\\&H & B\\&H collapse & Do nothing\\\\\n")
        f.write("\\midrule\n")
        for reg, algo, exc, b, c, d in rows:
            f.write(f"{reg} & {algo} & {exc} & {b} & {c} & {d}\\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main() -> None:
    df = load_summary(SUMMARY_CSV)

    plot_grouped_excess(df, FIG_MAIN)
    plot_per_regime(df, FIG_PER_REGIME_PREFIX)
    write_latex_table(df, TEX_TABLE)

    print("Wrote:")
    print(f" - {FIG_MAIN}")
    print(f" - {FIG_PER_REGIME_PREFIX}<regime>.png (one per regime)")
    print(f" - {TEX_TABLE}")


if __name__ == "__main__":
    main()
