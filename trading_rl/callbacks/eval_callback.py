import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from trading_rl.baselines.baselines import compute_buy_and_hold, compute_sma_crossover


def _max_drawdown(pv: np.ndarray) -> float:
    pv = np.asarray(pv, dtype=np.float64)
    if pv.size < 2:
        return 0.0
    running_max = np.maximum.accumulate(pv)
    dd = pv / np.maximum(running_max, 1e-12) - 1.0
    return float(dd.min())  # negative


def _sharpe(step_returns: np.ndarray) -> float:
    r = np.asarray(step_returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    std = r.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(r.mean() / std)


def _percentiles(x, ps=(10, 25, 50, 75, 90)):
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return {f"p{p}": 0.0 for p in ps}
    vals = np.percentile(arr, ps)
    return {f"p{p}": float(v) for p, v in zip(ps, vals)}


def _downsample_indices(size: int, max_points: int) -> np.ndarray:
    if size <= 0:
        return np.asarray([], dtype=int)
    if max_points <= 0 or size <= max_points:
        return np.arange(size, dtype=int)
    return np.linspace(0, size - 1, num=max_points, dtype=int)


def _downsample_curve(values, max_points: int):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.asarray([], dtype=int), arr
    idx = _downsample_indices(arr.size, max_points)
    return idx, arr[idx]


def _downsample_values(values, max_points: int):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0 or max_points <= 0 or arr.size <= max_points:
        return arr
    idx = _downsample_indices(arr.size, max_points)
    return arr[idx]


class WandbEvalCallback(BaseCallback):
    """
    Runs evaluation episodes every eval_freq steps and logs results to W&B.
    """

    def __init__(
        self,
        eval_env,
        eval_freq=10_000,
        n_eval_episodes=1,
        deterministic=True,
        log_eval_curves: bool = True,
        log_baseline_curves: bool = True,
        log_action_hist: bool = True,
        log_debug: bool = False,
        wandb_curve_max_points: int = 300,
        wandb_action_hist_freq: int = 8192,
        wandb_hist_max_points: int = 2000,
        verbose=0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.log_eval_curves = bool(log_eval_curves)
        self.log_baseline_curves = bool(log_baseline_curves)
        self.log_action_hist = bool(log_action_hist)
        self.log_debug = bool(log_debug)
        self.curve_max_points = max(int(wandb_curve_max_points), 0)
        self.action_hist_freq = max(int(wandb_action_hist_freq), 0)
        self.hist_max_points = max(int(wandb_hist_max_points), 0)
        self._baseline_logged = False
        self._baseline_return_pct = None
        self._baseline_sma_return_pct = None
        self._baseline_end_pv = None
        self._bh_curve = None
        self._sma_curve = None

    def _ensure_baselines(self) -> None:
        if self._bh_curve is not None and self._sma_curve is not None:
            return

        venv = getattr(self.eval_env, "venv", self.eval_env)
        base_env = venv.envs[0].unwrapped
        prices = base_env.prices

        trade_cost = 0.001
        initial_cash = 1_000_000.0
        cfg = getattr(base_env, "config", None)
        if cfg is not None:
            trade_cost = float(getattr(cfg, "trading_cost_pct", trade_cost))
            initial_cash = float(getattr(cfg, "initial_cash", initial_cash))

        bh_curve = compute_buy_and_hold(
            prices, cost=trade_cost, include_exit_cost=False
        )
        sma_curve = compute_sma_crossover(prices, cost=trade_cost)

        self._bh_curve = bh_curve
        self._sma_curve = sma_curve
        self._baseline_return_pct = float((bh_curve[-1] - 1) * 100)
        self._baseline_sma_return_pct = float((sma_curve[-1] - 1) * 100)
        self._baseline_end_pv = float(initial_cash * bh_curve[-1])

    def _log_baseline_curves(self, step: int) -> None:
        if not self.log_baseline_curves or self.curve_max_points == 0:
            return
        if self._bh_curve is None or self._sma_curve is None:
            return

        xs_bh, ys_bh = _downsample_curve(self._bh_curve, self.curve_max_points)
        xs_sma, ys_sma = _downsample_curve(self._sma_curve, self.curve_max_points)

        wandb.log(
            {
                "baseline/buy_and_hold_curve": wandb.plot.line_series(
                    xs=xs_bh.tolist(),
                    ys=[ys_bh.tolist()],
                    keys=["buy_and_hold"],
                    title="Buy & Hold",
                    xname="step",
                ),
                "baseline/sma_curve": wandb.plot.line_series(
                    xs=xs_sma.tolist(),
                    ys=[ys_sma.tolist()],
                    keys=["sma"],
                    title="SMA Baseline",
                    xname="step",
                ),
            },
            step=step,
        )

    def _on_training_start(self):
        if wandb.run is None:
            return
        self._ensure_baselines()
        if not self._baseline_logged:
            self._log_baseline_curves(step=0)
            self._baseline_logged = True

    # ------------------------------
    # TRAIN-STEP CALLBACK
    # ------------------------------

    def _on_step(self):
        if wandb.run is None:
            return True

        if (
            self.log_action_hist
            and self.action_hist_freq > 0
            and self.num_timesteps % self.action_hist_freq == 0
        ):
            actions = self.locals.get("actions", None)
            if actions is not None:
                a = np.asarray(actions, dtype=np.float32)
                # flatten all envs + action dims
                flat = _downsample_values(a.reshape(-1), self.hist_max_points)
                if flat.size > 0:
                    wandb.log(
                        {
                            "train/action_mean": float(flat.mean()),
                            "train/action_std": float(flat.std()),
                            "train/action_min": float(flat.min()),
                            "train/action_max": float(flat.max()),
                            "train/action_hist": wandb.Histogram(flat),
                        },
                        step=int(self.num_timesteps),
                    )

        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            self._run_eval()

        return True

    # ------------------------------
    # EVALUATION LOGIC
    # ------------------------------

    def _run_eval(self):
        if wandb.run is None:
            return

        step = int(self.num_timesteps)

        self._ensure_baselines()

        if not self._baseline_logged:
            baseline_log = {
                "baseline/buy_and_hold_return_pct": float(self._baseline_return_pct)
                if self._baseline_return_pct is not None
                else np.nan,
                "baseline/sma_return_pct": float(self._baseline_sma_return_pct)
                if self._baseline_sma_return_pct is not None
                else np.nan,
            }
            if self.log_baseline_curves and self.curve_max_points != 0:
                xs_bh, ys_bh = _downsample_curve(
                    self._bh_curve, self.curve_max_points
                )
                xs_sma, ys_sma = _downsample_curve(
                    self._sma_curve, self.curve_max_points
                )
                baseline_log.update(
                    {
                        "baseline/buy_and_hold_curve": wandb.plot.line_series(
                            xs=xs_bh.tolist(),
                            ys=[ys_bh.tolist()],
                            keys=["buy_and_hold"],
                            title="Buy & Hold",
                            xname="step",
                        ),
                        "baseline/sma_curve": wandb.plot.line_series(
                            xs=xs_sma.tolist(),
                            ys=[ys_sma.tolist()],
                            keys=["sma"],
                            title="SMA Baseline",
                            xname="step",
                        ),
                    }
                )
            wandb.log(baseline_log, step=step)
            self._baseline_logged = True

        ep_returns = []
        ep_mdds = []
        ep_sharpes = []
        ep_abs_trade_values = []
        ep_turnovers = []
        ep_trades_counts = []

        for ep in range(self.n_eval_episodes):
            out = self.eval_env.reset()
            if isinstance(out, tuple) and len(out) == 2:
                obs, infos = out
            else:
                obs, infos = out, None

            # VecEnv reset returns infos as list[dict] (one per env)
            if infos is None:
                infos = [{} for _ in range(self.eval_env.num_envs)]
            elif isinstance(infos, dict):
                # just in case
                infos = [infos]

            i0 = 0  # evaluate env-0 deterministically

            pv0 = float(infos[i0].get("portfolio_value", 1_000_000.0))
            pv_curve = [pv0]

            abs_trade_value = 0.0
            trades_count = 0

            done0 = False
            step_idx = 0

            while not done0:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)

                step_out = self.eval_env.step(action)

                # SB3 VecEnv: (obs, rewards, dones, infos)  [4 items]
                if len(step_out) == 4:
                    obs, rewards, dones, infos = step_out
                    truncs = None
                else:
                    # if you ever use a VecEnv that returns truncs separately
                    obs, rewards, dones, truncs, infos = step_out

                # infos is list[dict]
                info0 = infos[i0]

                pv_next = info0.get(
                    "portfolio_value_next", info0.get("portfolio_value", pv_curve[-1])
                )
                pv_curve.append(float(pv_next))

                tv = float(info0.get("trade_value_executed", 0.0))
                abs_trade_value += abs(tv)
                if abs(tv) > 1e-6:
                    trades_count += 1

                # done for env0
                if truncs is None:
                    done0 = bool(dones[i0])
                else:
                    done0 = bool(dones[i0] or truncs[i0])

                # optional debug every N steps
                if self.log_debug and step_idx % 200 == 0:
                    wandb.log(
                        {
                            "debug/eval_action_target_weight_env0": float(
                                info0.get("action_target_weight", np.nan)
                            ),
                            "debug/eval_realized_weight_env0": float(
                                info0.get("realized_weight", np.nan)
                            ),
                            "debug/eval_trade_value_executed_env0": float(
                                info0.get("trade_value_executed", 0.0)
                            ),
                            "debug/eval_portfolio_value_env0": float(
                                info0.get("portfolio_value", pv_curve[-1])
                            ),
                        },
                        step=step,
                    )
                step_idx += 1

                # ---- finalize episode metrics ----
            pv_arr = np.asarray(pv_curve, dtype=np.float64)
            ret = float(pv_arr[-1] / max(pv_arr[0], 1e-12) - 1.0)

            step_rets = pv_arr[1:] / np.maximum(pv_arr[:-1], 1e-12) - 1.0
            sharpe = _sharpe(step_rets)
            mdd = _max_drawdown(pv_arr)

            turnover = float(abs_trade_value / max(pv_arr[0], 1e-12))

            ep_returns.append(ret)
            ep_sharpes.append(sharpe)
            ep_mdds.append(mdd)
            ep_abs_trade_values.append(float(abs_trade_value))

            ep_turnovers.append(turnover)
            ep_trades_counts.append(int(trades_count))

            # optional: log curve
            if self.log_eval_curves and ep == 0 and self.curve_max_points != 0:
                xs_eval, ys_eval = _downsample_curve(pv_curve, self.curve_max_points)
                keys = ["episode_0"]
                ys = [ys_eval.tolist()]
                if self._baseline_end_pv is not None:
                    keys.append("buy_and_hold_end")
                    ys.append([float(self._baseline_end_pv)] * len(xs_eval))
                wandb.log(
                    {
                        "eval/portfolio_curve": wandb.plot.line_series(
                            xs=xs_eval.tolist(),
                            ys=ys,
                            keys=keys,
                            title="Evaluation Portfolio Value",
                            xname="step",
                        )
                    },
                    step=step,
                )

        # Summary statistics
        ret_pct = [r * 100 for r in ep_returns]
        mdd_pct = [d * 100 for d in ep_mdds]

        ret_ps = _percentiles(ret_pct)
        mdd_ps = _percentiles(mdd_pct)

        wandb.log(
            {
                # returns
                "eval/mean_return_pct": float(np.mean(ret_pct)),
                "eval/median_return_pct": ret_ps["p50"],
                "eval/p10_return_pct": ret_ps["p10"],
                "eval/p25_return_pct": ret_ps["p25"],
                "eval/p75_return_pct": ret_ps["p75"],
                "eval/p90_return_pct": ret_ps["p90"],
                "eval/win_rate_pct": 100.0
                * float(np.mean(np.asarray(ep_returns) > 0.0)),
                # risk
                "eval/mean_max_drawdown_pct": float(np.mean(mdd_pct)),
                "eval/median_max_drawdown_pct": mdd_ps["p50"],
                # sharpe
                "eval/mean_sharpe": float(np.mean(ep_sharpes)),
                "eval/median_sharpe": float(np.median(ep_sharpes)),
                # trading activity
                "eval/mean_turnover": float(np.mean(ep_turnovers)),
                "eval/median_turnover": float(np.median(ep_turnovers)),
                "eval/mean_abs_trade_value": float(np.mean(ep_abs_trade_values)),
                "eval/median_abs_trade_value": float(np.median(ep_abs_trade_values)),
                "eval/mean_trades_count": float(np.mean(ep_trades_counts)),
                "eval/median_trades_count": float(np.median(ep_trades_counts)),
                "baseline/buy_and_hold_return_pct": float(self._baseline_return_pct)
                if self._baseline_return_pct is not None
                else np.nan,
                "baseline/sma_return_pct": float(self._baseline_sma_return_pct)
                if self._baseline_sma_return_pct is not None
                else np.nan,
            },
            step=step,
        )
