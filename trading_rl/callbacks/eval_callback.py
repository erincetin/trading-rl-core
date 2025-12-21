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
        verbose=0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic

    # ------------------------------
    # TRAIN-STEP CALLBACK
    # ------------------------------

    def _on_step(self):
        if self.num_timesteps % 8192 == 0:
            actions = self.locals.get("actions", None)
            if actions is not None:
                a = np.asarray(actions, dtype=np.float32)
                # flatten all envs + action dims
                flat = a.reshape(-1)
                if flat.size > 0:
                    wandb.log(
                        {
                            "train/action_mean": float(flat.mean()),
                            "train/action_std": float(flat.std()),
                            "train/action_min": float(flat.min()),
                            "train/action_max": float(flat.max()),
                            "train/action_hist": wandb.Histogram(flat),
                        }
                    )

        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            self._run_eval()

        return True

    # ------------------------------
    # EVALUATION LOGIC
    # ------------------------------

    def _run_eval(self):
        curves = []
        returns = []

        ep_returns = []
        ep_mdds = []
        ep_sharpes = []
        ep_abs_trade_values = []
        ep_turnovers = []
        ep_trades_counts = []

        # Resolve underlying env to access prices
        venv = getattr(
            self.eval_env, "venv", self.eval_env
        )  # unwrap VecNormalize if present
        base_env = venv.envs[0].unwrapped
        prices = base_env.prices

        trade_cost = 0.001
        cfg = getattr(base_env, "config", None)
        if cfg is not None:
            trade_cost = float(getattr(cfg, "trading_cost_pct", trade_cost))

        # Precompute baselines once
        bh_curve = compute_buy_and_hold(prices)
        sma_curve = compute_sma_crossover(prices, cost=trade_cost)

        wandb.log(
            {
                "baseline/buy_and_hold_return_pct": float((bh_curve[-1] - 1) * 100),
                "baseline/sma_return_pct": float((sma_curve[-1] - 1) * 100),
            }
        )

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
                if step_idx % 200 == 0:
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
                        }
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
            wandb.log(
                {
                    f"eval/portfolio_curve_ep{ep}": wandb.plot.line_series(
                        xs=list(range(len(pv_curve))),
                        ys=[pv_curve],
                        keys=[f"episode_{ep}"],
                        title="Evaluation Portfolio Value",
                        xname="step",
                    )
                }
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
            }
        )

        # Log baselines curves at the end of evaluation
        wandb.log(
            {
                "baseline/buy_and_hold_curve": wandb.plot.line_series(
                    xs=list(range(len(bh_curve))),
                    ys=[bh_curve],
                    keys=["buy_and_hold"],
                    title="Buy & Hold",
                    xname="step",
                ),
                "baseline/sma_curve": wandb.plot.line_series(
                    xs=list(range(len(sma_curve))),
                    ys=[sma_curve],
                    keys=["sma"],
                    title="SMA Baseline",
                    xname="step",
                ),
            }
        )
