import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from trading_rl.baselines.baselines import compute_buy_and_hold, compute_sma_crossover


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
        # Log action histograms during training (not evaluation)
        if self.num_timesteps % 5000 == 0:
            actions = self.locals.get("actions", None)
            if actions is not None:
                a = actions[0]
                wandb.log(
                    {
                        "train/action_value": a,
                        "train/action_hist": wandb.Histogram(a),
                    }
                )

        # Trigger periodic evaluation
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            self._run_eval()

        return True

    # ------------------------------
    # EVALUATION LOGIC
    # ------------------------------

    def _run_eval(self):
        curves = []
        returns = []

        # Resolve underlying env to access prices
        base_env = self.eval_env.envs[0].unwrapped
        prices = base_env.prices

        # Precompute baselines once
        bh_curve = compute_buy_and_hold(prices)
        sma_curve = compute_sma_crossover(prices)

        wandb.log(
            {
                "baseline/buy_and_hold_return_pct": float((bh_curve[-1] - 1) * 100),
                "baseline/sma_return_pct": float((sma_curve[-1] - 1) * 100),
            }
        )

        for ep in range(self.n_eval_episodes):
            out = self.eval_env.reset()
            if isinstance(out, tuple) and len(out) == 2:
                obs, info = out
            else:
                obs, info = out, {}
            done = False
            truncated = False
            pv_curve = [info.get("portfolio_value", 1_000_000)]

            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=self.deterministic)

                step_out = self.eval_env.step(action)

                # Handle old Gym (4 outputs) and new Gymnasium (5 outputs)
                if len(step_out) == 4:
                    obs, reward, done, info = step_out
                    truncated = False
                else:
                    obs, reward, done, truncated, info = step_out

                info = info[0]  # vec env wraps info in a list

                pv_curve.append(info.get("portfolio_value", pv_curve[-1]))

            curves.append(pv_curve)
            returns.append(pv_curve[-1] / pv_curve[0] - 1)

            # Log each evaluation curve
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
        wandb.log(
            {
                "eval/mean_return_pct": float(np.mean(returns) * 100),
                "eval/max_return_pct": float(np.max(returns) * 100),
                "eval/min_return_pct": float(np.min(returns) * 100),
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
