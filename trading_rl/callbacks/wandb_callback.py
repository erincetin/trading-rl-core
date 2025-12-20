import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class WandbCallback(BaseCallback):
    """
    Logs training signals aggregated across vectorized envs.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = int(log_freq)
        self._ep_rewards = None  # per-env running episode rewards

    def _on_training_start(self):
        if wandb.run is None:
            return

        wandb.define_metric("train/step")

        for k in [
            "train/reward_mean",
            "train/reward_median",
            "train/reward_min",
            "train/reward_max",
            "train/pv_mean",
            "train/pv_median",
            "train/pv_min",
            "train/pv_max",
            "train/done_count",
            "train/episode_reward_mean",
            "train/episode_reward_median",
        ]:
            wandb.define_metric(k, step_metric="train/step")

    def _on_step(self):
        infos = self.locals.get("infos", None)
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)

        if infos is None or rewards is None:
            return True

        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        n_envs = rewards.shape[0]

        if self._ep_rewards is None or len(self._ep_rewards) != n_envs:
            self._ep_rewards = np.zeros(n_envs, dtype=np.float32)

        self._ep_rewards += rewards

        # portfolio_value per env if available
        pvs = []
        if isinstance(infos, (list, tuple)):
            for inf in infos:
                if isinstance(inf, dict) and ("portfolio_value" in inf):
                    pvs.append(float(inf["portfolio_value"]))
        pvs = np.asarray(pvs, dtype=np.float32) if len(pvs) else None

        if wandb.run is None:
            return True

        if self.log_freq > 0 and (self.num_timesteps % self.log_freq == 0):
            log = {
                "train/step": int(self.num_timesteps),
                "train/reward_mean": float(rewards.mean()),
                "train/reward_median": float(np.median(rewards)),
                "train/reward_min": float(rewards.min()),
                "train/reward_max": float(rewards.max()),
            }
            if pvs is not None and len(pvs) == n_envs:
                log.update(
                    {
                        "train/pv_mean": float(pvs.mean()),
                        "train/pv_median": float(np.median(pvs)),
                        "train/pv_min": float(pvs.min()),
                        "train/pv_max": float(pvs.max()),
                    }
                )
            if dones is not None:
                done_arr = np.asarray(dones, dtype=bool).reshape(-1)
                log["train/done_count"] = int(done_arr.sum())

            wandb.log(log)

        # episode summary: log completed episodes (per-env)
        if dones is not None:
            done_arr = np.asarray(dones, dtype=bool).reshape(-1)
            if done_arr.any():
                finished = self._ep_rewards[done_arr]
                wandb.log(
                    {
                        "train/step": int(self.num_timesteps),
                        "train/episode_reward_mean": float(finished.mean()),
                        "train/episode_reward_median": float(np.median(finished)),
                    }
                )
                self._ep_rewards[done_arr] = 0.0

        return True
