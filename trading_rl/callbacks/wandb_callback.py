import wandb
from stable_baselines3.common.callbacks import BaseCallback


class WandbCallback(BaseCallback):
    """
    Logs:
      - episode rewards
      - episode portfolio value
      - loss & learning rate
      - actions
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = int(log_freq)
        self._ep_reward = None
        self._last_pv = None

    def _on_training_start(self):
        if wandb.run is None:
            return

        wandb.define_metric("train/step")
        wandb.define_metric("train/reward", step_metric="train/step")
        wandb.define_metric("train/portfolio_value", step_metric="train/step")
        wandb.define_metric("train/episode_reward", step_metric="train/step")
        wandb.define_metric("train/episode_portfolio_value", step_metric="train/step")

    def _on_step(self):
        infos = self.locals.get("infos", None)
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)

        if infos is None or rewards is None:
            return True

        info0 = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
        reward0 = float(rewards[0]) if hasattr(rewards, "__len__") else float(rewards)
        done0 = bool(dones[0]) if hasattr(dones, "__len__") and dones is not None else False

        pv = info0.get("portfolio_value", None)
        self._last_pv = pv

        if self._ep_reward is None:
            self._ep_reward = 0.0
        self._ep_reward += reward0

        if wandb.run is None:
            return True

        if self.log_freq > 0 and (self.num_timesteps % self.log_freq == 0):
            wandb.log(
                {
                    "train/step": self.num_timesteps,
                    "train/reward": reward0,
                    "train/portfolio_value": pv,
                }
            )

        if done0:
            wandb.log(
                {
                    "train/step": self.num_timesteps,
                    "train/episode_reward": float(self._ep_reward),
                    "train/episode_portfolio_value": pv,
                }
            )
            self._ep_reward = 0.0

        return True
