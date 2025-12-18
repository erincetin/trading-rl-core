# scripts/train.py
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize

from trading_rl.data.alpaca_loader import AlpacaDataLoader, AlpacaConfig
from trading_rl.data.indicators import add_talib_indicators
from trading_rl.data.loader import prepare_market_arrays

from trading_rl.envs.trading_env import TradingEnv, TradingEnvConfig
from trading_rl.envs.windowed_wrapper import WindowedTradingEnv, WindowedEnvConfig

from trading_rl.baselines.baselines import compute_buy_and_hold
from trading_rl.baselines.baselines import compute_sma_crossover

import wandb
from wandb.integration.sb3 import WandbCallback
from trading_rl.callbacks.eval_callback import WandbEvalCallback

# -------------------------------------------------------------------
# 1. Load + prepare data
# -------------------------------------------------------------------


def load_market(symbol="AAPL"):
    cfg = AlpacaConfig(cache_dir="data_cache")

    loader = AlpacaDataLoader(cfg)

    df = loader.load(
        symbol=symbol,
        start="2024-01-01",
        end="2024-12-31",
        timeframe="15Min",
    )

    df = add_talib_indicators(df)

    # Fill missing values from indicators
    df = df.ffill().dropna()

    return df


# -------------------------------------------------------------------
# 2. Build environments
# -------------------------------------------------------------------


def make_env(prices, features, is_train=True):
    base_env = TradingEnv(
        prices=prices,
        features=features,
        config=TradingEnvConfig(
            trading_cost_pct=0.001,
            reward_mode="diff_return",
        ),
    )

    if is_train:
        wrapper = WindowedTradingEnv(
            prices=prices,
            features=features,
            env_config=TradingEnvConfig(),
            window_cfg=WindowedEnvConfig(
                window_size=5000,
                random_start=True,
            ),
        )
        return wrapper

    # deterministic full episode for test
    return base_env


def build_vec_env(env_fn):
    return DummyVecEnv([env_fn])


# -------------------------------------------------------------------
# 3. Main training logic
# -------------------------------------------------------------------


def main():
    run = wandb.init(
        project="trading-rl",
        config={
            "timeframe": "15Min",
            "env": "WindowedTradingEnv",
            "algo": "PPO",
            "total_timesteps": 500000,
        },
        sync_tensorboard=True,  # <---- IMPORTANT
        monitor_gym=False,  # you can set True if you want videos
        save_code=True,
    )

    print("Loading market data...")
    df = load_market("AAPL")

    # Split into train & test
    df_train = df.loc["2024-01-01":"2024-08-31"]
    df_test = df.loc["2024-09-01":"2024-12-31"]

    print("Preparing arrays...")
    train_md = prepare_market_arrays(df_train)
    test_md = prepare_market_arrays(df_test)

    # Build environments
    print("Building training env...")

    def train_env_fn():
        return make_env(train_md.prices, train_md.features, is_train=True)

    train_env = build_vec_env(train_env_fn)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ----------------------------------------------------------------
    # Train PPO
    # ----------------------------------------------------------------
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        n_steps=2048,
        gamma=0.99,
        tensorboard_log=f"runs/{run.id}",
    )

    eval_env = make_env(test_md.prices, test_md.features, is_train=False)
    eval_env = DummyVecEnv([lambda: eval_env])

    prices = eval_env.envs[0].unwrapped.prices

    eval_env.baselines = {
        "buy_and_hold": compute_buy_and_hold(prices),
        "sma_20_50": compute_sma_crossover(prices, 20, 50),
    }

    callbacks = [
        WandbCallback(model_save_path=f"models/{run.id}", verbose=1),
        WandbEvalCallback(eval_env, eval_freq=10000, n_eval_episodes=1),
    ]

    print("Training...")
    model.learn(total_timesteps=500000, callback=callbacks)
    train_env.save("vecnormalize.pkl")

    # ----------------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------------
    print("Evaluating...")

    print("Building test env...")
    raw_test_env = DummyVecEnv(
        [lambda: make_env(test_md.prices, test_md.features, is_train=False)]
    )
    test_env = VecNormalize.load("vecnormalize.pkl", raw_test_env)

    test_env.training = False
    test_env.norm_reward = False

    obs = test_env.reset()
    info = {}
    done = False

    portfolio_values = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        info = info[0]  # unwrap from vec env
        portfolio_values.append(info["portfolio_value"])

    start = portfolio_values[0]
    end = portfolio_values[-1]

    print(f"Start PV: {start:.2f}")
    print(f"End PV:   {end:.2f}")
    print(f"Return:   {(end / start - 1) * 100:.2f}%")

    # Save model
    model.save("ppo_trading_model")
    wandb.save("ppo_trading_model.zip")

    metrics = {
        "final_pv": end,
        "test_return_pct": (end / start - 1) * 100,
        "steps": len(portfolio_values),
    }

    baseline = test_md.prices[-1] / test_md.prices[0] - 1
    wandb.log({"baseline_buy_and_hold_return": baseline})

    wandb.log(metrics)


if __name__ == "__main__":
    main()
