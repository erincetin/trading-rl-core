from types import SimpleNamespace

from trading_rl.experiment.config import build_experiment_config


def test_build_experiment_config_group_and_name():
    args = SimpleNamespace(
        group=None,
        run_name=None,
        project="proj",
        symbol="AAPL",
        timeframe="1Min",
        warmup_days=5,
        total_timesteps=100,
        eval_freq=10,
        eval_episodes=1,
        normalize=None,
        vecnorm_path=None,
        wandb_log_freq=1000,
        resume=False,
        checkpoint=None,
        sb3_log_interval=None,
        output_dir="models",
    )

    regime = {
        "name": "r0",
        "start": "2024-01-01",
        "end": "2024-01-02",
        "eval_start": "2024-01-01",
        "eval_end": "2024-01-02",
    }

    hp = {"shared": {"learning_rate": 0.001}, "ppo": {"policy": "MlpPolicy"}}

    exp = build_experiment_config(
        args=args,
        hyperparams=hp,
        regime=regime,
        algo="ppo",
        env_name="vanilla",
        seed=7,
    )

    assert exp.name == "ppo-vanilla-r0-seed7"
    assert exp.group == "ppo-vanilla-r0"
    assert exp.symbol == "AAPL"
    assert exp.algo_params["seed"] == 7


def test_build_experiment_config_uses_vecnormalize_enable():
    args = SimpleNamespace(
        group=None,
        run_name=None,
        project="proj",
        symbol="AAPL",
        timeframe="1Min",
        warmup_days=5,
        total_timesteps=100,
        eval_freq=10,
        eval_episodes=1,
        normalize=None,
        vecnorm_path=None,
        wandb_log_freq=1000,
        resume=False,
        checkpoint=None,
        sb3_log_interval=None,
        output_dir="models",
    )

    regime = {
        "name": "r0",
        "start": "2024-01-01",
        "end": "2024-01-02",
        "eval_start": "2024-01-01",
        "eval_end": "2024-01-02",
    }

    hp = {"vecnormalize": {"enable": True}}

    exp = build_experiment_config(
        args=args,
        hyperparams=hp,
        regime=regime,
        algo="ppo",
        env_name="vanilla",
        seed=7,
    )

    assert exp.normalize is True


def test_build_experiment_config_allows_algo_vecnormalize_override():
    args = SimpleNamespace(
        group=None,
        run_name=None,
        project="proj",
        symbol="AAPL",
        timeframe="1Min",
        warmup_days=5,
        total_timesteps=100,
        eval_freq=10,
        eval_episodes=1,
        normalize=None,
        vecnorm_path=None,
        wandb_log_freq=1000,
        resume=False,
        checkpoint=None,
        sb3_log_interval=None,
        output_dir="models",
    )

    regime = {
        "name": "r0",
        "start": "2024-01-01",
        "end": "2024-01-02",
        "eval_start": "2024-01-01",
        "eval_end": "2024-01-02",
    }

    hp = {"vecnormalize": {"enable": True}, "ppo": {"vecnormalize": {"enable": False}}}

    exp = build_experiment_config(
        args=args,
        hyperparams=hp,
        regime=regime,
        algo="ppo",
        env_name="vanilla",
        seed=7,
    )

    assert exp.normalize is False
