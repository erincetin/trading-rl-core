# Trading RL Core

RL finance sandbox with:
- reusable trading environments
- W&B logging (metrics, baselines, artifacts)
- resumable runs and N×M×K matrix sweeps (algos × envs × seeds)

## Setup (uv + Python 3.13)
```bash
uv sync
```

Env vars (for Alpaca fetches):
```
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
```

## Run experiments
`python scripts/runner.py` runs a matrix over algos/envs/seeds and logs to Weights & Biases (TensorBoard is auto-synced to W&B).

Examples:
```bash
# Basic PPO on windowed env, single seed
uv run python scripts/runner.py --algos ppo --envs windowed --seeds 0 --total-timesteps 200000 --eval-freq 10000 --normalize

# Matrix: PPO + A2C on two envs, 3 seeds
uv run python scripts/runner.py \
  --algos ppo,a2c \
  --envs windowed,vanilla \
  --seeds 0,1,2 \
  --total-timesteps 150000 \
  --project trading-rl

# Use local CSV instead of Alpaca
uv run python scripts/runner.py --csv-path data/my_prices.csv --algos ppo --envs vanilla --seeds 0

# Train/eval split (train on early range, eval on later range)
uv run python scripts/runner.py \
  --algos ppo \
  --envs windowed \
  --seeds 0 \
  --total-timesteps 300000 \
  --start 2024-01-01 --end 2024-12-31 \
  --eval-start 2024-10-01 --eval-end 2024-12-31 \
  --timeframe 15Min \
  --normalize

# Resume from a checkpoint (model + VecNormalize stats)
uv run python scripts/runner.py \
  --algos ppo --envs windowed --seeds 0 \
  --resume \
  --checkpoint models/<run_id>/model.zip \
  --vecnorm-path models/<run_id>/vecnormalize.pkl \
  --normalize
```

Key flags:
- `--algos`: comma list (`ppo,a2c,sac,td3`)
- `--envs`: comma list (`windowed,vanilla`)
- `--seeds`: comma list (overrides `--repeats`)
- `--repeats`: integer to auto-generate seeds `range(repeats)`
- `--hyperparams`: path to SB3+env defaults (`trading_rl/config/sb3_finance_hyperparams.yaml`)
- `--normalize/--no-normalize`: enable/disable VecNormalize (if omitted, YAML default is used)
- `--csv-path`: local OHLCV CSV (else Alpaca loader is used)
- `--eval-start/--eval-end`: optional datetime slice for evaluation set; training uses the remaining rows
- `--wandb-log-freq`: reduce per-step logging overhead (TensorBoard still syncs)
- `--sb3-log-interval`: SB3 logger dump interval in episodes (off-policy algos force `1`)

Outputs:
- checkpoints in `models/<wandb_run_id>/` (`model.zip`, `vecnormalize.pkl`, `dataset_manifest_train.json`, `dataset_manifest_eval.json`)
- W&B artifacts named `checkpoint-<run_id>`

## Repo layout
- `trading_rl/envs/`: trading env + windowed wrapper
- `trading_rl/baselines/`: simple baselines (buy/hold, SMA crossover)
- `trading_rl/callbacks/`: W&B training + eval callbacks
- `trading_rl/registry.py`: registry of algos/env builders for the runner
- `scripts/runner.py`: matrix runner + resume support
- `scripts/train.py`: original one-off training example

## Tests
```bash
uv run pytest
```
