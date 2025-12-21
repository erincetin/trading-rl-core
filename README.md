# Trading RL Core

RL finance sandbox with:
- reusable trading environments
- W&B logging (metrics, baselines, artifacts)
- resumable runs and N×M×K matrix sweeps (algos × envs × seeds)

## Setup (uv + Python 3.13)
```bash
uv sync
```

Heads-up: `uv sync` installs a CPU-only PyTorch build by default. If you want GPU training, install a CUDA-enabled PyTorch wheel in the venv after syncing:
```bash
# Pick the index URL that matches your CUDA runtime (cu118/cu121/cu124)
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Env vars (for Alpaca fetches), set these at `.env`:
```
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
```

## Run experiments
`python scripts/runner.py` runs a matrix over regimes/algos/envs/seeds and logs to Weights & Biases (TensorBoard is auto-synced to W&B).
By default it reads `scripts/config.yaml`, which uses an `experiments:` block per algo (SB3 params + per‑algo env overrides) and a top‑level `regimes:` list.

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

# Run a single regime from scripts/config.yaml
uv run python scripts/runner.py --regime aapl_2024_bull_trainH1_evalQ4
```

Key flags:
- `--algos`: comma list (`ppo,a2c,sac,td3`)
- `--envs`: comma list (`windowed,vanilla`)
- `--seeds`: comma list (overrides `--repeats`)
- `--repeats`: integer to auto-generate seeds `range(repeats)`
- `--hyperparams`: path to YAML hyperparams/regimes (default: `scripts/config.yaml`)
- `--normalize/--no-normalize`: enable/disable VecNormalize (if omitted, YAML default is used)
- `--csv-path`: local OHLCV CSV (else Alpaca loader is used)
- `--eval-start/--eval-end`: optional datetime slice for evaluation set; training uses the remaining rows
- `--regimes`: optional YAML/JSON regimes list (overrides hyperparams regimes)
- `--regime`: run only a single regime by name
- `--wandb-log-freq`: reduce per-step logging overhead (TensorBoard still syncs)
- `--sb3-log-interval`: SB3 logger dump interval in episodes (off-policy algos force `1`)

Outputs:
- checkpoints in `models/<wandb_run_id>/` (`model.zip`, `vecnormalize.pkl`, `dataset_manifest_train.json`, `dataset_manifest_eval.json`)
- W&B artifacts named `checkpoint-<run_id>`

## Repo layout
- `trading_rl/envs/`: trading env + windowed wrapper
- `trading_rl/baselines/`: simple baselines (buy/hold, SMA crossover)
- `trading_rl/callbacks/`: W&B training + eval callbacks
- `trading_rl/data/`: CSV/Alpaca loaders + TA-Lib indicators
- `trading_rl/experiment/`: experiment config, data pipeline, regimes, artifacts, orchestrator
- `trading_rl/config/`: hyperparams loader + helpers
- `trading_rl/registry.py`: registry of algos/env builders for the runner
- `scripts/runner.py`: matrix runner + resume support
- `scripts/config.yaml`: default hyperparams + regimes (`experiments.<algo>.algo` and `experiments.<algo>.env`)
- `scripts/train.py`: original one-off training example
- `scripts/print_tree.py`: quick repo tree output

## Tests
```bash
uv run pytest
```
