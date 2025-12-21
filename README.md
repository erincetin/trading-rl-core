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

Optional GPU deps (CUDA 12.6 group):
```bash
# Installs the optional group, then swap to CUDA wheels for GPU machines
uv sync --group cuda126
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Env vars (for Alpaca fetches), set these at `.env`:
```
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
```

## Run experiments
`python scripts/runner.py` runs a matrix over regimes/algos/envs/seeds and logs to Weights & Biases (TensorBoard is auto-synced to W&B).
By default it reads `scripts/config.yaml`, which uses:
- `experiments.<algo>.algo`: SB3 params per algorithm
- `experiments.<algo>.env`: per‑algo env overrides
- `experiments.<algo>.run`: per‑algo run overrides (e.g., `total_timesteps`)
- `run`: global run settings (envs, seeds/repeats, eval cadence, logging)
- `regimes`: list of train/eval date slices

Examples:
```bash
# Basic PPO on windowed env, single seed
uv run python scripts/runner.py --algos ppo

# Matrix: PPO + A2C using config.yaml run settings
uv run python scripts/runner.py \
  --algos ppo,a2c \
  --project trading-rl

# Use local CSV instead of Alpaca (set in config.yaml `run.csv_path` or pass once here)
uv run python scripts/runner.py --csv-path data/my_prices.csv --algos ppo

# Train/eval split is defined per regime in config.yaml (eval_start/eval_end)

# Resume from a checkpoint (model + VecNormalize stats), if not set in config.yaml
uv run python scripts/runner.py --algos ppo --resume --checkpoint models/<run_id>/model.zip --vecnorm-path models/<run_id>/vecnormalize.pkl

# Run a single regime from scripts/config.yaml
uv run python scripts/runner.py --regime aapl_2024_bull_trainH1_evalQ4
```

Key flags:
- `--algos`: comma list (`ppo,a2c,sac,td3`)
- `--project`: W&B project name
- `--hyperparams`: path to YAML hyperparams/regimes (default: `scripts/config.yaml`)
- `--regimes`: optional YAML/JSON regimes list (overrides hyperparams regimes)
- `--regime`: run only a single regime by name
- `--csv-path`: local OHLCV CSV (else Alpaca loader is used); overrides config
- `--normalize/--no-normalize`: enable/disable VecNormalize; overrides config

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
