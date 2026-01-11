# Trading RL Core

A reproducible SB3-based trading RL sandbox (Gymnasium envs + W&B logging + sweep runner) for quickly testing trading environments, features, and seed sensitivity.

What you get:
- Reusable trading environments (vanilla + windowed episodes)
- Weights & Biases logging (metrics, baselines, checkpoints, artifacts)
- Resumable runs and matrix sweeps (algos × regimes × seeds), with deterministic eval

---

## Quickstart (CPU)

```bash
uv sync
uv run python scripts/runner.py --algos ppo
```

Then open your W&B workspace/project to inspect eval curves and artifacts.

---

## Setup

### Requirements
- Python 3.13
- `uv`
- (Optional) GPU + CUDA runtime if you want GPU training
- (Optional) Alpaca credentials if you want to fetch data from Alpaca

### Install (CPU-only, default)

```bash
uv sync
```

Heads-up: `uv sync` will typically install a CPU-only PyTorch build unless you explicitly install CUDA wheels.

### GPU setup (optional)

If you want GPU training, install a CUDA-enabled PyTorch wheel inside the venv after syncing. Pick the index URL that matches your CUDA runtime (example below uses cu130):

```bash
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

If your system CUDA is different (e.g., cu121/cu124/cu128), switch the index URL accordingly.

---

## Data: Alpaca vs CSV

### Alpaca (optional)
Create a `.env` file at the repo root:

```bash
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
APCA_API_BASE_URL=https://paper-api.alpaca.markets
```

Notes:
- `APCA_API_BASE_URL` is Alpaca’s standard env var name; keep it unquoted.
- If you use live trading endpoints, change the base URL accordingly.

### CSV mode (no credentials)
You can run entirely from local OHLCV CSVs (recommended for reproducibility). Use either:
- `run.csv_path` in `scripts/config.yaml`, or
- pass `--csv-path` at runtime

---

## Running experiments

`scripts/runner.py` runs a matrix across regimes × algos × seeds and logs everything to Weights & Biases. (TensorBoard logs are also synced to W&B.)

By default it reads `scripts/config.yaml`, which contains:
- `experiments.<algo>.algo`: SB3 params per algorithm
- `experiments.<algo>.env`: per-algo env overrides
- `experiments.<algo>.run`: per-algo run overrides (e.g., `total_timesteps`)
- `run`: global run settings (seeds, eval cadence, logging settings)
- `regimes`: named train/eval date slices
- `env.action_transform`: `identity` or `symmetric`
  - `symmetric` maps `[-1, 1] -> [0, max_position]` (useful for on-policy algos producing symmetric actions)

### Examples

```bash
# Basic PPO on windowed env, single seed (uses scripts/config.yaml defaults)
uv run python scripts/runner.py --algos ppo

# Matrix: PPO + A2C using config.yaml run settings
uv run python scripts/runner.py --algos ppo,a2c --project trading-rl

# Use local CSV instead of Alpaca (overrides config)
uv run python scripts/runner.py --csv-path data/my_prices.csv --algos ppo

# Run a single regime by name (as defined in scripts/config.yaml)
uv run python scripts/runner.py --regime btcusd_15m_train2022_eval2023 --algos ppo

# Resume from a checkpoint (model + VecNormalize stats)
uv run python scripts/runner.py \
  --algos ppo \
  --resume \
  --checkpoint models/<wandb_run_id>/model.zip \
  --vecnorm-path models/<wandb_run_id>/vecnormalize.pkl

# Run a reproduction script, should take around 1 hour per run, for a total of 4 hours with 8 parallelism.
uv run python scripts/runner.py \
  --algos sac,td3,ppo,a2c \
  --project trading-rl \
  --config scripts/configs/config.yaml
  --parallel 8

# Run a reproduction script in Linux
MP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 uv run python scripts/runner.py \
  --algos sac,td3,ppo,a2c \
  --project trading-rl \
  --config scripts/configs/config.yaml
  --parallel 8
```

### Key flags
- `--algos`: comma-separated list (`ppo,a2c,sac,td3`)
- `--project`: W&B project name
- `--parallel`: max experiments to run concurrently
- `--parallel-delay`: delay (seconds) between launching parallel experiments
- `--hyperparams`: YAML config path (default: `scripts/config.yaml`)
- `--regime`: run only a single regime by name (from config)
- `--regimes`: provide an explicit regimes list (YAML/JSON) overriding config
- `--csv-path`: local OHLCV CSV; overrides config and bypasses Alpaca loader
- `--normalize/--no-normalize`: enable/disable VecNormalize; overrides config

---

## Metrics & evaluation (what to expect)

- Evaluation is deterministic by default (`n_eval_episodes=1`, `n_envs_eval=1`) on a fixed eval segment.
- Primary metric is typically excess return vs buy-and-hold (logged in W&B summary if enabled).
- Additional diagnostics may include raw return, max drawdown, Sharpe, trade count, and turnover (depending on your logging config).

---

## Outputs

Artifacts and files are written per W&B run:

- Checkpoints: `models/<wandb_run_id>/`
  - `model.zip`
  - `vecnormalize.pkl` (if VecNormalize enabled)
  - `dataset_manifest_train.json`
  - `dataset_manifest_eval.json`
- W&B artifacts:
  - checkpoint artifacts uploaded under names like `checkpoint-<run_id>`

---

## Repo layout

- `trading_rl/envs/`: trading env + windowed wrapper
- `trading_rl/baselines/`: simple baselines (buy-and-hold, SMA crossover)
- `trading_rl/callbacks/`: W&B training + eval callbacks
- `trading_rl/data/`: CSV/Alpaca loaders + TA-Lib indicators
- `trading_rl/experiment/`: experiment config, data pipeline, regimes, artifacts, orchestrator
- `trading_rl/config/`: hyperparams loader + helpers
- `trading_rl/registry.py`: registry of algos/env builders for the runner
- `scripts/runner.py`: matrix runner + resume support
- `scripts/config.yaml`: default hyperparams + regimes
- `scripts/train.py`: original one-off training example
- `scripts/print_tree.py`: quick repo tree output

---

## Tests

```bash
uv run pytest
```

---

## Parallelism estimate

```bash
uv run python scripts/parallel_estimator.py --algos sac
```

---

## Common issues

- **GPU not used / slow training:** you probably installed CPU-only torch. Install CUDA wheels (see GPU setup).
- **Slow on Linux Machine:** Before running commands set `MP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` to reduce unnecessary parallelism of helper Linux processes.
- **Alpaca auth errors:** double-check `.env` keys and `APCA_API_BASE_URL`.
- **Resume fails:** you must provide both `model.zip` and the matching `vecnormalize.pkl` (if normalization was enabled).
- **Weird trade counts / turnover:** confirm your action thresholding and turnover definition in your metrics code (log raw action stats when debugging).
- **Crash during training:** parallelism might be too high for your setup.