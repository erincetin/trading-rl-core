# AGENTS.md

Project: Trading RL Core

Purpose
- RL finance sandbox with reusable trading environments, experiment runner, and W&B logging.

Quick Start
- Install deps: `uv sync`
- Install deps with GPU support:
  - `uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130`
  - `uv sync --group cuda`
- Run tests: `uv run pytest`
- Run experiments: `uv run python scripts/runner.py --algos ppo`

Key Paths
- Envs: `trading_rl/envs/`
- Data + indicators: `trading_rl/data/`
- Experiment config + pipeline: `trading_rl/experiment/`
- Registry: `trading_rl/registry.py`
- Runner: `scripts/runner.py`
- Default config: `scripts/config.yaml`

Notes
- Alpaca API keys are read from `.env` (ALPACA_API_KEY, ALPACA_API_SECRET).
- W&B logging is used; run `wandb login`.
