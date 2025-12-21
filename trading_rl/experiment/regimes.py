import json
from copy import deepcopy
from pathlib import Path

import yaml


def load_regimes_file(path: str) -> list[dict]:
    """Load regimes from an external YAML/JSON file. Must be list[dict]."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--regimes not found: {p}")

    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    elif p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError("--regimes must be .yaml/.yml or .json")

    if not isinstance(data, list) or not data:
        raise ValueError("--regimes file must contain a non-empty list of regimes")

    _validate_regimes(data)
    return data


def load_regimes_from_hyperparams(hp: dict) -> list[dict]:
    """Load regimes from hyperparams YAML (key: regimes)."""
    regimes = hp.get("regimes", None)
    if regimes is None:
        return []
    if not isinstance(regimes, list) or not regimes:
        raise ValueError("hyperparams 'regimes' must be a non-empty list when present")
    _validate_regimes(regimes)
    return regimes


def _validate_regimes(regimes: list[dict]) -> None:
    required = {"name", "start", "end", "eval_start", "eval_end"}
    for i, r in enumerate(regimes):
        if not isinstance(r, dict):
            raise ValueError(f"Regime[{i}] must be a dict")
        missing = required - set(r.keys())
        if missing:
            raise ValueError(f"Regime[{i}] missing keys: {sorted(missing)}")
        if not isinstance(r["name"], str) or not r["name"]:
            raise ValueError(f"Regime[{i}].name must be non-empty str")


def apply_regime(args, regime: dict):
    """
    Copy args and override date/symbol/timeframe/warmup_days based on regime.
    """
    a = deepcopy(args)

    # required
    a.start = regime["start"]
    a.end = regime["end"]
    a.eval_start = regime["eval_start"]
    a.eval_end = regime["eval_end"]

    # optional overrides
    if "symbol" in regime and regime["symbol"] is not None:
        a.symbol = regime["symbol"]
    if "timeframe" in regime and regime["timeframe"] is not None:
        a.timeframe = regime["timeframe"]
    if "warmup_days" in regime and regime["warmup_days"] is not None:
        a.warmup_days = int(regime["warmup_days"])

    a.regime_name = regime["name"]
    return a


def resolve_regimes(args, hp: dict) -> list[dict]:
    regimes: list[dict] = []

    # 1) hyperparams regimes
    hp_regimes = load_regimes_from_hyperparams(hp)
    if hp_regimes:
        regimes = hp_regimes

    # 2) external regimes file
    elif getattr(args, "regimes", None):
        regimes = load_regimes_file(args.regimes)

    # 3) CLI fallback
    else:
        regimes = [
            {
                "name": "default",
                "start": args.start,
                "end": args.end,
                "eval_start": args.eval_start,
                "eval_end": args.eval_end,
                "symbol": getattr(args, "symbol", None),
                "timeframe": getattr(args, "timeframe", None),
                "warmup_days": getattr(args, "warmup_days", None),
            }
        ]

    # Optional: filter by --regime
    if getattr(args, "regime", None):
        filtered = [r for r in regimes if r["name"] == args.regime]
        if not filtered:
            available = [r["name"] for r in regimes]
            raise ValueError(
                f"--regime '{args.regime}' not found. Available regimes: {available}"
            )
        regimes = filtered

    return regimes
