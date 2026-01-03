from __future__ import annotations

import argparse
import os
from typing import Iterable

from trading_rl.config.hyperparams import load_hyperparams


def _get_total_memory_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().total) / (1024**3)
    except Exception:
        pass

    try:
        if os.name == "nt":
            import ctypes

            class _MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint),
                    ("dwMemoryLoad", ctypes.c_uint),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUS()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return float(stat.ullTotalPhys) / (1024**3)

        if hasattr(os, "sysconf"):
            pagesize = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            return float(pagesize * pages) / (1024**3)
    except Exception:
        return None

    return None


def _get_gpu_total_mem_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return float(props.total_memory) / (1024**3)
    except Exception:
        return None


def _collect_algo_devices(args) -> list[str]:
    devices = []
    hp = args.hyperparams_data or {}
    shared = hp.get("shared", {}) or {}
    for algo in args.algos or []:
        algo_cfg = hp.get(str(algo).lower(), {}) or {}
        device = algo_cfg.get("device", shared.get("device"))
        if device is not None:
            devices.append(str(device).lower())
    return devices


def _parse_list(arg: str | Iterable[str]) -> list[str]:
    if isinstance(arg, str):
        return [x for x in arg.split(",") if x]
    return list(arg)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Estimate parallelism for experiments.")
    parser.add_argument(
        "--hyperparams",
        type=str,
        default=str(os.path.join("scripts", "config.yaml")),
        help="Path to hyperparams YAML.",
    )
    parser.add_argument(
        "--algos",
        type=str,
        default="ppo",
        help="Comma-separated algo names.",
    )
    return parser.parse_args(argv)


def estimate_parallelism(
    args,
    combos: Iterable[dict] | None = None,
    *,
    cpu_count: int | None = None,
    total_mem_gb: float | None = None,
    gpu_total_mem_gb: float | None = None,
) -> dict:
    cpu_total = cpu_count or (os.cpu_count() or 1)
    algos = args.algos or []

    weights = []
    for algo in algos:
        weights.append(1)

    per_run_weight = max(weights) if weights else 1
    max_by_cpu = max(1, cpu_total // per_run_weight)

    run_cfg = (args.hyperparams_data or {}).get("run", {}) or {}
    mem_per_run_gb = float(run_cfg.get("memory_per_run_gb", 2.0))
    gpu_mem_per_run_gb = float(run_cfg.get("gpu_mem_per_run_gb", 4.0))

    total_mem_gb = total_mem_gb if total_mem_gb is not None else _get_total_memory_gb()
    if total_mem_gb is None or mem_per_run_gb <= 0:
        max_by_mem = None
    else:
        max_by_mem = max(1, int(total_mem_gb // mem_per_run_gb))

    devices = _collect_algo_devices(args)
    gpu_total_mem_gb = (
        gpu_total_mem_gb
        if gpu_total_mem_gb is not None
        else _get_gpu_total_mem_gb()
    )
    if gpu_total_mem_gb is None or gpu_mem_per_run_gb <= 0:
        max_by_gpu = None
    else:
        uses_gpu = any(d in {"cuda", "auto"} for d in devices)
        max_by_gpu = max(1, int(gpu_total_mem_gb // gpu_mem_per_run_gb)) if uses_gpu else None

    combos_len = len(combos) if combos is not None else None
    limits = [max_by_cpu]
    if max_by_mem is not None:
        limits.append(max_by_mem)
    if max_by_gpu is not None:
        limits.append(max_by_gpu)
    suggested = min(limits)
    if combos_len is not None:
        suggested = min(suggested, combos_len)

    return {
        "cpu_count": cpu_total,
        "per_run_weight": per_run_weight,
        "max_parallel_by_cpu": max_by_cpu,
        "total_mem_gb": total_mem_gb,
        "mem_per_run_gb": mem_per_run_gb,
        "max_parallel_by_mem": max_by_mem,
        "gpu_total_mem_gb": gpu_total_mem_gb,
        "gpu_mem_per_run_gb": gpu_mem_per_run_gb,
        "max_parallel_by_gpu": max_by_gpu,
        "suggested_parallel": suggested,
        "combos": combos_len,
        "devices": devices,
    }


def print_parallel_estimate(estimate: dict) -> None:
    print("Parallelism estimate:")
    print(f"  cpu_count: {estimate['cpu_count']}")
    print(f"  per_run_weight: {estimate['per_run_weight']}")
    print(f"  max_parallel_by_cpu: {estimate['max_parallel_by_cpu']}")
    if estimate["total_mem_gb"] is not None:
        print(f"  total_mem_gb: {estimate['total_mem_gb']:.1f}")
        print(f"  mem_per_run_gb: {estimate['mem_per_run_gb']:.1f}")
        print(f"  max_parallel_by_mem: {estimate['max_parallel_by_mem']}")
    else:
        print("  total_mem_gb: unknown")
    if estimate["gpu_total_mem_gb"] is not None:
        print(f"  gpu_total_mem_gb: {estimate['gpu_total_mem_gb']:.1f}")
        print(f"  gpu_mem_per_run_gb: {estimate['gpu_mem_per_run_gb']:.1f}")
        print(f"  max_parallel_by_gpu: {estimate['max_parallel_by_gpu']}")
    else:
        print("  gpu_total_mem_gb: unknown")
    if estimate["combos"] is not None:
        print(f"  combos: {estimate['combos']}")
    print(f"  suggested_parallel: {estimate['suggested_parallel']}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.hyperparams_data = (
        load_hyperparams(args.hyperparams) if args.hyperparams else {}
    )
    args.algos = _parse_list(args.algos)
    estimate = estimate_parallelism(args)
    print_parallel_estimate(estimate)


if __name__ == "__main__":
    main()
