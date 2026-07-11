"""
benchmarks/common.py
---------------------
Shared helpers for the Phase 13 experiment suite.
"""

import subprocess
from datetime import datetime, timezone


def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def base_row(seed: int, dataset: str, **extra) -> dict:
    """Every experiment row starts with this — commit, seed, dataset, timestamp."""
    return {
        "git_commit": get_git_commit_hash(),
        "seed": seed,
        "dataset": dataset,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **extra,
    }