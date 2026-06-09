"""Write run_metadata.json with git hash, seeds, and CLI arguments."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _package_versions(packages: list[str]) -> dict[str, str]:
    import importlib.metadata as meta
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = meta.version(pkg)
        except meta.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


KEY_PACKAGES = [
    "numpy", "pandas", "scikit-learn", "xgboost",
    "scipy", "umap-learn", "shap", "matplotlib", "seaborn",
]


def write_run_metadata(
    output_dir: Path,
    exp_id: str,
    subset: str,
    seed: int,
    cli_args: dict[str, Any],
    start_time: datetime,
    end_time: datetime | None = None,
) -> Path:
    """Write run_metadata.json into output_dir.

    Args:
        output_dir: Directory where run_metadata.json will be written.
        exp_id: Experiment identifier (e.g. 'exp05').
        subset: Dataset subset ('final' or 'pilot').
        seed: Random seed used.
        cli_args: Parsed CLI arguments as a dict.
        start_time: Experiment start timestamp (UTC).
        end_time: Experiment end timestamp (UTC); None if still running.

    Returns:
        Path to the written metadata file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "exp_id": exp_id,
        "subset": subset,
        "seed": seed,
        "git_commit": _git_commit_hash(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "requirements_txt": "uv.lock",
        "package_versions": _package_versions(KEY_PACKAGES),
        "cli_args": cli_args,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat() if end_time else None,
    }
    out_path = output_dir / "run_metadata.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    tmp_path.rename(out_path)
    return out_path


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
