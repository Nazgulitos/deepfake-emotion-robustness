"""DataFrame read/write helpers with CSV/Parquet support."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_parent(path: Path) -> None:
    """Ensure the parent directory exists for a file path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    """Load a table from CSV or Parquet based on extension."""
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table extension: {path.suffix}")


def write_table(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Write a table to CSV or Parquet based on extension."""
    ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=index)
        return
    raise ValueError(f"Unsupported table extension: {path.suffix}")
