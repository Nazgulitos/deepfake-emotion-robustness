"""Utilities for reproducible IDs and file names."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1


def build_run_id(stage: str, seed: int) -> str:
    """Create a deterministic-looking run identifier.

    Timestamp is included for traceability and a seed hash for reproducibility.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed_hash = sha1(f"{stage}:{seed}".encode("utf-8")).hexdigest()[:8]
    return f"{stage}_{timestamp}_{seed_hash}"


def make_video_id(source_name: str, index: int) -> str:
    """Generate a stable video identifier pattern for manifests."""
    clean = source_name.lower().replace(" ", "_")
    return f"{clean}_{index:06d}"
