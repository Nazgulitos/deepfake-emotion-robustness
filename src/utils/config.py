"""Configuration helpers for YAML-driven pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Expected top-level YAML mapping.")
    return data


def get_config_value(config: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """Read nested config value from list of keys.

    Args:
        config: Config mapping.
        keys: Nested keys sequence.
        default: Fallback value.

    Returns:
        Nested value or default.
    """
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
