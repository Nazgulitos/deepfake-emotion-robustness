"""Merge stage outputs into a single analysis table."""

from __future__ import annotations

from functools import reduce

import pandas as pd


def merge_on_video_id(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Left-merge all tables on video_id, preserving first table as base."""
    if not tables:
        return pd.DataFrame(columns=["video_id"])

    for idx, table in enumerate(tables, start=1):
        if "video_id" not in table.columns:
            raise ValueError(f"Table #{idx} does not contain 'video_id'.")

    return reduce(lambda left, right: left.merge(right, on="video_id", how="left"), tables)
