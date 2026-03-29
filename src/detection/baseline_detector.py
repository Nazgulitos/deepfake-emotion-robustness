"""Baseline deepfake detector scaffolding."""

from __future__ import annotations

import pandas as pd


def run_baseline_detector(
    subset_manifest: pd.DataFrame,
    model_name: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Create a detector score table with placeholder values.

    TODO: Integrate pretrained detector inference.
    """
    rows: list[dict[str, object]] = []

    for _, item in subset_manifest.iterrows():
        score = pd.NA
        rows.append(
            {
                "video_id": str(item["video_id"]),
                "model_name": model_name,
                "detector_score": score,
                "predicted_label": pd.NA,
                "threshold": threshold,
                "status": "todo_infer",
            }
        )

    return pd.DataFrame(rows)
