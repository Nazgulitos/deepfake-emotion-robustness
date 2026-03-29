"""Simple late-fusion baseline using detector score + emotional descriptors."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.linear_model import LogisticRegression


def run_late_fusion(
    table: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
) -> tuple[pd.DataFrame, LogisticRegression]:
    """Train a simple logistic regression on selected features.

    This function trains and predicts on the same table for a lightweight baseline.
    For a final thesis analysis, replace with proper train/val/test protocol.
    """
    missing = set(feature_columns) | {target_column}
    missing = {c for c in missing if c not in table.columns}
    if missing:
        raise ValueError(f"Missing columns for fusion: {sorted(missing)}")

    data = table.dropna(subset=list(feature_columns) + [target_column]).copy()
    if data.empty:
        raise ValueError("No rows available for late fusion after dropna.")

    x = data[list(feature_columns)]
    y = data[target_column].astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(x, y)

    data["fusion_score"] = model.predict_proba(x)[:, 1]
    data["fusion_pred_label"] = (data["fusion_score"] >= 0.5).astype(int)
    return data, model
