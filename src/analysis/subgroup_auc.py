"""Reusable subgroup AUC analysis used by Exp. 04b, 05, 06."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_subgroup_auc(
    df: pd.DataFrame,
    label_col: str,
    score_col: str,
    group_col: str,
    min_group_size: int = 10,
) -> pd.DataFrame:
    """Compute AUC for each stratum of group_col.

    Args:
        df: DataFrame with label, score, and group columns.
        label_col: Binary label column (0/1).
        score_col: Continuous score column (higher = more likely fake).
        group_col: Column to stratify by.
        min_group_size: Groups smaller than this are skipped.

    Returns:
        DataFrame with columns [group_col, 'n', 'n_real', 'n_fake', 'AUC'].
    """
    rows: list[dict] = []
    for group_val, gdf in df.groupby(group_col, dropna=False):
        gdf = gdf.dropna(subset=[label_col, score_col])
        n = len(gdf)
        if n < min_group_size:
            continue
        n_real = int((gdf[label_col] == 0).sum())
        n_fake = int((gdf[label_col] == 1).sum())
        if n_real == 0 or n_fake == 0:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(gdf[label_col].astype(int), gdf[score_col]))
        rows.append({group_col: group_val, "n": n, "n_real": n_real, "n_fake": n_fake, "AUC": auc})
    return pd.DataFrame(rows)


def add_arousal_tercile(
    df: pd.DataFrame,
    arousal_col: str = "mean_arousal",
    tercile_col: str = "arousal_tercile",
) -> pd.DataFrame:
    """Add an arousal tercile label column to df in-place.

    Labels are 'low', 'medium', 'high'.
    """
    out = df.copy()
    out[tercile_col] = pd.qcut(
        out[arousal_col],
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    ).astype(str)
    return out
