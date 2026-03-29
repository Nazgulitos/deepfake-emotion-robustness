"""Metric computation for detector robustness analysis."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def compute_binary_metrics(y_true: pd.Series, y_score: pd.Series) -> dict[str, float]:
    """Compute AUROC, AP, and thresholded accuracy at 0.5."""
    y_true_num = pd.to_numeric(y_true, errors="coerce")
    y_score_num = pd.to_numeric(y_score, errors="coerce")
    valid = y_true_num.notna() & y_score_num.notna()
    if valid.sum() == 0:
        return {"auroc": float("nan"), "average_precision": float("nan"), "accuracy@0.5": float("nan")}

    y_true_final = y_true_num[valid].astype(int)
    y_score_final = y_score_num[valid]
    y_pred = (y_score_final >= 0.5).astype(int)

    if y_true_final.nunique() < 2:
        auroc = float("nan")
        ap = float("nan")
    else:
        auroc = float(roc_auc_score(y_true_final, y_score_final))
        ap = float(average_precision_score(y_true_final, y_score_final))

    acc = float(accuracy_score(y_true_final, y_pred))
    return {"auroc": auroc, "average_precision": ap, "accuracy@0.5": acc}


def stratify_by_column(
    table: pd.DataFrame,
    label_column: str,
    score_column: str,
    group_column: str,
) -> pd.DataFrame:
    """Compute metrics in each group of the provided column."""
    if group_column not in table.columns:
        raise ValueError(f"Missing group column: {group_column}")

    rows: list[dict[str, object]] = []
    for group_name, group in table.groupby(group_column, dropna=False):
        metrics = compute_binary_metrics(group[label_column], group[score_column])
        rows.append(
            {
                "group_column": group_column,
                "group_value": str(group_name),
                "n": int(len(group)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)
