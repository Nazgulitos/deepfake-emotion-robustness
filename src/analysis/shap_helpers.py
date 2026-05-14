"""SHAP analysis helpers for the XGBoost fusion model (Exp. 09)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def compute_shap_values(
    model,
    x: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values using the TreeExplainer.

    Args:
        model: Fitted XGBoost or tree-based model.
        x: Feature matrix (rows = samples, cols = features).

    Returns:
        (shap_values, base_values) arrays.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install shap: pip install shap") from exc

    explainer = shap.TreeExplainer(model)
    explanation = explainer(x)
    shap_values = explanation.values
    base_values = explanation.base_values
    return np.array(shap_values), np.array(base_values)


def shap_importance_table(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Convert SHAP values to a feature importance summary table.

    Returns DataFrame sorted descending by mean |SHAP|.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": mean_abs.tolist()})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def save_shap_summary_plot(
    shap_values: np.ndarray,
    x: pd.DataFrame,
    output_path: Path,
    max_display: int = 20,
) -> None:
    """Save a SHAP beeswarm summary plot to output_path."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise ImportError("Install shap and matplotlib.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, x, max_display=max_display, show=False)
    plt.tight_layout()
    tmp = output_path.with_name(output_path.stem + ".tmp.png")
    plt.savefig(tmp, dpi=150, bbox_inches="tight")
    plt.close()
    tmp.rename(output_path)


def save_shap_dependence_plot(
    shap_values: np.ndarray,
    x: pd.DataFrame,
    feature: str,
    output_path: Path,
) -> None:
    """Save a SHAP dependence plot for a single feature."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise ImportError("Install shap and matplotlib.") from exc

    feature_names = list(x.columns)
    feat_idx = feature_names.index(feature)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.dependence_plot(feat_idx, shap_values, x, show=False)
    plt.tight_layout()
    tmp = output_path.with_name(output_path.stem + ".tmp.png")
    plt.savefig(tmp, dpi=150, bbox_inches="tight")
    plt.close()
    tmp.rename(output_path)
