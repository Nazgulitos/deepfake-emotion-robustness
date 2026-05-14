"""Tests for src/evaluation/metrics.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import compute_binary_metrics, stratify_by_column


def _make_scores(n: int = 100, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    y_true = pd.Series(rng.integers(0, 2, n))
    y_score = pd.Series(rng.uniform(0, 1, n))
    return y_true, y_score


def test_auroc_matches_sklearn():
    y_true, y_score = _make_scores()
    result = compute_binary_metrics(y_true, y_score)
    expected = roc_auc_score(y_true, y_score)
    assert abs(result["auroc"] - expected) < 1e-9


def test_perfect_classifier():
    y_true = pd.Series([0, 0, 1, 1])
    y_score = pd.Series([0.1, 0.2, 0.8, 0.9])
    result = compute_binary_metrics(y_true, y_score)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["accuracy@0.5"] == pytest.approx(1.0)


def test_all_same_label_returns_nan():
    y_true = pd.Series([1, 1, 1])
    y_score = pd.Series([0.8, 0.9, 0.7])
    result = compute_binary_metrics(y_true, y_score)
    assert np.isnan(result["auroc"])


def test_nan_inputs_handled():
    y_true = pd.Series([0, 1, float("nan")])
    y_score = pd.Series([0.2, 0.8, 0.5])
    result = compute_binary_metrics(y_true, y_score)
    # Two valid rows: one real, one fake — AUC should be defined
    assert not np.isnan(result["auroc"])


def test_stratify_by_column_shape():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "label": rng.integers(0, 2, 60),
        "score": rng.uniform(0, 1, 60),
        "group": np.repeat(["A", "B", "C"], 20),
    })
    result = stratify_by_column(df, "label", "score", "group")
    assert set(result["group_value"]) == {"A", "B", "C"}
    assert "auroc" in result.columns


def test_stratify_missing_column_raises():
    df = pd.DataFrame({"label": [0, 1], "score": [0.2, 0.8]})
    with pytest.raises(ValueError, match="Missing group column"):
        stratify_by_column(df, "label", "score", "nonexistent")
