"""Tests for src/analysis/statistical_tests.py."""

import numpy as np
import pytest

from src.analysis.statistical_tests import (
    bootstrap_auc_ci,
    delong_auc_variance,
    delong_compare,
    permutation_auc_test,
    spearman_test,
)


def _synthetic(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, good_scores, random_scores)."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    good = y_true * 0.6 + rng.uniform(0, 0.4, n)
    rand = rng.uniform(0, 1, n)
    return y_true, good, rand


def _noisy_scores(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, scores) where AUC ~ 0.75 (imperfect, so variance > 0)."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    scores = y_true * 0.4 + rng.uniform(0, 1, n) * 0.6
    return y_true, scores


def test_delong_variance_positive():
    # Use a deliberately noisy (imperfect) classifier so structural components vary.
    y_true, scores = _noisy_scores()
    _, var = delong_auc_variance(y_true, scores)
    assert var > 0


def test_delong_compare_identical_scores_large_p():
    y_true, scores = _noisy_scores()
    result = delong_compare(y_true, scores, scores)
    # Identical scores → difference = 0 → p = 1.0
    assert result["p_value"] == pytest.approx(1.0)


def test_delong_compare_clearly_different_scores_small_p():
    # Use Gaussian noise so classes overlap → AUC < 1.0 → DeLong variance > 0.
    rng = np.random.default_rng(0)
    n = 400
    y_true = rng.integers(0, 2, n)
    good = y_true * 0.5 + rng.normal(0, 0.3, n)   # AUC ~0.87
    random = rng.uniform(0, 1, n)                   # AUC ~0.50
    result = delong_compare(y_true, good, random)
    assert result["p_value"] < 0.05


def test_bootstrap_ci_contains_true_auc():
    from sklearn.metrics import roc_auc_score
    y_true, good, _ = _synthetic(n=300)
    true_auc = roc_auc_score(y_true, good)
    ci = bootstrap_auc_ci(y_true, good, n_bootstrap=500, seed=42)
    assert ci["ci_lower"] <= true_auc <= ci["ci_upper"]


def test_bootstrap_ci_range_sane():
    y_true, good, _ = _synthetic(n=200)
    ci = bootstrap_auc_ci(y_true, good, n_bootstrap=500)
    assert 0.0 <= ci["ci_lower"] <= ci["auc"] <= ci["ci_upper"] <= 1.0


def test_spearman_perfect_monotone():
    x = np.arange(10, dtype=float)
    y = x ** 2
    result = spearman_test(x, y)
    assert result["rho"] == pytest.approx(1.0)


def test_spearman_anti_monotone():
    x = np.arange(10, dtype=float)
    result = spearman_test(x, -x)
    assert result["rho"] == pytest.approx(-1.0)


def test_spearman_ignores_nan():
    x = np.array([1.0, 2.0, float("nan"), 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = spearman_test(x, y)
    assert result["n"] == 3


def test_permutation_test_identical_p_near_1():
    y_true, good, _ = _synthetic(n=100)
    result = permutation_auc_test(y_true, good, good, n_permutations=200, seed=0)
    assert result["p_value"] > 0.05


def test_permutation_test_very_different_p_small():
    rng = np.random.default_rng(3)
    n = 200
    y_true = rng.integers(0, 2, n)
    perfect = y_true.astype(float) + rng.uniform(0, 0.01, n)
    random = rng.uniform(0, 1, n)
    result = permutation_auc_test(y_true, perfect, random, n_permutations=500, seed=3)
    assert result["p_value"] < 0.05
