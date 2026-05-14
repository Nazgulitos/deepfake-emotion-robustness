"""Statistical tests: DeLong AUC comparison, bootstrap CI, Spearman correlation."""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# DeLong AUC comparison (parametric, based on structural components)
# ---------------------------------------------------------------------------

def _auc_structural_components(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (AUC, V10, V01) structural components for DeLong variance."""
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    pos_scores = y_score[pos_mask]
    neg_scores = y_score[neg_mask]

    m, n = len(pos_scores), len(neg_scores)
    if m == 0 or n == 0:
        raise ValueError("DeLong requires at least one positive and one negative sample.")

    # Placement values
    v10 = np.array([np.mean((s > neg_scores) + 0.5 * (s == neg_scores)) for s in pos_scores])
    v01 = np.array([np.mean((pos_scores > s) + 0.5 * (pos_scores == s)) for s in neg_scores])
    auc = float(np.mean(v10))
    return auc, v10, v01


def delong_auc_variance(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Compute AUC and its DeLong variance for a single classifier.

    Returns:
        (auc, variance)
    """
    auc, v10, v01 = _auc_structural_components(y_true, y_score)
    m, n = len(v10), len(v01)
    var = (np.var(v10, ddof=1) / m) + (np.var(v01, ddof=1) / n)
    return auc, var


def delong_compare(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> dict[str, float]:
    """Two-tailed DeLong test comparing AUC(A) vs AUC(B) on the same samples.

    Returns dict with keys: auc_a, auc_b, z_stat, p_value.
    """
    auc_a, v10_a, v01_a = _auc_structural_components(y_true, y_score_a)
    auc_b, v10_b, v01_b = _auc_structural_components(y_true, y_score_b)

    m, n = len(v10_a), len(v01_a)

    s_10 = np.cov(v10_a, v10_b, ddof=1)
    s_01 = np.cov(v01_a, v01_b, ddof=1)

    var_diff = s_10[0, 0] / m + s_01[0, 0] / n - 2 * (s_10[0, 1] / m + s_01[0, 1] / n)

    diff = float(auc_a - auc_b)

    if abs(diff) < 1e-12 and var_diff <= 1e-14:
        # Identical classifiers — difference and variance both zero.
        return {"auc_a": float(auc_a), "auc_b": float(auc_b), "z_stat": 0.0, "p_value": 1.0}

    if var_diff <= 1e-14:
        # Covariance collapsed (e.g. one classifier is perfect on one class).
        # Fall back to conservative uncorrelated variance estimate.
        var_a, _ = delong_auc_variance(y_true, y_score_a)
        var_b, _ = delong_auc_variance(y_true, y_score_b)
        var_diff = var_a + var_b
        if var_diff <= 0:
            warnings.warn("Cannot estimate DeLong variance; returning p=NaN.")
            return {"auc_a": float(auc_a), "auc_b": float(auc_b), "z_stat": float("nan"), "p_value": float("nan")}

    z = diff / np.sqrt(var_diff)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(auc_a), "auc_b": float(auc_b), "z_stat": float(z), "p_value": p}


# ---------------------------------------------------------------------------
# Bootstrap AUC confidence interval
# ---------------------------------------------------------------------------

def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap percentile CI for AUC.

    Returns dict with keys: auc, ci_lower, ci_upper.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aucs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aucs.append(float(roc_auc_score(yt, ys)))

    auc = float(roc_auc_score(y_true, y_score))
    lo = float(np.percentile(boot_aucs, 100 * alpha / 2))
    hi = float(np.percentile(boot_aucs, 100 * (1 - alpha / 2)))
    return {"auc": auc, "ci_lower": lo, "ci_upper": hi, "n_bootstrap": n_bootstrap}


# ---------------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------------

def spearman_test(
    x: Sequence[float] | np.ndarray | pd.Series,
    y: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute Spearman rank correlation and two-tailed p-value.

    Returns dict with keys: rho, p_value, n.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean, y_clean = x_arr[mask], y_arr[mask]
    rho, pval = stats.spearmanr(x_clean, y_clean)
    return {"rho": float(rho), "p_value": float(pval), "n": int(mask.sum())}


# ---------------------------------------------------------------------------
# Permutation test for AUC difference
# ---------------------------------------------------------------------------

def permutation_auc_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Non-parametric permutation test for AUC difference (AUC_A - AUC_B).

    Returns dict with keys: observed_diff, p_value, n_permutations.
    """
    rng = np.random.default_rng(seed)
    obs_a = float(roc_auc_score(y_true, y_score_a))
    obs_b = float(roc_auc_score(y_true, y_score_b))
    obs_diff = obs_a - obs_b

    count = 0
    for _ in range(n_permutations):
        swap = rng.random(len(y_true)) < 0.5
        perm_a = np.where(swap, y_score_b, y_score_a)
        perm_b = np.where(swap, y_score_a, y_score_b)
        if len(np.unique(y_true)) < 2:
            continue
        diff = float(roc_auc_score(y_true, perm_a)) - float(roc_auc_score(y_true, perm_b))
        if abs(diff) >= abs(obs_diff):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return {
        "auc_a": obs_a,
        "auc_b": obs_b,
        "observed_diff": obs_diff,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }
