"""Shared utilities for exp15_three_modality."""

import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def hash_config(cfg: dict) -> str:
    keys = [
        "embed_dim", "gate_hidden", "dropout", "batch_size", "n_epochs",
        "patience", "lr", "weight_decay", "optimizer", "n_folds",
        "quality_features", "emotion_static_features", "emotion_temporal_features",
    ]
    subset = {k: cfg[k] for k in keys if k in cfg}
    return hashlib.md5(json.dumps(subset, sort_keys=True).encode()).hexdigest()[:8]


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Paths ──────────────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "datasets").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no 'datasets' dir found)")


def require_file(path: Path, hint: str = "") -> Path:
    if not path.exists():
        msg = f"Required file not found: {path}"
        if hint:
            msg += f"\n  Hint: {hint}"
        raise FileNotFoundError(msg)
    return path


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_iter: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(compute_auc(yt, ys))
    aucs = np.array(aucs)
    lo = float(np.percentile(aucs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(aucs, (1 + ci) / 2 * 100))
    return float(np.mean(aucs)), lo, hi


def delong_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> tuple[float, float]:
    """DeLong's test for comparing two AUCs on the same samples."""
    from scipy import stats

    def _structural_components(y_true, y_score):
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        m, n = len(pos), len(neg)
        v10 = np.array([(np.sum(p > neg) + 0.5 * np.sum(p == neg)) / n for p in pos])
        v01 = np.array([(np.sum(q < pos) + 0.5 * np.sum(q == pos)) / m for q in neg])
        return v10, v01, m, n

    v10_a, v01_a, m, n = _structural_components(y_true, y_score_a)
    v10_b, v01_b, _, _ = _structural_components(y_true, y_score_b)

    auc_a = v10_a.mean()
    auc_b = v10_b.mean()

    s10 = np.cov(v10_a, v10_b)
    s01 = np.cov(v01_a, v01_b)
    var = s10 / m + s01 / n

    diff = auc_a - auc_b
    se = np.sqrt(var[0, 0] + var[1, 1] - 2 * var[0, 1])
    if se == 0:
        return 0.0, 1.0
    z = diff / se
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return float(z), p


def permutation_test_auc(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    n_iter: int = 10000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    obs_delta = compute_auc(y_true, y_score_a) - compute_auc(y_true, y_score_b)
    deltas = []
    for _ in range(n_iter):
        mask = rng.integers(0, 2, size=len(y_true)).astype(bool)
        mixed_a = np.where(mask, y_score_a, y_score_b)
        mixed_b = np.where(mask, y_score_b, y_score_a)
        deltas.append(compute_auc(y_true, mixed_a) - compute_auc(y_true, mixed_b))
    deltas = np.array(deltas)
    p = float(np.mean(np.abs(deltas) >= abs(obs_delta)))
    return {"observed_delta": float(obs_delta), "p_value": p, "n_iter": n_iter}
