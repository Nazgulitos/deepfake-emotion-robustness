"""Shared utilities: seeds, metrics, logging, plotting helpers."""

import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def hash_config(cfg: dict) -> str:
    serialized = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.md5(serialized).hexdigest()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, log_path: str, level=logging.INFO) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def log_run_metadata(logger: logging.Logger, cfg: dict, config_path: str) -> None:
    import torch
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Config hash: {hash_config(cfg)}")
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        logger.info(f"Git commit: {git_hash}")
    except Exception:
        logger.info("Git commit: unavailable")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[eer_idx] + fnr[eer_idx]) / 2)


def bootstrap_auc_ci(
    y_true: np.ndarray, y_score: np.ndarray, n_iter: int = 2000, seed: int = 42
) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(compute_auc(yt, ys))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def delong_test(y_true: np.ndarray, y_score1: np.ndarray, y_score2: np.ndarray) -> Tuple[float, float]:
    """DeLong's test for comparing two AUCs. Returns (z_stat, p_value)."""
    from scipy import stats

    def _auc_variance(y_true, y_score):
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        n_pos, n_neg = len(pos), len(neg)
        if n_pos == 0 or n_neg == 0:
            return 0.0, 0.0, 0.0
        # Structural components
        v10 = np.array([np.mean(pi > neg) + 0.5 * np.mean(pi == neg) for pi in pos])
        v01 = np.array([np.mean(nj < pos) + 0.5 * np.mean(nj == pos) for nj in neg])
        auc = np.mean(v10)
        s10 = np.var(v10, ddof=1) / n_pos
        s01 = np.var(v01, ddof=1) / n_neg
        var = s10 + s01
        return auc, var, v10, v01

    auc1, var1, v10_1, v01_1 = _auc_variance(y_true, y_score1)
    auc2, var2, v10_2, v01_2 = _auc_variance(y_true, y_score2)

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())

    # Covariance
    cov = (np.cov(v10_1, v10_2)[0, 1] / n_pos +
           np.cov(v01_1, v01_2)[0, 1] / n_neg)

    var_diff = var1 + var2 - 2 * cov
    if var_diff <= 0:
        return 0.0, 1.0

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def permutation_test_auc(
    y_true: np.ndarray,
    y_score1: np.ndarray,
    y_score2: np.ndarray,
    n_iter: int = 10000,
    seed: int = 42,
) -> Dict:
    rng = np.random.RandomState(seed)
    obs_auc1 = compute_auc(y_true, y_score1)
    obs_auc2 = compute_auc(y_true, y_score2)
    obs_diff = obs_auc1 - obs_auc2

    diffs = []
    for _ in range(n_iter):
        mask = rng.rand(len(y_true)) > 0.5
        s1 = np.where(mask, y_score1, y_score2)
        s2 = np.where(mask, y_score2, y_score1)
        diffs.append(compute_auc(y_true, s1) - compute_auc(y_true, s2))

    diffs = np.array(diffs)
    p_value = float(np.mean(np.abs(diffs) >= np.abs(obs_diff)))
    return {
        "auc1": obs_auc1,
        "auc2": obs_auc2,
        "observed_diff": obs_diff,
        "p_value": p_value,
        "n_iter": n_iter,
    }


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def require_file(path: str, desc: str = "") -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Required file not found: {p}"
            + (f" ({desc})" if desc else "")
            + "\nCheck that input data files exist before running this script."
        )
    return p


def get_project_root() -> Path:
    """Locate project root by finding the deepfake-emotion-robustness directory."""
    here = Path(__file__).resolve().parent
    for parent in [here] + list(here.parents):
        if (parent / "datasets").exists():
            return parent
    raise RuntimeError(
        "Cannot locate project root (directory containing 'datasets/'). "
        "Run scripts from within the deepfake-emotion-robustness tree."
    )


def get_output_dir(cfg: dict) -> Path:
    root = get_project_root()
    out = root / cfg["paths"]["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    return out
