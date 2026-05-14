"""Exp.11 — Pilot holdout validation.

Trains fusion model on final subset, evaluates zero-shot on pilot subset.
This is the only experiment that touches pilot data for evaluation.

Reads:
  datasets/metadata/final_merged_xception_emotion.csv  (train/val)
  datasets/detector_processed/pilot_detector_scores.csv
  datasets/emotion_annotated/metadata/pilot_video_emotion_features.csv

Writes: outputs/results/YYYY-MM-DD/exp11/
    tables/pilot_exp11_holdout_results.csv
    tables/pilot_exp11_holdout_results.tex
    figures/pilot_exp11_roc.png
    run_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from src.utils.logging_utils import setup_logging
from src.utils.run_metadata import now_utc, write_run_metadata

SEED = 42

FEATURE_COLS = [
    "detector_score",
    "mean_arousal",
    "mean_valence",
    "max_arousal",
    "arousal_variation",
    "emotion_entropy",
    "transition_rate",
    "neutral_ratio",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp_id", default="exp11")
    p.add_argument("--final_merged", type=Path,
                   default=Path("datasets/metadata/final_merged_xception_emotion.csv"))
    p.add_argument("--pilot_scores", type=Path,
                   default=Path("datasets/detector_processed/pilot_detector_scores.csv"))
    p.add_argument("--pilot_emotion", type=Path,
                   default=Path("datasets/emotion_annotated/metadata/pilot_video_emotion_features.csv"))
    p.add_argument("--date", default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _build_pilot_df(scores_path: Path, emotion_path: Path,
                    logger: logging.Logger) -> pd.DataFrame:
    scores = pd.read_csv(scores_path)
    # Keep only emotion-specific columns (+ video_id key) to avoid duplicate columns on merge
    emotion = pd.read_csv(emotion_path)
    shared = set(scores.columns) - {"video_id"}
    emotion_only = [c for c in emotion.columns if c not in shared]
    emotion_slim = emotion[emotion_only]
    merged = scores.merge(emotion_slim, on="video_id", how="inner")
    # Build binary label — always compare as string ('real'/'fake')
    merged["y"] = merged["label"].astype(str).map({"fake": 1, "real": 0}).fillna(0).astype(int)
    logger.info("Pilot merged: %d rows", len(merged))
    return merged


def _save_roc(y_true: np.ndarray, y_score: np.ndarray,
              auc: float, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"Fusion LR (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — Pilot Holdout (Exp.11)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    tmp = out_path.with_name(out_path.stem + ".tmp.png")
    plt.savefig(tmp, dpi=150, bbox_inches="tight")
    plt.close()
    tmp.rename(out_path)


def main() -> None:
    args = parse_args()
    start_time = now_utc()
    date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path("outputs/results") / date_str / args.exp_id
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger(args.exp_id)
    logger.info("Starting %s — pilot holdout validation", args.exp_id)

    # --- Build training data from final subset ---
    final_df = pd.read_csv(args.final_merged)
    feat_cols = [c for c in FEATURE_COLS if c in final_df.columns]
    train_df = final_df.dropna(subset=feat_cols + ["y"]).copy()
    X_train = train_df[feat_cols].values
    y_train = train_df["y"].values
    logger.info("Training on %d final-subset samples", len(train_df))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_train_s, y_train)

    # --- Build pilot test data ---
    pilot_df = _build_pilot_df(args.pilot_scores, args.pilot_emotion, logger)
    feat_cols_pilot = [c for c in feat_cols if c in pilot_df.columns]
    missing_feats = set(feat_cols) - set(feat_cols_pilot)
    if missing_feats:
        logger.warning("Pilot missing features: %s — imputing with 0", sorted(missing_feats))
        for f in missing_feats:
            pilot_df[f] = 0.0

    pilot_clean = pilot_df.dropna(subset=feat_cols + ["y"]).copy()
    X_pilot = scaler.transform(pilot_clean[feat_cols].values)
    y_pilot = pilot_clean["y"].values

    fusion_scores = lr.predict_proba(X_pilot)[:, 1]
    baseline_scores = pilot_clean["detector_score"].values

    # Metrics
    def _metrics(y: np.ndarray, scores: np.ndarray, name: str) -> dict:
        preds = (scores >= 0.5).astype(int)
        return {
            "model": name,
            "AUC": float(roc_auc_score(y, scores)),
            "ACC": float(accuracy_score(y, preds)),
            "F1": float(f1_score(y, preds, zero_division=0)),
            "Precision": float(precision_score(y, preds, zero_division=0)),
            "Recall": float(recall_score(y, preds, zero_division=0)),
            "n": int(len(y)),
        }

    rows = [
        _metrics(y_pilot, baseline_scores, "xception_baseline"),
        _metrics(y_pilot, fusion_scores, "fusion_logreg"),
    ]
    result_df = pd.DataFrame(rows)
    logger.info("Pilot results:\n%s", result_df.to_string(index=False))

    # Save table
    csv_path = out_dir / "tables" / "pilot_exp11_holdout_results.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    result_df.to_csv(tmp, index=False)
    tmp.rename(csv_path)

    tex_path = out_dir / "tables" / "pilot_exp11_holdout_results.tex"
    fmt = result_df.copy()
    for col in ["AUC", "ACC", "F1", "Precision", "Recall"]:
        fmt[col] = fmt[col].map(lambda x: f"{x:.3f}")
    tmp = tex_path.with_suffix(".tex.tmp")
    fmt.to_latex(tmp, index=False, escape=True)
    tmp.rename(tex_path)

    # ROC for fusion
    fig_path = out_dir / "figures" / "pilot_exp11_roc.png"
    _save_roc(y_pilot, fusion_scores, rows[1]["AUC"], fig_path)
    logger.info("Saved ROC → %s", fig_path)

    write_run_metadata(
        out_dir, exp_id=args.exp_id, subset="pilot", seed=SEED,
        cli_args=vars(args), start_time=start_time, end_time=now_utc(),
    )
    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
