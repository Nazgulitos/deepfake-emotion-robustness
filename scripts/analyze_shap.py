"""Exp.09 — SHAP analysis of the XGBoost fusion model.

Reads: datasets/metadata/final_merged_xception_emotion.csv
Writes: outputs/results/YYYY-MM-DD/exp09/
    tables/final_exp09_shap_importance.csv
    tables/final_exp09_shap_importance.tex
    figures/final_exp09_shap_summary.png
    figures/final_exp09_dependence_mean_arousal.png
    run_metadata.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.analysis.shap_helpers import (
    compute_shap_values,
    save_shap_dependence_plot,
    save_shap_summary_plot,
    shap_importance_table,
)
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
    p.add_argument("--exp_id", default="exp09")
    p.add_argument("--subset", default="final", choices=["final"])
    p.add_argument("--merged_table", type=Path,
                   default=Path("datasets/metadata/final_merged_xception_emotion.csv"))
    p.add_argument("--dependence_feature", default="mean_arousal",
                   help="Feature to plot in dependence plot.")
    p.add_argument("--date", default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


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
    logger.info("Starting %s subset=%s", args.exp_id, args.subset)

    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed. Run: uv pip install xgboost  (into the active venv at %s)", sys.executable)
        sys.exit(1)
    try:
        import shap  # noqa: F401
    except ImportError:
        logger.error("shap not installed. Run: uv pip install shap  (into the active venv at %s)", sys.executable)
        logger.error("Active python: %s", sys.executable)
        sys.exit(1)

    df = pd.read_csv(args.merged_table)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(feat_cols)
    if missing:
        logger.warning("Missing feature columns (skipped): %s", sorted(missing))

    clean = df.dropna(subset=feat_cols + ["y"]).copy()
    X = clean[feat_cols]
    y = clean["y"].values

    logger.info("Training XGBoost on %d samples with %d features", len(X), len(feat_cols))
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X, y)

    # SHAP
    shap_values, _ = compute_shap_values(model, X)

    importance_df = shap_importance_table(shap_values, feat_cols)
    csv_path = out_dir / "tables" / f"{args.subset}_exp09_shap_importance.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    importance_df.to_csv(tmp, index=False)
    tmp.rename(csv_path)
    logger.info("Saved importance table → %s", csv_path)

    tex_path = out_dir / "tables" / f"{args.subset}_exp09_shap_importance.tex"
    fmt = importance_df.copy()
    fmt["mean_abs_shap"] = fmt["mean_abs_shap"].map(lambda x: f"{x:.4f}")
    tmp = tex_path.with_suffix(".tex.tmp")
    fmt.to_latex(tmp, index=False, escape=True)
    tmp.rename(tex_path)

    fig_summary = out_dir / "figures" / f"{args.subset}_exp09_shap_summary.png"
    save_shap_summary_plot(shap_values, X, fig_summary)
    logger.info("Saved summary plot → %s", fig_summary)

    dep_feat = args.dependence_feature
    if dep_feat in feat_cols:
        fig_dep = out_dir / "figures" / f"{args.subset}_exp09_dependence_{dep_feat}.png"
        save_shap_dependence_plot(shap_values, X, dep_feat, fig_dep)
        logger.info("Saved dependence plot → %s", fig_dep)
    else:
        logger.warning("Dependence feature '%s' not in feature set — skipped", dep_feat)

    write_run_metadata(
        out_dir, exp_id=args.exp_id, subset=args.subset, seed=SEED,
        cli_args=vars(args), start_time=start_time, end_time=now_utc(),
    )
    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
