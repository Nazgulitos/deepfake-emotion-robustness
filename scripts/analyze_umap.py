"""Exp.10 — UMAP projection of the emotion+detector feature space.

Reads: datasets/metadata/final_merged_xception_emotion.csv
Writes: outputs/results/YYYY-MM-DD/exp10/
    figures/final_exp10_umap_by_label.png
    figures/final_exp10_umap_by_emotion.png
    figures/final_exp10_umap_by_forgery.png
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
from sklearn.preprocessing import StandardScaler

from src.analysis.umap_helpers import fit_umap, save_umap_scatter
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
    p.add_argument("--exp_id", default="exp10")
    p.add_argument("--subset", default="final", choices=["final", "pilot"])
    p.add_argument("--merged_table", type=Path,
                   default=Path("datasets/metadata/final_merged_xception_emotion.csv"))
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--date", default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start_time = now_utc()
    date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path("outputs/results") / date_str / args.exp_id
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger(args.exp_id)
    logger.info("Starting %s subset=%s", args.exp_id, args.subset)

    try:
        import umap  # noqa: F401
    except ImportError:
        logger.error("umap-learn not installed. Run: pip install umap-learn")
        sys.exit(1)

    df = pd.read_csv(args.merged_table)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    clean = df.dropna(subset=feat_cols).copy()
    logger.info("Using %d samples, %d features", len(clean), len(feat_cols))

    X = StandardScaler().fit_transform(clean[feat_cols].values)
    embedding = fit_umap(X, n_neighbors=args.n_neighbors,
                         min_dist=args.min_dist, seed=SEED)
    logger.info("UMAP embedding shape: %s", embedding.shape)

    # Plot 1: coloured by real/fake label
    label_series = clean["label"].astype(str)
    save_umap_scatter(
        embedding, label_series,
        out_dir / "figures" / f"{args.subset}_exp10_umap_by_label.png",
        title="UMAP — Real vs Fake",
    )
    logger.info("Saved by-label plot")

    # Plot 2: coloured by dominant emotion
    emotion_series = clean["dominant_emotion"].astype(str)
    save_umap_scatter(
        embedding, emotion_series,
        out_dir / "figures" / f"{args.subset}_exp10_umap_by_emotion.png",
        title="UMAP — Dominant Emotion",
    )
    logger.info("Saved by-emotion plot")

    # Plot 3: coloured by forgery family (real videos shown as 'real')
    forgery_series = clean["manipulation_family"].fillna("real").astype(str)
    save_umap_scatter(
        embedding, forgery_series,
        out_dir / "figures" / f"{args.subset}_exp10_umap_by_forgery.png",
        title="UMAP — Forgery Family",
    )
    logger.info("Saved by-forgery plot")

    write_run_metadata(
        out_dir, exp_id=args.exp_id, subset=args.subset, seed=SEED,
        cli_args=vars(args), start_time=start_time, end_time=now_utc(),
    )
    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
