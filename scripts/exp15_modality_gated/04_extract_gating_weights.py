"""
Stage 04 — Extract and aggregate gating weights from OOF predictions.

Reads:  outputs/predictions/final_exp15_oof_predictions.csv

Writes:
  outputs/tables/final_exp15_gating_per_forgery.csv  (+.tex)
  outputs/tables/final_exp15_gating_per_emotion.csv  (+.tex)
  outputs/tables/final_exp15_per_video_gating.csv    (+.tex)

Run from project root:
  python scripts/exp15_modality_gated/04_extract_gating_weights.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd

from utils import get_project_root, load_config, require_file, set_seeds, setup_logger

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()


def main():
    set_seeds(42)
    cfg = load_config(str(CONFIG_PATH))
    out_dir = ROOT / cfg["paths"]["output_root"]
    pred_dir = out_dir / "predictions"
    table_dir = out_dir / "tables"
    log_dir = out_dir / "logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15.gating", str(log_dir / "run.log"))
    logger.info("=== Stage 04: Extract Gating Weights ===")

    oof_path = require_file(pred_dir / "final_exp15_oof_predictions.csv",
                            "Run 02_train_modality_gated.py first")
    oof = pd.read_csv(oof_path)
    logger.info(f"Loaded {len(oof)} OOF predictions")

    gate_cols = ["gate_det", "gate_emo", "gate_qual"]

    # ----------------------------------------------------------------
    # Per-forgery family
    # ----------------------------------------------------------------
    forgery_grp = (
        oof.groupby("forgery_family")[gate_cols]
        .agg(["mean"])
        .reset_index()
    )
    forgery_grp.columns = ["forgery_family", "mean_gate_det", "mean_gate_emo", "mean_gate_qual"]
    counts = oof.groupby("forgery_family").size().rename("n").reset_index()
    forgery_grp = forgery_grp.merge(counts, on="forgery_family")

    # Dominant modality per family
    forgery_grp["dominant_modality"] = forgery_grp[
        ["mean_gate_det", "mean_gate_emo", "mean_gate_qual"]
    ].idxmax(axis=1).str.replace("mean_gate_", "")

    csv_forg = table_dir / "final_exp15_gating_per_forgery.csv"
    tex_forg = table_dir / "final_exp15_gating_per_forgery.tex"
    forgery_grp.to_csv(csv_forg, index=False)
    forgery_grp.to_latex(tex_forg, index=False, float_format="%.4f")
    logger.info(f"Per-forgery gating saved: {csv_forg}")

    # ----------------------------------------------------------------
    # Per-dominant emotion (n >= 10 filter)
    # ----------------------------------------------------------------
    emo_counts = oof.groupby("dominant_emotion").size().rename("n")
    emo_grp = (
        oof.groupby("dominant_emotion")[gate_cols]
        .mean()
        .join(emo_counts)
        .reset_index()
    )
    emo_grp.columns = ["dominant_emotion", "mean_gate_det", "mean_gate_emo", "mean_gate_qual", "n"]
    emo_grp = emo_grp[emo_grp["n"] >= 10].copy()
    emo_grp["dominant_modality"] = emo_grp[
        ["mean_gate_det", "mean_gate_emo", "mean_gate_qual"]
    ].idxmax(axis=1).str.replace("mean_gate_", "")
    emo_grp = emo_grp.sort_values("mean_gate_emo", ascending=False)

    csv_emo = table_dir / "final_exp15_gating_per_emotion.csv"
    tex_emo = table_dir / "final_exp15_gating_per_emotion.tex"
    emo_grp.to_csv(csv_emo, index=False)
    emo_grp.to_latex(tex_emo, index=False, float_format="%.4f")
    logger.info(f"Per-emotion gating saved: {csv_emo}")

    # ----------------------------------------------------------------
    # Per-video extreme examples
    # ----------------------------------------------------------------
    top_emo = oof.nlargest(10, "gate_emo")[
        ["video_id", "forgery_family", "dominant_emotion", "label",
         "prediction", "gate_det", "gate_emo", "gate_qual"]
    ].copy()
    top_emo["dominant_modality"] = "emotion"

    top_qual = oof.nlargest(10, "gate_qual")[
        ["video_id", "forgery_family", "dominant_emotion", "label",
         "prediction", "gate_det", "gate_emo", "gate_qual"]
    ].copy()
    top_qual["dominant_modality"] = "quality"

    top_det = oof.nlargest(10, "gate_det")[
        ["video_id", "forgery_family", "dominant_emotion", "label",
         "prediction", "gate_det", "gate_emo", "gate_qual"]
    ].copy()
    top_det["dominant_modality"] = "detector"

    per_video = pd.concat([top_det, top_emo, top_qual], ignore_index=True)
    csv_pv = table_dir / "final_exp15_per_video_gating.csv"
    tex_pv = table_dir / "final_exp15_per_video_gating.tex"
    per_video.to_csv(csv_pv, index=False)
    per_video.to_latex(tex_pv, index=False, float_format="%.4f")
    logger.info(f"Per-video gating extremes saved: {csv_pv}")

    # ----------------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------------
    overall_means = oof[gate_cols].mean()
    print("\n=== Gating Weight Summary ===")
    print(f"Overall mean gates:  det={overall_means['gate_det']:.3f}  "
          f"emo={overall_means['gate_emo']:.3f}  qual={overall_means['gate_qual']:.3f}")

    print("\nPer-forgery family:")
    print(forgery_grp.to_string(index=False))

    print("\nPer-emotion (n>=10):")
    print(emo_grp.to_string(index=False))


if __name__ == "__main__":
    main()
