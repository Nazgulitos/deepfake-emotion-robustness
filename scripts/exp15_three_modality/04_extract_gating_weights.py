"""
Stage 04 — Extract and aggregate gating weights.

Groups gate weights by:
  - forgery_family
  - dominant_emotion (n >= 10)
  - arousal tercile (low/mid/high)

Also finds top-10 examples per dominant modality.

Reads:
  outputs/predictions/trainval_oof_predictions.csv
  outputs/predictions/test_exp15_predictions.csv

Writes:
  outputs/tables/final_exp15_gating_per_forgery.csv
  outputs/tables/final_exp15_gating_per_emotion.csv
  outputs/tables/final_exp15_gating_per_arousal_tercile.csv
  outputs/tables/final_exp15_per_video_gating.csv

Run from project root:
  python scripts/exp15_three_modality/04_extract_gating_weights.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from utils import get_project_root, load_config, require_file, setup_logger

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()


def load_combined_predictions(pred_dir: Path) -> pd.DataFrame:
    oof = pd.read_csv(pred_dir / "trainval_oof_predictions.csv")
    test = pd.read_csv(pred_dir / "test_exp15_predictions.csv")

    # Align column names
    if "label" not in oof.columns and "label_int" in oof.columns:
        oof["label"] = oof["label_int"]
    if "label_int" not in test.columns and "label" in test.columns:
        test["label_int"] = test["label"]

    shared = ["video_id", "label_int", "forgery_family", "dominant_emotion",
              "prediction", "gate_q", "gate_s", "gate_t"]
    oof_sub = oof[[c for c in shared if c in oof.columns]].copy()
    test_sub = test[[c for c in shared if c in test.columns]].copy()
    oof_sub["split"] = "oof"
    test_sub["split"] = "test"
    return pd.concat([oof_sub, test_sub], ignore_index=True)


def dominant_modality(row) -> str:
    vals = [row["gate_q"], row["gate_s"], row["gate_t"]]
    names = ["quality", "emotion_static", "emotion_temporal"]
    return names[int(np.argmax(vals))]


def main():
    cfg = load_config(str(CONFIG_PATH))
    out_dir = ROOT / cfg["paths"]["output_root"]
    pred_dir = out_dir / "predictions"
    table_dir = out_dir / "tables"
    log_dir = out_dir / "logs"
    table_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15_tm.gating", str(log_dir / "run.log"))
    logger.info("=== Stage 04: Extract Gating Weights ===")

    require_file(pred_dir / "trainval_oof_predictions.csv", "Run 02_train_three_modality.py")
    require_file(pred_dir / "test_exp15_predictions.csv", "Run 03_evaluate_test.py")

    df = load_combined_predictions(pred_dir)
    logger.info(f"Combined predictions: {len(df)} videos  (oof+test)")
    df["dominant_modality"] = df.apply(dominant_modality, axis=1)

    # ── Per-forgery family ─────────────────────────────────────────────────────
    forgery_rows = []
    for fam, grp in df.groupby("forgery_family"):
        forgery_rows.append({
            "forgery_family": fam,
            "n": len(grp),
            "n_correct": int(((grp["prediction"] >= 0.5) == grp["label_int"]).sum()),
            "accuracy": round(float(((grp["prediction"] >= 0.5) == grp["label_int"]).mean()), 4),
            "mean_gate_q": round(grp["gate_q"].mean(), 4),
            "mean_gate_s": round(grp["gate_s"].mean(), 4),
            "mean_gate_t": round(grp["gate_t"].mean(), 4),
            "std_gate_q": round(grp["gate_q"].std(), 4),
            "std_gate_s": round(grp["gate_s"].std(), 4),
            "std_gate_t": round(grp["gate_t"].std(), 4),
            "dominant_modality": grp["dominant_modality"].mode().iloc[0],
        })
    forgery_df = pd.DataFrame(forgery_rows)
    forgery_df.to_csv(table_dir / "final_exp15_gating_per_forgery.csv", index=False)
    logger.info(f"Per-forgery gating:\n{forgery_df.to_string(index=False)}")

    # ── Per dominant emotion ───────────────────────────────────────────────────
    emo_rows = []
    for emo, grp in df.groupby("dominant_emotion"):
        if len(grp) < 10:
            continue
        emo_rows.append({
            "dominant_emotion": emo,
            "n": len(grp),
            "mean_gate_q": round(grp["gate_q"].mean(), 4),
            "mean_gate_s": round(grp["gate_s"].mean(), 4),
            "mean_gate_t": round(grp["gate_t"].mean(), 4),
            "dominant_modality": grp["dominant_modality"].mode().iloc[0],
        })
    emo_df = pd.DataFrame(emo_rows).sort_values("mean_gate_t", ascending=False)
    emo_df.to_csv(table_dir / "final_exp15_gating_per_emotion.csv", index=False)
    logger.info(f"Per-emotion gating ({len(emo_df)} classes with n>=10):\n{emo_df.to_string(index=False)}")

    # ── Per arousal tercile ────────────────────────────────────────────────────
    # Load arousal from video emotion features to enrich df
    video_emo_path = ROOT / cfg["paths"]["video_emotion"]
    if video_emo_path.exists():
        video_emo = pd.read_csv(video_emo_path)[["video_id", "mean_arousal"]]
        df2 = df.merge(video_emo, on="video_id", how="left")
        if df2["mean_arousal"].notna().sum() > 0:
            df2["arousal_tercile"] = pd.qcut(
                df2["mean_arousal"].fillna(df2["mean_arousal"].median()),
                q=3, labels=["low", "mid", "high"]
            )
            arousal_rows = []
            for tercile, grp in df2.groupby("arousal_tercile", observed=True):
                arousal_rows.append({
                    "arousal_tercile": str(tercile),
                    "n": len(grp),
                    "mean_arousal": round(grp["mean_arousal"].mean(), 4),
                    "mean_gate_q": round(grp["gate_q"].mean(), 4),
                    "mean_gate_s": round(grp["gate_s"].mean(), 4),
                    "mean_gate_t": round(grp["gate_t"].mean(), 4),
                })
            arousal_df = pd.DataFrame(arousal_rows)
            arousal_df.to_csv(table_dir / "final_exp15_gating_per_arousal_tercile.csv", index=False)
            logger.info(f"Per-arousal-tercile gating:\n{arousal_df.to_string(index=False)}")
    else:
        logger.warning("video_emotion not found — skipping arousal tercile analysis")

    # ── Top-10 per dominant modality ───────────────────────────────────────────
    top_rows = []
    for modality, col in [("quality", "gate_q"), ("emotion_static", "gate_s"),
                           ("emotion_temporal", "gate_t")]:
        top10 = df.nlargest(10, col)[
            ["video_id", "label_int", "forgery_family", "dominant_emotion",
             "prediction", "gate_q", "gate_s", "gate_t", "split"]
        ].copy()
        top10["dominant_modality_label"] = modality
        top_rows.append(top10)
    per_video_df = pd.concat(top_rows, ignore_index=True)
    per_video_df.to_csv(table_dir / "final_exp15_per_video_gating.csv", index=False)
    logger.info(f"Per-video top-10 table saved ({len(per_video_df)} rows)")

    # ── Overall gate summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Overall mean gate weights (combined OOF + test):")
    print(f"  quality          = {df['gate_q'].mean():.4f}  ± {df['gate_q'].std():.4f}")
    print(f"  emotion_static   = {df['gate_s'].mean():.4f}  ± {df['gate_s'].std():.4f}")
    print(f"  emotion_temporal = {df['gate_t'].mean():.4f}  ± {df['gate_t'].std():.4f}")
    dom_counts = df["dominant_modality"].value_counts()
    print(f"\nDominant modality distribution:\n{dom_counts.to_string()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
