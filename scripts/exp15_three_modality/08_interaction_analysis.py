"""
Stage 08 — Pairwise modality interaction analysis.

For each video in test holdout computes joint contribution (interaction) terms:
  interaction_qs = gate_q * gate_s   (quality × emotion_static)
  interaction_qt = gate_q * gate_t   (quality × emotion_temporal)
  interaction_st = gate_s * gate_t   (emotion_static × emotion_temporal)

Then correlates each interaction term with prediction correctness (Spearman),
and identifies top-5 videos where specific interactions are strongest or in conflict.

Reads:
  outputs/predictions/test_exp15_predictions.csv

Writes:
  outputs/tables/final_exp15_interaction_pairs.csv
  outputs/stats/final_exp15_modality_redundancy_test.json

Run from project root:
  python scripts/exp15_three_modality/08_interaction_analysis.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from scipy import stats

from utils import get_project_root, load_config, require_file, setup_logger

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()


def main():
    cfg = load_config(str(CONFIG_PATH))
    out_dir = ROOT / cfg["paths"]["output_root"]
    pred_dir = out_dir / "predictions"
    table_dir = out_dir / "tables"
    stats_dir = out_dir / "stats"
    log_dir = out_dir / "logs"
    table_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15_tm.interaction", str(log_dir / "run.log"))
    logger.info("=== Stage 08: Pairwise Modality Interaction Analysis ===")

    test_path = pred_dir / "test_exp15_predictions.csv"
    if not test_path.exists():
        logger.error("test_exp15_predictions.csv not found — run 03_evaluate_test.py first")
        return

    df = pd.read_csv(test_path)

    # Resolve label column
    label_col = "label" if "label" in df.columns else "label_int"
    df["label_int"] = df[label_col].astype(int)
    df["correct"] = ((df["prediction"] >= 0.5).astype(int) == df["label_int"]).astype(int)

    # ── Interaction terms ──────────────────────────────────────────────────────
    df["interaction_qs"] = df["gate_q"] * df["gate_s"]
    df["interaction_qt"] = df["gate_q"] * df["gate_t"]
    df["interaction_st"] = df["gate_s"] * df["gate_t"]

    pairs = [
        ("q × s (quality × emotion_static)", "interaction_qs"),
        ("q × t (quality × emotion_temporal)", "interaction_qt"),
        ("s × t (emotion_static × emotion_temporal)", "interaction_st"),
    ]

    interaction_rows = []
    for pair_name, col in pairs:
        spearman_r, spearman_p = stats.spearmanr(df[col], df["correct"])
        interaction_rows.append({
            "pair": pair_name,
            "mean_interaction": round(float(df[col].mean()), 4),
            "std_interaction": round(float(df[col].std()), 4),
            "min_interaction": round(float(df[col].min()), 4),
            "max_interaction": round(float(df[col].max()), 4),
            "spearman_with_correctness": round(float(spearman_r), 4),
            "p_value": float(spearman_p),
        })
        logger.info(
            f"{pair_name}: mean={df[col].mean():.4f}  "
            f"spearman_r={spearman_r:.4f}  p={spearman_p:.3e}"
        )

    interaction_df = pd.DataFrame(interaction_rows)
    interaction_df.to_csv(table_dir / "final_exp15_interaction_pairs.csv", index=False)
    logger.info(f"Interaction table saved: {table_dir / 'final_exp15_interaction_pairs.csv'}")

    # ── Top-5 highest qt interaction (quality and temporal both contributing) ──
    top5_qt_high = df.nlargest(5, "interaction_qt")[
        ["video_id", "label_int", "forgery_family", "dominant_emotion",
         "prediction", "gate_q", "gate_s", "gate_t", "interaction_qt", "correct"]
    ]

    # ── Top-5 conflict: gate_q high, gate_t low (quality dominates, temporal weak) ──
    df["qt_conflict"] = df["gate_q"] - df["gate_t"]
    top5_qt_conflict = df.nlargest(5, "qt_conflict")[
        ["video_id", "label_int", "forgery_family", "dominant_emotion",
         "prediction", "gate_q", "gate_s", "gate_t", "qt_conflict", "correct"]
    ]

    # ── Top-5 highest st interaction (static × temporal) ─────────────────────
    top5_st_high = df.nlargest(5, "interaction_st")[
        ["video_id", "label_int", "forgery_family", "dominant_emotion",
         "prediction", "gate_q", "gate_s", "gate_t", "interaction_st", "correct"]
    ]

    # ── Modality redundancy test ───────────────────────────────────────────────
    # Test if any pair of modalities is redundant (correlation of gate weights)
    gate_corr = df[["gate_q", "gate_s", "gate_t"]].corr()
    gate_corr_spearman = df[["gate_q", "gate_s", "gate_t"]].corr(method="spearman")

    redundancy = {}
    for a, b in [("gate_q", "gate_s"), ("gate_q", "gate_t"), ("gate_s", "gate_t")]:
        r, p = stats.spearmanr(df[a], df[b])
        redundancy[f"{a}_vs_{b}"] = {
            "spearman_r": round(float(r), 4),
            "p_value": float(p),
            "interpretation": (
                "potentially redundant" if abs(r) > 0.6
                else "moderately correlated" if abs(r) > 0.3
                else "independent"
            ),
        }

    # Also: correlation between interaction and prediction confidence
    pred_conf = np.abs(df["prediction"] - 0.5) * 2  # 0 = uncertain, 1 = confident
    for col_name, col in [("interaction_qt", "interaction_qt"), ("interaction_st", "interaction_st")]:
        r, p = stats.spearmanr(df[col], pred_conf)
        redundancy[f"{col_name}_vs_confidence"] = {
            "spearman_r": round(float(r), 4),
            "p_value": float(p),
        }

    with open(stats_dir / "final_exp15_modality_redundancy_test.json", "w") as f:
        json.dump(redundancy, f, indent=2)

    # ── Console summary ────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("Modality Interaction Analysis")
    print(f"{'='*68}")
    print(interaction_df.to_string(index=False))

    print(f"\nGate weight Spearman correlations:")
    for key, val in redundancy.items():
        if "vs_confidence" not in key:
            print(f"  {key}: r={val['spearman_r']:.4f}  p={val['p_value']:.3e}  → {val['interpretation']}")

    print(f"\nTop-5 highest quality×temporal interaction:")
    print(top5_qt_high.to_string(index=False))

    print(f"\nTop-5 highest static×temporal interaction:")
    print(top5_st_high.to_string(index=False))

    print(f"\nTop-5 quality–temporal conflict (quality dominates, temporal weak):")
    print(top5_qt_conflict.to_string(index=False))
    print(f"{'='*68}")

    logger.info("Stage 08 complete.")


if __name__ == "__main__":
    main()
