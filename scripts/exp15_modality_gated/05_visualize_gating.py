"""
Stage 05 — Generate all figures for Exp.15.

Figure 1: Per-forgery modality dominance (stacked horizontal bar)
Figure 2: Per-emotion modality dominance (stacked horizontal bar)
Figure 3: ROC overlay — UCF only vs ModalityGated (+ Exp.12 if available)
Figure 4: Modality dominance examples scatter (top-10 per modality)

Reads:
  outputs/tables/final_exp15_gating_per_forgery.csv
  outputs/tables/final_exp15_gating_per_emotion.csv
  outputs/tables/final_exp15_per_video_gating.csv
  outputs/predictions/final_exp15_oof_predictions.csv
  datasets/detector_processed/final_ucf_scores.csv

Writes:
  outputs/figures/final_exp15_gating_per_forgery.png
  outputs/figures/final_exp15_gating_per_emotion.png
  outputs/figures/final_exp15_modality_dominance_examples.png
  outputs/figures/final_exp15_roc_overlay.png

Run from project root:
  python scripts/exp15_modality_gated/05_visualize_gating.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from utils import compute_auc, get_project_root, load_config, require_file, set_seeds, setup_logger

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()

DET_COLOR = "steelblue"
EMO_COLOR = "darkorange"
QUAL_COLOR = "forestgreen"
DPI = 300


def stacked_bar_gating(df: pd.DataFrame, row_col: str, title: str, out_path: Path) -> None:
    """Draw a horizontal stacked bar chart of [det | emo | qual] gate weights."""
    df = df.sort_values("mean_gate_emo", ascending=True).reset_index(drop=True)
    labels = df[row_col].tolist()
    det_vals = df["mean_gate_det"].values
    emo_vals = df["mean_gate_emo"].values
    qual_vals = df["mean_gate_qual"].values

    y = np.arange(len(labels))
    height = 0.55

    fig, ax = plt.subplots(figsize=(9, max(3, len(labels) * 0.65 + 1.2)))

    bars_d = ax.barh(y, det_vals, height, color=DET_COLOR, label="Detector")
    bars_e = ax.barh(y, emo_vals, height, left=det_vals, color=EMO_COLOR, label="Emotion")
    bars_q = ax.barh(y, qual_vals, height, left=det_vals + emo_vals, color=QUAL_COLOR, label="Quality")

    # Value annotations
    for i, (d, e, q) in enumerate(zip(det_vals, emo_vals, qual_vals)):
        if d > 0.05:
            ax.text(d / 2, i, f"{d:.2f}", va="center", ha="center", fontsize=8, color="white", fontweight="bold")
        if e > 0.05:
            ax.text(d + e / 2, i, f"{e:.2f}", va="center", ha="center", fontsize=8, color="white", fontweight="bold")
        if q > 0.05:
            ax.text(d + e + q / 2, i, f"{q:.2f}", va="center", ha="center", fontsize=8, color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean Gate Weight", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.axvline(x=1 / 3, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # Sample size annotation
    if "n" in df.columns:
        for i, n in enumerate(df["n"].values):
            ax.text(1.01, i, f"n={n}", va="center", fontsize=8, color="gray",
                    transform=ax.get_yaxis_transform())

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def roc_overlay(oof: pd.DataFrame, ucf: pd.DataFrame, out_path: Path,
                exp12_path: Path = None) -> None:
    """ROC curves: UCF only, ModalityGated, optionally Exp.12 UCF+quality."""
    merged = oof[["video_id", "label", "prediction"]].merge(
        ucf[["video_id", "detector_score"]].rename(columns={"detector_score": "ucf_score"}),
        on="video_id", how="inner"
    )
    y = merged["label"].values
    y_mgf = merged["prediction"].values
    y_ucf = merged["ucf_score"].values

    fig, ax = plt.subplots(figsize=(7, 6))

    def _plot_roc(y_true, y_score, label, color, lw=2):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = compute_auc(y_true, y_score)
        ax.plot(fpr, tpr, color=color, lw=lw, label=f"{label} (AUC={auc:.3f})")

    _plot_roc(y, y_ucf, "UCF only", "gray", lw=1.5)

    if exp12_path and exp12_path.exists():
        exp12 = pd.read_csv(exp12_path)
        score_col = next((c for c in ["prediction", "y_score", "pred_proba"] if c in exp12.columns), None)
        if score_col:
            m2 = merged[["video_id", "label"]].merge(
                exp12[["video_id", score_col]].rename(columns={score_col: "ucfq"}),
                on="video_id", how="inner"
            )
            if len(m2) > 10:
                _plot_roc(m2["label"].values, m2["ucfq"].values, "UCF+quality (Exp.12)", "darkorange")

    _plot_roc(y, y_mgf, "ModalityGated (Exp.15)", "steelblue", lw=2.5)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison — Exp.15 ModalityGatedFusion", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"Saved: {out_path}")


def dominance_examples_scatter(per_video: pd.DataFrame, out_path: Path) -> None:
    """Scatter plot of top-10 examples per dominant modality."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    modalities = [("detector", DET_COLOR), ("emotion", EMO_COLOR), ("quality", QUAL_COLOR)]
    gate_map = {"detector": "gate_det", "emotion": "gate_emo", "quality": "gate_qual"}

    for ax, (mod, color) in zip(axes, modalities):
        sub = per_video[per_video["dominant_modality"] == mod].copy()
        if len(sub) == 0:
            ax.set_title(f"Top-10: {mod.capitalize()} dominant\n(no examples)")
            continue
        x = sub["gate_det"].values
        y = sub["gate_emo"].values
        s = (sub["gate_qual"].values * 300 + 30)
        scatter = ax.scatter(x, y, s=s, c=color, alpha=0.75, edgecolors="black", linewidths=0.5)
        # Label fake vs real
        for _, row in sub.iterrows():
            marker = "F" if row["label"] == 1 else "R"
            ax.text(row["gate_det"] + 0.005, row["gate_emo"] + 0.005, marker,
                    fontsize=7, alpha=0.7)
        ax.set_xlabel("Gate weight: Detector", fontsize=10)
        ax.set_ylabel("Gate weight: Emotion", fontsize=10)
        ax.set_title(f"Top-10: {mod.capitalize()} dominant", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.2)

    plt.suptitle("Modality Dominance Examples (bubble size ∝ quality gate)", fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    set_seeds(42)
    cfg = load_config(str(CONFIG_PATH))
    out_dir = ROOT / cfg["paths"]["output_root"]
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    pred_dir = out_dir / "predictions"
    log_dir = out_dir / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15.visualize", str(log_dir / "run.log"))
    logger.info("=== Stage 05: Visualize Gating ===")

    # ----------------------------------------------------------------
    # Figure 1: Per-forgery
    # ----------------------------------------------------------------
    forg_path = require_file(table_dir / "final_exp15_gating_per_forgery.csv",
                             "Run 04_extract_gating_weights.py first")
    forg_df = pd.read_csv(forg_path)
    stacked_bar_gating(
        forg_df, "forgery_family",
        "Modality Contribution by Forgery Family",
        fig_dir / "final_exp15_gating_per_forgery.png",
    )

    # ----------------------------------------------------------------
    # Figure 2: Per-emotion
    # ----------------------------------------------------------------
    emo_path = require_file(table_dir / "final_exp15_gating_per_emotion.csv",
                            "Run 04_extract_gating_weights.py first")
    emo_df = pd.read_csv(emo_path)
    stacked_bar_gating(
        emo_df, "dominant_emotion",
        "Modality Contribution by Dominant Emotion (n≥10)",
        fig_dir / "final_exp15_gating_per_emotion.png",
    )

    # ----------------------------------------------------------------
    # Figure 3: ROC overlay
    # ----------------------------------------------------------------
    oof_path = require_file(pred_dir / "final_exp15_oof_predictions.csv",
                            "Run 02_train_modality_gated.py first")
    oof = pd.read_csv(oof_path)
    ucf = pd.read_csv(require_file(ROOT / cfg["paths"]["ucf_scores"], "UCF scores"))

    exp12_candidates = [
        ROOT / "scripts/exp12_ucf_quality_fusion/outputs/predictions/final_exp12_oof_predictions.csv",
        ROOT / "outputs/results/exp12_oof_predictions.csv",
    ]
    exp12_path = next((p for p in exp12_candidates if p.exists()), None)

    roc_overlay(oof, ucf, fig_dir / "final_exp15_roc_overlay.png", exp12_path=exp12_path)

    # ----------------------------------------------------------------
    # Figure 4: Dominance examples scatter
    # ----------------------------------------------------------------
    pv_path = require_file(table_dir / "final_exp15_per_video_gating.csv",
                           "Run 04_extract_gating_weights.py first")
    per_video = pd.read_csv(pv_path)
    dominance_examples_scatter(per_video, fig_dir / "final_exp15_modality_dominance_examples.png")

    logger.info("All figures saved.")
    print("\nAll figures written to:", fig_dir)


if __name__ == "__main__":
    main()
