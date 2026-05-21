"""
Stage 05 — Visualize gating weights and model performance.

Generates 8 figures saved to outputs/figures/:
  1. final_exp15_gating_per_forgery.png     — stacked bar per forgery family
  2. final_exp15_gating_per_emotion.png     — stacked bar per dominant emotion
  3. final_exp15_gating_per_arousal.png     — stacked bar per arousal tercile
  4. final_exp15_modality_dominance_examples.png  — scatter top-10 per modality
  5. final_exp15_roc_overlay.png            — ROC curves
  6. final_exp15_training_curves.png        — already saved by stage 02
  7. final_exp15_modality_correlation_heatmap.png  — already saved by stage 01
  8. final_exp15_ablation_bars.png          — saved by stage 07

Run from project root:
  python scripts/exp15_three_modality/05_visualize_gating.py
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

from utils import compute_auc, get_project_root, load_config, require_file, setup_logger

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()

MODALITY_COLORS = {
    "quality": "#4C72B0",
    "emotion_static": "#DD8452",
    "emotion_temporal": "#55A868",
}
GATE_COLS = ["mean_gate_q", "mean_gate_s", "mean_gate_t"]
GATE_LABELS = ["Quality", "Emo-Static", "Emo-Temporal"]
GATE_COLORS = [MODALITY_COLORS["quality"], MODALITY_COLORS["emotion_static"],
               MODALITY_COLORS["emotion_temporal"]]


def stacked_bar_plot(df_grouped: pd.DataFrame, label_col: str, gate_cols: list,
                     gate_labels: list, colors: list, title: str, ylabel: str,
                     outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(df_grouped) * 1.2), 5))
    labels = df_grouped[label_col].tolist()
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for i, (col, label, color) in enumerate(zip(gate_cols, gate_labels, colors)):
        vals = df_grouped[col].values
        ax.bar(x, vals, bottom=bottom, label=label, color=color, width=0.6)
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.07:
                ax.text(x[j], b + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean gate weight")
    ax.set_xlabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main():
    cfg = load_config(str(CONFIG_PATH))
    out_dir = ROOT / cfg["paths"]["output_root"]
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    pred_dir = out_dir / "predictions"
    log_dir = out_dir / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15_tm.visualize", str(log_dir / "run.log"))
    logger.info("=== Stage 05: Visualize Gating Weights ===")

    # ── Figure 1: Per-forgery ──────────────────────────────────────────────────
    forgery_path = table_dir / "final_exp15_gating_per_forgery.csv"
    if forgery_path.exists():
        forgery_df = pd.read_csv(forgery_path).dropna(subset=["forgery_family"])
        stacked_bar_plot(
            forgery_df, "forgery_family", GATE_COLS, GATE_LABELS, GATE_COLORS,
            title="Modality gate weights per forgery family",
            ylabel="Forgery family",
            outpath=fig_dir / "final_exp15_gating_per_forgery.png",
        )
        logger.info("Figure 1 saved: gating_per_forgery.png")
    else:
        logger.warning("Per-forgery table missing — run stage 04")

    # ── Figure 2: Per-emotion ──────────────────────────────────────────────────
    emo_path = table_dir / "final_exp15_gating_per_emotion.csv"
    if emo_path.exists():
        emo_df = pd.read_csv(emo_path).dropna(subset=["dominant_emotion"])
        stacked_bar_plot(
            emo_df, "dominant_emotion", GATE_COLS, GATE_LABELS, GATE_COLORS,
            title="Modality gate weights per dominant emotion (n≥10)",
            ylabel="Dominant emotion",
            outpath=fig_dir / "final_exp15_gating_per_emotion.png",
        )
        logger.info("Figure 2 saved: gating_per_emotion.png")
    else:
        logger.warning("Per-emotion table missing — run stage 04")

    # ── Figure 3: Per-arousal tercile ──────────────────────────────────────────
    arousal_path = table_dir / "final_exp15_gating_per_arousal_tercile.csv"
    if arousal_path.exists():
        ar_df = pd.read_csv(arousal_path)
        stacked_bar_plot(
            ar_df, "arousal_tercile", GATE_COLS, GATE_LABELS, GATE_COLORS,
            title="Modality gate weights per arousal tercile",
            ylabel="Arousal tercile",
            outpath=fig_dir / "final_exp15_gating_per_arousal.png",
        )
        logger.info("Figure 3 saved: gating_per_arousal.png")

    # ── Figure 4: Modality dominance scatter ───────────────────────────────────
    per_video_path = table_dir / "final_exp15_per_video_gating.csv"
    if per_video_path.exists():
        pv = pd.read_csv(per_video_path)
        modalities = ["quality", "emotion_static", "emotion_temporal"]
        pairs = [("gate_q", "gate_s"), ("gate_q", "gate_t"), ("gate_s", "gate_t")]
        xlabels = ["gate_quality", "gate_quality", "gate_static"]
        ylabels = ["gate_static", "gate_temporal", "gate_temporal"]
        dominant_labels = ["quality", "quality", "emotion_static"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        dom_labels_list = ["quality", "emotion_static", "emotion_temporal"]
        dom_gate_cols = ["gate_q", "gate_s", "gate_t"]

        for ax_idx, (dom_label, dom_col) in enumerate(zip(dom_labels_list, dom_gate_cols)):
            ax = axes[ax_idx]
            sub = pv[pv["dominant_modality_label"] == dom_label]
            if len(sub) == 0:
                ax.set_title(f"Top-10 {dom_label}-dominant (no data)")
                continue

            # x = dom gate, y = second largest gate
            remaining = [c for c in ["gate_q", "gate_s", "gate_t"] if c != dom_col]
            x_vals = sub[dom_col].values
            y_vals = sub[remaining[0]].values

            # Bubble size = log(prediction+0.01) * 200, clipped to [20, 400]
            bubble_size = np.clip(
                np.log1p(sub["prediction"].values) * 200, 20, 400
            )
            colors = ["#d73027" if l == 1 else "#4575b4" for l in sub["label_int"].values]

            scatter = ax.scatter(x_vals, y_vals, s=bubble_size, c=colors, alpha=0.7,
                                 edgecolors="k", linewidths=0.5)

            for _, row in sub.iterrows():
                fam = str(row.get("forgery_family", "real"))
                ax.annotate(fam[:8], (row[dom_col], row[remaining[0]]),
                            fontsize=6, ha="left", va="bottom", alpha=0.7)

            ax.set_xlabel(dom_col.replace("gate_", "gate "))
            ax.set_ylabel(remaining[0].replace("gate_", "gate "))
            ax.set_title(f"Top-10 {dom_label}-dominant")

            # Legend for color
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#d73027",
                       markersize=8, label="fake"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#4575b4",
                       markersize=8, label="real"),
            ]
            ax.legend(handles=legend_elements, fontsize=7)

        fig.suptitle("Top-10 examples per dominant modality", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_dir / "final_exp15_modality_dominance_examples.png", dpi=300)
        plt.close(fig)
        logger.info("Figure 4 saved: modality_dominance_examples.png")

    # ── Figure 5: ROC overlay ──────────────────────────────────────────────────
    test_pred_path = pred_dir / "test_exp15_predictions.csv"
    oof_pred_path = pred_dir / "trainval_oof_predictions.csv"
    ucf_path = ROOT / cfg["paths"]["ucf_scores"]

    if test_pred_path.exists() and oof_pred_path.exists() and ucf_path.exists():
        test_pred = pd.read_csv(test_pred_path)
        oof_pred = pd.read_csv(oof_pred_path)
        ucf_df = pd.read_csv(ucf_path)[["video_id", "detector_score"]]

        fig, ax = plt.subplots(figsize=(6, 6))

        # OOF ROC
        y_oof = oof_pred["label_int"].values
        p_oof = oof_pred["prediction"].values
        fpr_oof, tpr_oof, _ = roc_curve(y_oof, p_oof)
        auc_oof = compute_auc(y_oof, p_oof)
        ax.plot(fpr_oof, tpr_oof, color="#55A868", lw=2,
                label=f"ThreeModality OOF (AUC={auc_oof:.3f})", linestyle="--")

        # Test ROC
        y_test = test_pred["label"].values if "label" in test_pred.columns else test_pred["label_int"].values
        p_test = test_pred["prediction"].values
        fpr_test, tpr_test, _ = roc_curve(y_test, p_test)
        auc_test = compute_auc(y_test, p_test)
        ax.plot(fpr_test, tpr_test, color="#55A868", lw=2,
                label=f"ThreeModality Test (AUC={auc_test:.3f})")

        # UCF test ROC
        test_label_col = "label" if "label" in test_pred.columns else "label_int"
        test_ucf = test_pred[["video_id", test_label_col]].merge(ucf_df, on="video_id", how="inner")
        if len(test_ucf) > 1:
            fpr_ucf, tpr_ucf, _ = roc_curve(test_ucf[test_label_col].values,
                                              test_ucf["detector_score"].values)
            auc_ucf = compute_auc(test_ucf[test_label_col].values, test_ucf["detector_score"].values)
            ax.plot(fpr_ucf, tpr_ucf, color="#C44E52", lw=2, linestyle=":",
                    label=f"UCF only Test (AUC={auc_ucf:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curves — ThreeModality vs UCF baseline")
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / "final_exp15_roc_overlay.png", dpi=300)
        plt.close(fig)
        logger.info("Figure 5 saved: roc_overlay.png")
    else:
        logger.warning("Prediction files missing — skipping ROC figure")

    logger.info("Stage 05 complete.")
    print("\nFigures saved to:", fig_dir)


if __name__ == "__main__":
    main()
