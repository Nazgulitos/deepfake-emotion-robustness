"""Exp.05 — Per-emotion-class subgroup AUC analysis.

Reads: datasets/metadata/final_merged_xception_emotion.csv
Writes: outputs/results/YYYY-MM-DD/exp05/
    tables/final_exp05_per_emotion_auc.csv
    tables/final_exp05_per_emotion_auc.tex
    figures/final_exp05_emotion_auc_bars.png
    stats/final_exp05_delong.json
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

from src.analysis.statistical_tests import delong_compare
from src.analysis.subgroup_auc import compute_subgroup_auc
from src.utils.logging_utils import setup_logging
from src.utils.run_metadata import now_utc, write_run_metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp_id", default="exp05")
    p.add_argument("--subset", default="final", choices=["final", "pilot"])
    p.add_argument("--merged_table", type=Path,
                   default=Path("datasets/metadata/final_merged_xception_emotion.csv"))
    p.add_argument("--min_group_size", type=int, default=5)
    p.add_argument("--date", default=None,
                   help="Output date folder override (YYYY-MM-DD). Default: today.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _save_bar_chart(df: pd.DataFrame, out_path: Path, group_col: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_df = df.sort_values("AUC", ascending=True)
    colors = ["#d62728" if auc < 0.5 else "#1f77b4" for auc in sorted_df["AUC"]]

    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_df) * 0.4)))
    bars = ax.barh(sorted_df[group_col].astype(str), sorted_df["AUC"], color=colors)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (AUC=0.5)")
    ax.set_xlabel("AUC (AUROC)")
    ax.set_title("Deepfake Detection AUC by Dominant Emotion Class")
    ax.set_xlim(0, 1)

    for bar, n in zip(bars, sorted_df["n"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"n={n}", va="center", fontsize=7)

    ax.legend()
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
    (out_dir / "stats").mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger(args.exp_id)
    logger.info("Starting %s subset=%s", args.exp_id, args.subset)

    df = pd.read_csv(args.merged_table)
    # label: 'real'/'fake' string → binary int via 'y' column
    label_col = "y"
    score_col = "detector_score"
    group_col = "dominant_emotion"

    # --- Subgroup AUC ---
    result = compute_subgroup_auc(df, label_col=label_col, score_col=score_col,
                                  group_col=group_col, min_group_size=args.min_group_size)
    result = result.sort_values("AUC", ascending=False).reset_index(drop=True)
    logger.info("Computed AUC for %d emotion classes", len(result))

    # Tables
    csv_path = out_dir / "tables" / f"{args.subset}_exp05_per_emotion_auc.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    result.to_csv(tmp, index=False)
    tmp.rename(csv_path)
    logger.info("Saved table → %s", csv_path)

    tex_path = out_dir / "tables" / f"{args.subset}_exp05_per_emotion_auc.tex"
    tex = result.rename(columns={"dominant_emotion": "Emotion", "n": "N",
                                  "n_real": "N Real", "n_fake": "N Fake",
                                  "AUC": "AUC"})
    tex["AUC"] = tex["AUC"].map(lambda x: f"{x:.3f}" if not np.isnan(x) else "—")
    tmp = tex_path.with_suffix(".tex.tmp")
    tex.to_latex(tmp, index=False, escape=True)
    tmp.rename(tex_path)

    # Figure
    fig_path = out_dir / "figures" / f"{args.subset}_exp05_emotion_auc_bars.png"
    _save_bar_chart(result.dropna(subset=["AUC"]), fig_path, group_col)
    logger.info("Saved figure → %s", fig_path)

    # Bootstrap CI per emotion class (DeLong not applicable — groups are disjoint partitions)
    # Each video belongs to exactly one dominant_emotion, so there are no shared video IDs
    # between groups. Bootstrap CI per group is the correct uncertainty measure here.
    from src.analysis.statistical_tests import bootstrap_auc_ci

    valid = result.dropna(subset=["AUC"])
    delong_out: dict = {
        "method": "bootstrap_ci_per_class",
        "note": (
            "Emotion classes are disjoint video partitions — DeLong comparison is not applicable. "
            "95% bootstrap CIs are reported per class instead."
        ),
        "per_class_ci": [],
    }
    for _, row in valid.iterrows():
        grp = df[df[group_col] == row[group_col]][[label_col, score_col]].dropna()
        if grp[label_col].nunique() < 2 or len(grp) < 5:
            continue
        ci = bootstrap_auc_ci(
            grp[label_col].values.astype(int), grp[score_col].values,
            n_bootstrap=1000, seed=42,
        )
        delong_out["per_class_ci"].append({
            "emotion": row[group_col],
            "n": int(row["n"]),
            **ci,
        })
    logger.info("Computed bootstrap CIs for %d emotion classes",
                len(delong_out["per_class_ci"]))

    stats_path = out_dir / "stats" / f"{args.subset}_exp05_delong.json"
    tmp = stats_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(delong_out, indent=2, default=str), encoding="utf-8")
    tmp.rename(stats_path)
    logger.info("Saved stats → %s", stats_path)

    write_run_metadata(
        out_dir, exp_id=args.exp_id, subset=args.subset, seed=42,
        cli_args=vars(args), start_time=start_time, end_time=now_utc(),
    )
    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
