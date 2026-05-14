"""Exp.06 — Forgery-family × dominant-emotion cross-tabulation.

Reads: datasets/metadata/final_merged_xception_emotion.csv
Writes: outputs/results/YYYY-MM-DD/exp06/
    tables/final_exp06_forgery_emotion.csv
    tables/final_exp06_forgery_emotion.tex
    figures/final_exp06_heatmap.png
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

from src.analysis.subgroup_auc import compute_subgroup_auc
from src.utils.logging_utils import setup_logging
from src.utils.run_metadata import now_utc, write_run_metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp_id", default="exp06")
    p.add_argument("--subset", default="final", choices=["final", "pilot"])
    p.add_argument("--merged_table", type=Path,
                   default=Path("datasets/metadata/final_merged_xception_emotion.csv"))
    p.add_argument("--min_group_size", type=int, default=3)
    p.add_argument("--date", default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _build_auc_pivot(df: pd.DataFrame, label_col: str, score_col: str,
                     family_col: str, emotion_col: str,
                     min_group_size: int) -> pd.DataFrame:
    """Build pivot table: rows=forgery_family, cols=emotion, values=AUC.

    Real videos (family=NaN) are included in every family's pool as the
    negative class — this is required for AUC to be defined, since each
    forgery family contains only fake videos.
    """
    rows = []
    reals = df[df[label_col] == 0]          # all real videos
    families = sorted(df[family_col].dropna().unique())
    emotions = sorted(df[emotion_col].dropna().unique())

    for family in families:
        fakes_this_family = df[df[family_col] == family]
        # Pool = real videos of any emotion + fake videos of this family
        pool = pd.concat([reals, fakes_this_family], ignore_index=True)
        auc_by_emo = compute_subgroup_auc(
            pool, label_col=label_col, score_col=score_col,
            group_col=emotion_col, min_group_size=min_group_size,
        ).set_index(emotion_col)["AUC"]
        row = {"forgery_family": family}
        for emo in emotions:
            row[emo] = auc_by_emo.get(emo, float("nan"))
        rows.append(row)

    return pd.DataFrame(rows).set_index("forgery_family")


def _save_heatmap(pivot: pd.DataFrame, n_pivot: pd.DataFrame,
                  out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Drop all-NaN columns
    pivot_clean = pivot.dropna(axis=1, how="all")

    fig, ax = plt.subplots(figsize=(max(8, len(pivot_clean.columns) * 0.6),
                                    max(3, len(pivot_clean) * 0.8)))
    im = ax.imshow(pivot_clean.values.astype(float), aspect="auto",
                   cmap="RdYlGn", vmin=0.3, vmax=1.0)

    ax.set_xticks(range(len(pivot_clean.columns)))
    ax.set_xticklabels(pivot_clean.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot_clean.index)))
    ax.set_yticklabels(pivot_clean.index, fontsize=9)

    # Annotate cells
    for i, family in enumerate(pivot_clean.index):
        for j, emo in enumerate(pivot_clean.columns):
            val = pivot_clean.loc[family, emo]
            n_val = n_pivot.loc[family, emo] if not np.isnan(val) else ""
            txt = f"{val:.2f}\n(n={int(n_val)})" if not np.isnan(val) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="black" if 0.35 < val < 0.85 else "white")

    plt.colorbar(im, ax=ax, label="AUC")
    ax.set_title("Deepfake Detection AUC: Forgery Family × Dominant Emotion")
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
    logger.info("Starting %s subset=%s", args.exp_id, args.subset)

    df = pd.read_csv(args.merged_table)
    label_col = "y"
    score_col = "detector_score"
    family_col = "manipulation_family"
    emotion_col = "dominant_emotion"

    pivot_auc = _build_auc_pivot(df, label_col, score_col, family_col, emotion_col,
                                  args.min_group_size)

    # Also build n-count pivot for annotation (count of fake videos per cell)
    reals = df[df[label_col] == 0]

    def _n_pivot(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for family in pivot_auc.index:
            fakes = df[df[family_col] == family]
            pool = pd.concat([reals, fakes], ignore_index=True)
            row = {"forgery_family": family}
            for emo in pivot_auc.columns:
                g = pool[pool[emotion_col] == emo]
                row[emo] = len(g) if len(g) >= args.min_group_size else float("nan")
            rows.append(row)
        return pd.DataFrame(rows).set_index("forgery_family")

    pivot_n = _n_pivot(df)

    # Flat CSV (long format)
    long = (pivot_auc.reset_index()
            .melt(id_vars="forgery_family", var_name=emotion_col, value_name="AUC")
            .dropna(subset=["AUC"])
            .sort_values(["forgery_family", "AUC"], ascending=[True, False])
            .reset_index(drop=True))

    csv_path = out_dir / "tables" / f"{args.subset}_exp06_forgery_emotion.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    long.to_csv(tmp, index=False)
    tmp.rename(csv_path)
    logger.info("Saved table → %s", csv_path)

    # LaTeX pivot
    tex_path = out_dir / "tables" / f"{args.subset}_exp06_forgery_emotion_pivot.tex"
    pivot_fmt = pivot_auc.map(lambda x: f"{x:.3f}" if (isinstance(x, float) and not np.isnan(x)) else "—")
    tmp = tex_path.with_suffix(".tex.tmp")
    pivot_fmt.to_latex(tmp, escape=True)
    tmp.rename(tex_path)

    # Heatmap
    fig_path = out_dir / "figures" / f"{args.subset}_exp06_heatmap.png"
    _save_heatmap(pivot_auc, pivot_n, fig_path)
    logger.info("Saved figure → %s", fig_path)

    write_run_metadata(
        out_dir, exp_id=args.exp_id, subset=args.subset, seed=42,
        cli_args=vars(args), start_time=start_time, end_time=now_utc(),
    )
    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
