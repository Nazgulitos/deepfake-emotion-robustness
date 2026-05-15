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
    p.add_argument("--min_fake", type=int, default=5,
                   help="Minimum number of fake videos required per cell. Cells below this are marked '—'.")
    p.add_argument("--date", default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _build_auc_pivot(df: pd.DataFrame, label_col: str, score_col: str,
                     family_col: str, emotion_col: str,
                     min_fake: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build AUC pivot and n_fake pivot filtered by min_fake per cell.

    Returns (pivot_auc, pivot_nfake) both indexed by forgery_family.

    Only cells where n_fake >= min_fake are populated; others are NaN.
    Real videos are always included as negatives — they are not counted
    in n_fake (which reflects only the fake video count per cell).
    """
    reals = df[df[label_col] == 0]
    families = sorted(df[family_col].dropna().astype(str).unique())
    emotions = sorted(df[emotion_col].dropna().astype(str).unique())

    auc_rows, n_rows = [], []
    for family in families:
        fakes_family = df[df[family_col].astype(str) == family]
        auc_row = {"forgery_family": family}
        n_row = {"forgery_family": family}
        for emo in emotions:
            fakes_cell = fakes_family[fakes_family[emotion_col].astype(str) == emo]
            n_fake = len(fakes_cell)
            n_row[emo] = n_fake
            if n_fake < min_fake:
                auc_row[emo] = float("nan")
                continue
            # Pool = all reals (any emotion) + fakes from this family×emotion cell
            reals_cell = reals[reals[emotion_col].astype(str) == emo]
            pool = pd.concat([reals_cell, fakes_cell], ignore_index=True)
            if pool[label_col].nunique() < 2 or len(pool) < 2:
                auc_row[emo] = float("nan")
                continue
            from sklearn.metrics import roc_auc_score
            try:
                auc_row[emo] = float(roc_auc_score(pool[label_col].values,
                                                    pool[score_col].values))
            except Exception:
                auc_row[emo] = float("nan")
        auc_rows.append(auc_row)
        n_rows.append(n_row)

    pivot_auc = pd.DataFrame(auc_rows).set_index("forgery_family")
    pivot_n = pd.DataFrame(n_rows).set_index("forgery_family")
    return pivot_auc, pivot_n


def _save_heatmap(pivot: pd.DataFrame, n_pivot: pd.DataFrame,
                  out_path: Path, min_fake: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Drop columns where every cell is NaN (no family has enough fakes)
    pivot_clean = pivot.dropna(axis=1, how="all")
    n_clean = n_pivot.reindex(columns=pivot_clean.columns)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot_clean.columns) * 1.1),
                                    max(3, len(pivot_clean) * 1.2)))

    # Mask NaN cells so imshow doesn't colour them
    data = pivot_clean.values.astype(float)
    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#e0e0e0")   # grey for masked cells
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.3, vmax=1.0)

    ax.set_xticks(range(len(pivot_clean.columns)))
    ax.set_xticklabels(pivot_clean.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot_clean.index)))
    ax.set_yticklabels(pivot_clean.index, fontsize=10)

    for i, family in enumerate(pivot_clean.index):
        for j, emo in enumerate(pivot_clean.columns):
            val = pivot_clean.loc[family, emo]
            n_fake = int(n_clean.loc[family, emo]) if not np.isnan(n_clean.loc[family, emo]) else 0
            if np.isnan(val):
                ax.text(j, i, f"—\n(n={n_fake})", ha="center", va="center",
                        fontsize=7, color="#888888")
            else:
                txt_color = "white" if (val < 0.4 or val > 0.85) else "black"
                ax.text(j, i, f"{val:.2f}\n(n={n_fake})", ha="center", va="center",
                        fontsize=8, color=txt_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="AUC (AUROC)", shrink=0.8)
    ax.set_title("Deepfake Detection AUC: Forgery Family × Dominant Emotion\n"
                 f"(n = fake videos per cell; cells with n < {min_fake} shown in grey)",
                 fontsize=10)
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

    pivot_auc, pivot_n = _build_auc_pivot(df, label_col, score_col, family_col,
                                           emotion_col, args.min_fake)
    logger.info("Cells with AUC: %d / %d  (min_fake=%d)",
                int(pivot_auc.notna().sum().sum()),
                int(pivot_auc.size), args.min_fake)

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

    # LaTeX pivot — show AUC where available, n_fake in parentheses, else "—"
    tex_path = out_dir / "tables" / f"{args.subset}_exp06_forgery_emotion_pivot.tex"
    pivot_fmt = pivot_auc.copy().astype(object)
    for family in pivot_auc.index:
        for emo in pivot_auc.columns:
            val = pivot_auc.loc[family, emo]
            n = int(pivot_n.loc[family, emo])
            if isinstance(val, float) and not np.isnan(val):
                pivot_fmt.loc[family, emo] = f"{val:.3f} (n={n})"
            else:
                pivot_fmt.loc[family, emo] = f"— (n={n})"
    tmp = tex_path.with_suffix(".tex.tmp")
    pivot_fmt.to_latex(tmp, escape=True)
    tmp.rename(tex_path)

    # Heatmap
    fig_path = out_dir / "figures" / f"{args.subset}_exp06_heatmap.png"
    _save_heatmap(pivot_auc, pivot_n, fig_path, args.min_fake)
    logger.info("Saved figure → %s", fig_path)

    write_run_metadata(
        out_dir, exp_id=args.exp_id, subset=args.subset, seed=42,
        cli_args=vars(args), start_time=start_time, end_time=now_utc(),
    )
    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
