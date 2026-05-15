"""Aggregate all experiment outputs into thesis-ready artifacts.

Reads:  outputs/results/YYYY-MM-DD/  (most recent dated folder by default)
Writes: outputs/thesis_artifacts/YYYY-MM-DD/
    all_tables.tex
    all_figures.zip
    results_summary.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.logging_utils import setup_logging
from src.utils.run_metadata import now_utc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default=None,
                   help="Results date folder (YYYY-MM-DD). Default: most recent.")
    p.add_argument("--results_root", type=Path, default=Path("outputs/results"))
    p.add_argument("--out_root", type=Path, default=Path("outputs/thesis_artifacts"))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _find_results_dir(results_root: Path, date: str | None,
                      logger: logging.Logger) -> Path:
    if date:
        d = results_root / date
        if not d.exists():
            raise FileNotFoundError(f"Results folder not found: {d}")
        return d
    candidates = sorted(
        [p for p in results_root.iterdir() if p.is_dir() and p.name[:4].isdigit()],
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No dated results folders found in {results_root}")
    logger.info("Using most recent results folder: %s", candidates[0])
    return candidates[0]


def _collect_tex_tables(results_dir: Path, logger: logging.Logger) -> list[tuple[str, str]]:
    """Return list of (exp_label, tex_content) for all .tex table files."""
    tables = []
    for tex_path in sorted(results_dir.rglob("*.tex")):
        exp_id = tex_path.parts[-3] if len(tex_path.parts) >= 3 else "unknown"
        label = f"{exp_id}/{tex_path.name}"
        content = tex_path.read_text(encoding="utf-8")
        tables.append((label, content))
        logger.info("  + table: %s", label)
    return tables


def _collect_figures(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("*.png"))


def _read_stats_json(results_dir: Path) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for json_path in sorted(results_dir.rglob("*.json")):
        if "metadata" in json_path.name:
            continue
        exp_id = json_path.parts[-3] if len(json_path.parts) >= 3 else "unknown"
        key = f"{exp_id}/{json_path.stem}"
        try:
            stats[key] = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return stats


def _read_csv_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(results_dir.rglob("*.csv")):
        exp_id = csv_path.parts[-3] if len(csv_path.parts) >= 3 else "unknown"
        key = f"{exp_id}/{csv_path.stem}"
        tables[key] = pd.read_csv(csv_path)
    return tables


def _build_latex(tables: list[tuple[str, str]]) -> str:
    parts = [
        r"\documentclass[12pt]{article}",
        r"\usepackage{booktabs,longtable,caption}",
        r"\begin{document}",
        r"\section*{Thesis Experiment Results}",
    ]
    for label, tex in tables:
        parts.append(f"\n% --- {label} ---")
        parts.append(r"\begin{table}[htbp]\centering\small")
        parts.append(tex)
        parts.append(rf"\caption{{{label.replace('_', ' ')}}}")
        parts.append(r"\end{table}")
        parts.append(r"\clearpage")
    parts.append(r"\end{document}")
    return "\n".join(parts)


def _build_summary(csv_tables: dict[str, pd.DataFrame],
                   stats: dict[str, dict]) -> str:
    lines = ["# Results Summary — Chapter 4\n",
             f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"]

    # Exp 05 — per-emotion AUC
    exp05_key = next((k for k in csv_tables if "exp05_per_emotion" in k), None)
    if exp05_key:
        df = csv_tables[exp05_key]
        if "AUC" in df.columns and "dominant_emotion" in df.columns:
            top = df.nlargest(3, "AUC")[["dominant_emotion", "n", "AUC"]]
            bot = df.nsmallest(3, "AUC")[["dominant_emotion", "n", "AUC"]]
            lines += [
                "## Exp.05 — Per-Emotion AUC\n",
                f"Detection AUC varied substantially across emotion classes "
                f"(range: {df['AUC'].min():.3f}–{df['AUC'].max():.3f}). "
                f"Highest detection in: {', '.join(top['dominant_emotion'].tolist())}. "
                f"Lowest detection in: {', '.join(bot['dominant_emotion'].tolist())}.\n",
            ]

    # Exp 06 — forgery × emotion
    exp06_key = next((k for k in csv_tables if "exp06_forgery" in k), None)
    if exp06_key:
        df = csv_tables[exp06_key]
        lines += [
            "## Exp.06 — Forgery × Emotion Cross-Tabulation\n",
            f"Cross-tabulation of {len(df)} (forgery_family, emotion) pairs reveals "
            "heterogeneous detection difficulty. See heatmap figure for full breakdown.\n",
        ]

    # Exp 07 — statistical tests
    h1_key = next((k for k in stats if "exp07_h1" in k), None)
    h3_key = next((k for k in stats if "exp07_h3" in k), None)
    h2_key = next((k for k in csv_tables if "exp07_h2_spearman" in k), None)

    if h1_key:
        h1 = stats[h1_key]
        spread = h1.get("emotion_class_auc_spread", {})
        lines += [
            "## Exp.07 — Statistical Tests\n",
            f"**H1 (Arousal → AUC):** AUC across arousal terciles: "
            f"{[r.get('AUC', 'N/A') for r in h1.get('arousal_tercile_auc', [])]}. ",
            f"Emotion AUC range: {spread.get('min_auc', 'N/A'):.3f}–"
            f"{spread.get('max_auc', 'N/A'):.3f} "
            f"(range={spread.get('range', 'N/A'):.3f}).\n" if isinstance(
                spread.get("min_auc"), float) else "\n",
        ]

    if h2_key:
        df = csv_tables[h2_key]
        if "rho" in df.columns:
            top_feat = df.iloc[0] if len(df) else None
            if top_feat is not None:
                lines.append(
                    f"**H2 (Descriptors → Error):** Strongest Spearman correlation: "
                    f"{top_feat['descriptor']} (ρ={top_feat['rho']:.3f}, "
                    f"p={top_feat['p_value']:.4f}).\n"
                )

    if h3_key:
        h3 = stats[h3_key]
        auc_fusion = h3.get("auc_fusion_oof", h3.get("auc_fusion", float("nan")))
        auc_base = h3.get("auc_baseline_only", float("nan"))
        delta = h3.get("delta_auc", float("nan"))
        p_delong = h3.get("delong", {}).get("p_value", float("nan"))
        eval_method = h3.get("evaluation_method", "")
        lines.append(
            f"**H3 (Fusion vs Baseline):** Baseline AUC={auc_base:.3f}, "
            f"Fusion AUC (OOF)={auc_fusion:.3f} "
            f"(Δ={delta:.3f}). "
            f"DeLong p={p_delong:.4g}. "
            f"Method: {eval_method}.\n"
        )

    # Exp 11 — pilot holdout
    exp11_key = next((k for k in csv_tables if "exp11_holdout" in k), None)
    if exp11_key:
        df = csv_tables[exp11_key]
        lines += [
            "## Exp.11 — Pilot Holdout\n",
            df.to_string(index=False) + "\n",
        ]

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    start_time = now_utc()
    date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = args.out_root / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "build.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger("build_thesis_artifacts")

    results_dir = _find_results_dir(args.results_root, args.date, logger)
    logger.info("Reading results from %s", results_dir)

    # Collect
    tex_tables = _collect_tex_tables(results_dir, logger)
    figures = _collect_figures(results_dir)
    stats = _read_stats_json(results_dir)
    csv_tables = _read_csv_results(results_dir)
    logger.info("Found: %d tex tables, %d figures, %d stat files, %d csv tables",
                len(tex_tables), len(figures), len(stats), len(csv_tables))

    # all_tables.tex
    latex = _build_latex(tex_tables)
    tex_out = out_dir / "all_tables.tex"
    tmp = tex_out.with_suffix(".tex.tmp")
    tmp.write_text(latex, encoding="utf-8")
    tmp.rename(tex_out)
    logger.info("Saved → %s", tex_out)

    # all_figures.zip
    zip_out = out_dir / "all_figures.zip"
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, fig in enumerate(figures, 1):
            exp_part = fig.parts[-3] if len(fig.parts) >= 3 else "misc"
            arcname = f"Fig_{i:02d}_{exp_part}_{fig.name}"
            zf.write(fig, arcname)
    logger.info("Saved → %s (%d figures)", zip_out, len(figures))

    # results_summary.md
    summary = _build_summary(csv_tables, stats)
    md_out = out_dir / "results_summary.md"
    tmp = md_out.with_suffix(".md.tmp")
    tmp.write_text(summary, encoding="utf-8")
    tmp.rename(md_out)
    logger.info("Saved → %s", md_out)

    logger.info("Thesis artifacts complete in %s", out_dir)


if __name__ == "__main__":
    main()
