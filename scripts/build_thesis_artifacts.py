"""Aggregate all experiment outputs into thesis-ready artifacts.

Reads:  outputs/results/  — scans ALL dated sub-folders and uses the
        most recent run per experiment (expXX), so re-running one
        experiment never loses results from others.
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logging_utils import setup_logging
from src.utils.run_metadata import now_utc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default=None,
                   help="Output artifact date folder (YYYY-MM-DD). Default: today. "
                        "Input: always scans ALL dated results folders, newest-per-experiment wins.")
    p.add_argument("--results_root", type=Path, default=Path("outputs/results"))
    p.add_argument("--out_root", type=Path, default=Path("outputs/thesis_artifacts"))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _latest_exp_dirs(results_root: Path, logger: logging.Logger) -> dict[str, Path]:
    """Return {exp_id: most-recent-dated-path} by scanning all dated folders.

    Example: if exp05 ran on 2026-05-14 and exp09 ran on 2026-05-15, both
    are returned — the newer date wins per experiment.
    """
    dated_dirs = sorted(
        [p for p in results_root.iterdir() if p.is_dir() and p.name[:4].isdigit()],
        reverse=True,  # newest first
    )
    if not dated_dirs:
        raise FileNotFoundError(f"No dated results folders found in {results_root}")

    latest: dict[str, Path] = {}
    for dated in dated_dirs:
        for exp_dir in dated.iterdir():
            if exp_dir.is_dir() and exp_dir.name not in latest:
                latest[exp_dir.name] = exp_dir
                logger.info("  using %s from %s", exp_dir.name, dated.name)
    return latest


def _collect_tex_tables(exp_dirs: dict[str, Path],
                        logger: logging.Logger) -> list[tuple[str, str]]:
    """Return list of (exp_label, tex_content) for all .tex table files."""
    tables = []
    for exp_id, exp_dir in sorted(exp_dirs.items()):
        for tex_path in sorted(exp_dir.rglob("*.tex")):
            label = f"{exp_id}/{tex_path.name}"
            content = tex_path.read_text(encoding="utf-8")
            tables.append((label, content))
            logger.info("  + table: %s", label)
    return tables


def _collect_figures(exp_dirs: dict[str, Path]) -> list[Path]:
    figures = []
    for exp_dir in sorted(exp_dirs.values()):
        figures.extend(sorted(exp_dir.rglob("*.png")))
    return figures


def _read_stats_json(exp_dirs: dict[str, Path]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for exp_id, exp_dir in sorted(exp_dirs.items()):
        for json_path in sorted(exp_dir.rglob("*.json")):
            if "metadata" in json_path.name:
                continue
            key = f"{exp_id}/{json_path.stem}"
            try:
                stats[key] = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return stats


def _read_csv_results(exp_dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for exp_id, exp_dir in sorted(exp_dirs.items()):
        for csv_path in sorted(exp_dir.rglob("*.csv")):
            key = f"{exp_id}/{csv_path.stem}"
            tables[key] = pd.read_csv(csv_path)
    return tables


def _read_canonical_csvs(root: Path, logger: logging.Logger) -> dict[str, pd.DataFrame]:
    """Read legacy/canonical result CSVs that are not under outputs/results."""
    paths = {
        "exp01/final_huggingface_scores": root / "datasets/detector_processed/final_huggingface_scores.csv",
        "exp02/final_xception_ablation_results": root / "datasets/metadata/final_xception_ablation_results.csv",
        "exp03/final_xception_xgboost_ablation_results": root / "datasets/metadata/final_xception_xgboost_ablation_results.csv",
        "exp04/final_xception_auc_by_arousal": root / "datasets/metadata/final_xception_auc_by_arousal.csv",
        "exp04/final_xception_auc_by_emotion": root / "datasets/metadata/final_xception_auc_by_emotion.csv",
        "exp08/final_ucf_scores": root / "datasets/detector_processed/final_ucf_scores.csv",
        "exp08/pilot_ucf_scores": root / "datasets/detector_processed/pilot_ucf_scores.csv",
    }

    tables: dict[str, pd.DataFrame] = {}
    for key, path in paths.items():
        if path.exists():
            tables[key] = pd.read_csv(path)
            logger.info("  + canonical csv: %s", path)
    return tables


def _binary_metrics(df: pd.DataFrame) -> dict[str, float]:
    y = df["y"] if "y" in df.columns else df["label"].astype(str).map({"fake": 1, "real": 0})
    valid = df.assign(_y=y).dropna(subset=["_y", "detector_score"])
    y_true = valid["_y"].astype(int)
    scores = valid["detector_score"].astype(float)
    preds = (scores >= 0.5).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, scores)),
        "ACC": float(accuracy_score(y_true, preds)),
        "F1": float(f1_score(y_true, preds, zero_division=0)),
        "Precision": float(precision_score(y_true, preds, zero_division=0)),
        "Recall": float(recall_score(y_true, preds, zero_division=0)),
        "n": float(len(valid)),
    }


def _best_row(df: pd.DataFrame, name_col: str) -> pd.Series | None:
    if df.empty or "AUC" not in df.columns or name_col not in df.columns:
        return None
    return df.sort_values("AUC", ascending=False).iloc[0]


def _format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"AUC={metrics['AUC']:.3f}, ACC={metrics['ACC']:.3f}, "
        f"F1={metrics['F1']:.3f}, Precision={metrics['Precision']:.3f}, "
        f"Recall={metrics['Recall']:.3f}, n={int(metrics['n'])}"
    )


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
                   stats: dict[str, dict],
                   exp_dirs: dict[str, Path]) -> str:
    lines = ["# Results Summary — Chapter 4\n",
             f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"]

    # Exp 01 — HuggingFace detector baseline
    exp01_key = next((k for k in csv_tables if "final_huggingface_scores" in k), None)
    if exp01_key:
        metrics = _binary_metrics(csv_tables[exp01_key])
        lines += [
            "## Exp.01 — HuggingFace Detector Baseline\n",
            f"Final-set HuggingFace detector performance: {_format_metrics(metrics)}.\n",
        ]

    # Exp 02 — Xception + logistic regression ablation
    exp02_key = next((k for k in csv_tables if "final_xception_ablation_results" in k), None)
    if exp02_key:
        df = csv_tables[exp02_key]
        best = _best_row(df, "ablation")
        if best is not None:
            lines += [
                "## Exp.02 — Xception + Logistic Regression Ablation\n",
                f"Best ablation: {best['ablation']} "
                f"(AUC={best['AUC']:.3f}, ACC={best['ACC']:.3f}, F1={best['F1']:.3f}).\n",
            ]

    # Exp 03 — Xception + XGBoost ablation
    exp03_key = next((k for k in csv_tables if "final_xception_xgboost_ablation_results" in k), None)
    if exp03_key:
        df = csv_tables[exp03_key]
        best = _best_row(df, "ablation")
        if best is not None:
            lines += [
                "## Exp.03 — Xception + XGBoost Ablation\n",
                f"Best ablation: {best['ablation']} "
                f"(AUC={best['AUC']:.3f}, ACC={best['ACC']:.3f}, F1={best['F1']:.3f}).\n",
            ]

    # Exp 04 — arousal/emotion subgroup robustness
    exp04_arousal_key = next((k for k in csv_tables if "final_xception_auc_by_arousal" in k), None)
    exp04_emotion_key = next((k for k in csv_tables if "final_xception_auc_by_emotion" in k), None)
    if exp04_arousal_key or exp04_emotion_key:
        lines.append("## Exp.04 — Xception Subgroup Robustness\n")
        if exp04_arousal_key:
            df = csv_tables[exp04_arousal_key]
            if {"arousal_bin", "AUC"}.issubset(df.columns):
                high = df.sort_values("AUC", ascending=False).iloc[0]
                low = df.sort_values("AUC", ascending=True).iloc[0]
                lines.append(
                    f"Arousal subgroup AUC ranged from {low['AUC']:.3f} "
                    f"({low['arousal_bin']}) to {high['AUC']:.3f} ({high['arousal_bin']}).\n"
                )
        if exp04_emotion_key:
            df = csv_tables[exp04_emotion_key].dropna(subset=["AUC"])
            if {"dominant_emotion", "AUC"}.issubset(df.columns) and not df.empty:
                high = df.sort_values("AUC", ascending=False).iloc[0]
                low = df.sort_values("AUC", ascending=True).iloc[0]
                lines.append(
                    f"Emotion subgroup AUC ranged from {low['AUC']:.3f} "
                    f"({low['dominant_emotion']}) to {high['AUC']:.3f} "
                    f"({high['dominant_emotion']}).\n"
                )

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

    # Exp 08 — UCF detector
    exp08_final_key = next((k for k in csv_tables if "final_ucf_scores" in k), None)
    exp08_pilot_key = next((k for k in csv_tables if "pilot_ucf_scores" in k), None)
    if exp08_final_key or exp08_pilot_key:
        lines.append("## Exp.08 — UCF Detector (DeepfakeBench)\n")
        if exp08_final_key:
            metrics = _binary_metrics(csv_tables[exp08_final_key])
            lines.append(f"Final-set UCF performance: {_format_metrics(metrics)}.\n")
        if exp08_pilot_key:
            metrics = _binary_metrics(csv_tables[exp08_pilot_key])
            lines.append(f"Pilot-set UCF performance: {_format_metrics(metrics)}.\n")

    # Exp 09 — SHAP / descriptor importance
    exp09_key = next((k for k in csv_tables if "exp09_shap_importance" in k), None)
    if exp09_key:
        df = csv_tables[exp09_key]
        value_col = next(
            (c for c in ["mean_abs_shap", "importance", "importance_share"] if c in df.columns),
            None,
        )
        feature_col = "feature" if "feature" in df.columns else None
        if value_col and feature_col and not df.empty:
            top = df.sort_values(value_col, ascending=False).head(5)
            top_features = ", ".join(
                f"{row[feature_col]} ({row[value_col]:.3f})"
                for _, row in top.iterrows()
            )
            lines += [
                "## Exp.09 — SHAP Feature Importance\n",
                f"Top SHAP-ranked descriptors/features: {top_features}. "
                "See dependence and SHAP summary figures for full feature-level behavior.\n",
            ]

    # Exp 10 — representation visualization
    if "exp10" in exp_dirs:
        lines += [
            "## Exp.10 — Emotion/Detector Representation Visualization\n",
            "UMAP visualizations were generated by label, dominant emotion, and forgery family. "
            "These figures provide qualitative evidence of how emotional descriptors and "
            "forgery categories structure the evaluation space.\n",
        ]

    # Exp 11 — pilot holdout
    exp11_key = next((k for k in csv_tables if "exp11_holdout" in k), None)
    if exp11_key:
        df = csv_tables[exp11_key]
        lines += [
            "## Exp.11 — Pilot Holdout\n",
            df.to_string(index=False) + "\n",
        ]

    def add_fusion_ablation(exp_id: str, detector: str, title: str) -> None:
        final_key = next(
            (k for k in csv_tables if exp_id in k and f"{detector}_fusion_results" in k and "final" in k),
            None,
        )
        pilot_key = next(
            (k for k in csv_tables if exp_id in k and f"{detector}_fusion_results" in k and "pilot" in k),
            None,
        )
        if not final_key and not pilot_key:
            return

        pretty = detector.upper() if detector == "ucf" else detector.capitalize()

        def _best_model_row(df: pd.DataFrame, predicate) -> pd.Series | None:
            subset = df[df["model"].map(predicate)]
            if subset.empty:
                return None
            return subset.sort_values("AUC", ascending=False).iloc[0]

        lines.append(f"## {title}\n")
        if final_key:
            df = csv_tables[final_key].sort_values("AUC", ascending=False)
            best = df.iloc[0]
            base_row = _best_model_row(df, lambda m: m == f"{detector}_only")
            emotion_row = _best_model_row(df, lambda m: m == f"{detector}_emotion_lr")
            quality_only_row = _best_model_row(df, lambda m: str(m).startswith("quality_only"))
            quality_row = _best_model_row(df, lambda m: str(m).startswith(f"{detector}_quality"))
            full_row = _best_model_row(
                df, lambda m: str(m).startswith(f"{detector}_emotion_quality")
            )
            if full_row is None:
                full_row = best

            base_auc = base_row["AUC"] if base_row is not None else float("nan")
            emotion_auc = emotion_row["AUC"] if emotion_row is not None else float("nan")
            quality_only_auc = (
                quality_only_row["AUC"] if quality_only_row is not None else float("nan")
            )
            quality_auc = quality_row["AUC"] if quality_row is not None else float("nan")
            full_auc = full_row["AUC"]
            delta = full_auc - base_auc
            lines.append(
                f"Final OOF best model: {best['model']} "
                f"(AUC={best['AUC']:.3f}, ACC={best['ACC']:.3f}, F1={best['F1']:.3f}); "
                f"delta vs {pretty}-only AUC={delta:.3f}.\n"
            )
            lines.append(
                f"Ablation decomposition: {pretty}-only AUC={base_auc:.3f}; "
                f"{pretty}+emotion AUC={emotion_auc:.3f} "
                f"(delta={emotion_auc - base_auc:.3f}); "
                f"quality-only AUC={quality_only_auc:.3f}; "
                f"{pretty}+quality AUC={quality_auc:.3f} "
                f"(delta={quality_auc - base_auc:.3f}); "
                f"{pretty}+emotion+quality AUC={full_auc:.3f} "
                f"(additional delta over quality={full_auc - quality_auc:.3f}).\n"
            )

            stats_key = next((k for k in stats if exp_id in k and "model_selection" in k), None)
            if stats_key:
                st = stats[stats_key]
                emotion_perm = st.get("permutation_emotion_vs_detector", {})
                controlled_perm = st.get("permutation_emotion_quality_vs_quality", {})
                lines.append(
                    f"Permutation tests: {pretty}+emotion vs {pretty}-only "
                    f"p={emotion_perm.get('p_value', float('nan')):.4f}; "
                    f"{pretty}+emotion+quality vs {pretty}+quality "
                    f"p={controlled_perm.get('p_value', float('nan')):.4f}.\n"
                )

            importance_key = next(
                (
                    k
                    for k in csv_tables
                    if exp_id in k and f"{detector}_quality_feature_importance" in k
                ),
                None,
            )
            if importance_key:
                imp = csv_tables[importance_key].sort_values("importance", ascending=False)
                if not imp.empty:
                    top = imp.iloc[0]
                    lines.append(
                        f"Top quality feature for the tuned quality model: "
                        f"{top['feature']} "
                        f"(importance share={top.get('importance_share', float('nan')):.3f}).\n"
                    )
        if pilot_key:
            df = csv_tables[pilot_key].sort_values("AUC", ascending=False)
            best = df.iloc[0]
            by_model = df.set_index("model")
            base_name = f"{detector}_only"
            base_auc = by_model.loc[base_name, "AUC"] if base_name in by_model.index else float("nan")
            delta = best["AUC"] - base_auc
            lines.append(
                f"Pilot transfer best model: {best['model']} "
                f"(AUC={best['AUC']:.3f}, ACC={best['ACC']:.3f}, F1={best['F1']:.3f}); "
                f"delta vs {pretty}-only AUC={delta:.3f}.\n"
            )

    add_fusion_ablation("exp12", "ucf", "Exp.12 — UCF + Emotion/Quality Fusion")
    add_fusion_ablation("exp13", "huggingface", "Exp.13 — HuggingFace + Emotion/Quality Fusion")
    add_fusion_ablation("exp14", "xception", "Exp.14 — Xception + Emotion/Quality Fusion")

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

    logger.info("Scanning all dated folders in %s (newest wins per experiment)", args.results_root)
    exp_dirs = _latest_exp_dirs(args.results_root, logger)
    logger.info("Found %d experiments: %s", len(exp_dirs), sorted(exp_dirs))

    # Collect
    tex_tables = _collect_tex_tables(exp_dirs, logger)
    figures = _collect_figures(exp_dirs)
    stats = _read_stats_json(exp_dirs)
    csv_tables = _read_csv_results(exp_dirs)
    csv_tables.update(_read_canonical_csvs(ROOT, logger))
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
    summary = _build_summary(csv_tables, stats, exp_dirs)
    md_out = out_dir / "results_summary.md"
    tmp = md_out.with_suffix(".md.tmp")
    tmp.write_text(summary, encoding="utf-8")
    tmp.rename(md_out)
    logger.info("Saved → %s", md_out)

    logger.info("Thesis artifacts complete in %s", out_dir)


if __name__ == "__main__":
    main()
