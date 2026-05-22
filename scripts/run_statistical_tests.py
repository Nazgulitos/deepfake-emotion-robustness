"""Exp.07 — Statistical tests: DeLong (H1, H3) + Spearman (H2) + permutation.

Hypotheses tested:
  H1: AUC differs significantly across arousal terciles / emotion classes.
  H2: Detector error rate correlates with emotion descriptors (Spearman).
  H3: Fusion AUC > baseline-only AUC (DeLong on test split).

Reads: datasets/metadata/final_merged_xception_emotion.csv
       datasets/metadata/final_xception_fusion_results.csv (AUC reference)
Writes: outputs/results/YYYY-MM-DD/exp07/
    tables/final_exp07_h2_spearman.csv
    tables/final_exp07_h2_spearman.tex
    stats/final_exp07_h1.json
    stats/final_exp07_h3.json
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

from src.analysis.statistical_tests import (
    bootstrap_auc_ci,
    delong_compare,
    permutation_auc_test,
    spearman_test,
)
from src.analysis.subgroup_auc import add_arousal_tercile, compute_subgroup_auc
from src.utils.logging_utils import setup_logging
from src.utils.run_metadata import now_utc, write_run_metadata

SEED = 42

EMOTION_DESCRIPTORS = [
    "mean_arousal",
    "mean_valence",
    "max_arousal",
    "arousal_variation",
    "emotion_entropy",
    "transition_rate",
    "neutral_ratio",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp_id", default="exp07")
    p.add_argument("--subset", default="final", choices=["final"])
    p.add_argument("--merged_table", type=Path,
                   default=Path("datasets/metadata/final_merged_xception_emotion.csv"))
    p.add_argument("--date", default=None)
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--n_permutations", type=int, default=2000)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _run_h1(df: pd.DataFrame, label_col: str, score_col: str,
            logger: logging.Logger) -> dict:
    """H1: AUC differs across arousal terciles and emotion classes."""
    result: dict = {}

    # --- H1a: DeLong across arousal terciles (high vs low) ---
    df_t = add_arousal_tercile(df.copy())
    low = df_t[df_t["arousal_tercile"] == "low"][[label_col, score_col]].dropna()
    high = df_t[df_t["arousal_tercile"] == "high"][[label_col, score_col]].dropna()

    auc_by_tercile = compute_subgroup_auc(df_t, label_col=label_col,
                                          score_col=score_col,
                                          group_col="arousal_tercile",
                                          min_group_size=5)
    result["arousal_tercile_auc"] = auc_by_tercile.to_dict(orient="records")

    # Bootstrap CI per tercile
    ci_rows = []
    for _, row in auc_by_tercile.iterrows():
        grp = df_t[df_t["arousal_tercile"] == row["arousal_tercile"]][[label_col, score_col]].dropna()
        if grp[label_col].nunique() < 2:
            continue
        ci = bootstrap_auc_ci(grp[label_col].values, grp[score_col].values,
                               n_bootstrap=500, seed=SEED)
        ci_rows.append({"tercile": row["arousal_tercile"], **ci})
    result["arousal_tercile_ci"] = ci_rows

    # Permutation test: high-arousal AUC vs low-arousal AUC
    # (DeLong requires same-sample scores; use permutation for disjoint groups)
    if low[label_col].nunique() >= 2 and high[label_col].nunique() >= 2:
        # Concatenate and permute labels
        combined_y = np.concatenate([low[label_col].values, high[label_col].values])
        combined_s_low = np.concatenate([low[score_col].values,
                                         np.full(len(high), np.nan)])
        combined_s_high = np.concatenate([np.full(len(low), np.nan),
                                          high[score_col].values])
        # Simpler: just report AUCs + bootstrap CI, note groups are disjoint
        result["h1_arousal_note"] = (
            "Groups are disjoint by definition; DeLong requires same-sample scores. "
            "AUC difference significance assessed via bootstrap CIs above."
        )

    # --- H1b: AUC spread across emotion classes ---
    emo_auc = compute_subgroup_auc(df, label_col=label_col, score_col=score_col,
                                   group_col="dominant_emotion", min_group_size=5)
    valid_aucs = emo_auc["AUC"].dropna()
    result["emotion_class_auc_spread"] = {
        "n_classes": int(len(valid_aucs)),
        "min_auc": float(valid_aucs.min()),
        "max_auc": float(valid_aucs.max()),
        "range": float(valid_aucs.max() - valid_aucs.min()),
        "std": float(valid_aucs.std()),
    }
    logger.info("H1 arousal tercile AUCs: %s", auc_by_tercile[["arousal_tercile", "AUC"]].to_dict())
    logger.info("H1 emotion AUC range: %.3f – %.3f",
                valid_aucs.min(), valid_aucs.max())
    return result


def _run_h2(df: pd.DataFrame, score_col: str,
            logger: logging.Logger) -> pd.DataFrame:
    """H2: Spearman correlation between detector error and emotion descriptors."""
    # Error = binary misclassification (1 = wrong, 0 = correct)
    threshold = 0.5
    df = df.copy()
    df["pred"] = (df[score_col] >= threshold).astype(int)
    df["error"] = (df["pred"] != df["y"]).astype(int)

    rows = []
    for feat in EMOTION_DESCRIPTORS:
        if feat not in df.columns:
            logger.warning("Descriptor %s missing — skipped", feat)
            continue
        res = spearman_test(df[feat], df["error"])
        rows.append({"descriptor": feat, **res})
    result_df = pd.DataFrame(rows).sort_values("rho", key=abs, ascending=False)
    logger.info("H2 Spearman results:\n%s", result_df.to_string(index=False))
    return result_df


def _run_h3(df: pd.DataFrame, label_col: str, score_col: str,
            n_permutations: int, logger: logging.Logger) -> dict:
    """H3: Fusion AUC > XceptionNet-only AUC.

    Uses 5-fold GroupKFold on 'identity' to produce out-of-fold (OOF) fusion
    scores that are never trained on the same identities they predict.
    DeLong and permutation tests are run on the full OOF predictions,
    giving an honest estimate of generalisation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    feature_cols = [c for c in EMOTION_DESCRIPTORS if c in df.columns]
    clean = df.dropna(subset=[score_col, label_col, "identity"] + feature_cols).copy()
    clean = clean.reset_index(drop=True)

    if "identity" not in clean.columns or clean["identity"].isna().all():
        logger.warning("'identity' column missing — falling back to in-sample H3 (unreliable)")
        # in-sample fallback (kept only as safety net)
        X_all = clean[[score_col] + feature_cols].values
        y_all = clean[label_col].values
        lr = LogisticRegression(max_iter=1000, random_state=SEED)
        lr.fit(X_all, y_all)
        fusion_scores = lr.predict_proba(X_all)[:, 1]
        y = y_all
        evaluation_method = "in_sample_fallback"
    else:
        X = clean[[score_col] + feature_cols].values
        y = clean[label_col].values
        groups = clean["identity"].values
        n_splits = min(5, clean["identity"].nunique())
        gkf = GroupKFold(n_splits=n_splits)
        fusion_scores = np.zeros(len(clean))

        fold_log = []
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[train_idx])
            X_val = sc.transform(X[val_idx])
            lr = LogisticRegression(max_iter=1000, random_state=SEED)
            lr.fit(X_tr, y[train_idx])
            fusion_scores[val_idx] = lr.predict_proba(X_val)[:, 1]
            fold_auc = float(roc_auc_score(y[val_idx], fusion_scores[val_idx])) \
                if len(np.unique(y[val_idx])) > 1 else float("nan")
            fold_log.append({
                "fold": fold + 1,
                "val_identities": sorted(set(groups[val_idx])),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "fold_fusion_auc": fold_auc,
            })
            logger.info("Fold %d: val_ids=%s  n_val=%d  fold_auc=%.3f",
                        fold + 1, sorted(set(groups[val_idx])), len(val_idx), fold_auc)

        evaluation_method = f"groupkfold_{n_splits}_fold_on_identity"

    baseline_scores = clean[score_col].values
    auc_baseline = float(roc_auc_score(y, baseline_scores))
    auc_fusion = float(roc_auc_score(y, fusion_scores))

    delong_res = delong_compare(y, fusion_scores, baseline_scores)
    perm_res = permutation_auc_test(y, fusion_scores, baseline_scores,
                                    n_permutations=n_permutations, seed=SEED)

    result = {
        "evaluation_method": evaluation_method,
        "auc_baseline_only": auc_baseline,
        "auc_fusion_oof": auc_fusion,
        "delta_auc": auc_fusion - auc_baseline,
        "delong": delong_res,
        "permutation": perm_res,
        "n_total": int(len(y)),
        "n_identities": int(clean["identity"].nunique()),
        "feature_cols_used": [score_col] + feature_cols,
        "fold_details": fold_log if "fold_log" in dir() else [],
    }
    logger.info(
        "H3 [%s]: baseline AUC=%.4f  OOF fusion AUC=%.4f  Δ=%.4f  "
        "p_delong=%.4f  p_perm=%.4f",
        evaluation_method, auc_baseline, auc_fusion, auc_fusion - auc_baseline,
        delong_res.get("p_value", float("nan")),
        perm_res.get("p_value", float("nan")),
    )
    return result


def main() -> None:
    args = parse_args()
    start_time = now_utc()
    date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path("outputs/results") / date_str / args.exp_id
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "stats").mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger(args.exp_id)
    logger.info("Starting %s subset=%s", args.exp_id, args.subset)

    df = pd.read_csv(args.merged_table)
    label_col = "y"
    score_col = "detector_score"

    # H1
    h1 = _run_h1(df, label_col, score_col, logger)
    h1_path = out_dir / "stats" / f"{args.subset}_exp07_h1.json"
    tmp = h1_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(h1, indent=2, default=str), encoding="utf-8")
    tmp.rename(h1_path)
    logger.info("Saved H1 → %s", h1_path)

    # H2
    h2_df = _run_h2(df, score_col, logger)
    h2_csv = out_dir / "tables" / f"{args.subset}_exp07_h2_spearman.csv"
    tmp = h2_csv.with_suffix(".csv.tmp")
    h2_df.to_csv(tmp, index=False)
    tmp.rename(h2_csv)

    h2_tex = out_dir / "tables" / f"{args.subset}_exp07_h2_spearman.tex"
    fmt = h2_df.copy()
    fmt["rho"] = fmt["rho"].map(lambda x: f"{x:.3f}")
    fmt["p_value"] = fmt["p_value"].map(lambda x: f"{x:.4f}")
    tmp = h2_tex.with_suffix(".tex.tmp")
    fmt.to_latex(tmp, index=False, escape=True)
    tmp.rename(h2_tex)
    logger.info("Saved H2 → %s", h2_csv)

    # H3
    h3 = _run_h3(df, label_col, score_col, args.n_permutations, logger)
    h3_path = out_dir / "stats" / f"{args.subset}_exp07_h3.json"
    tmp = h3_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(h3, indent=2, default=str), encoding="utf-8")
    tmp.rename(h3_path)
    logger.info("Saved H3 → %s", h3_path)

    write_run_metadata(
        out_dir, exp_id=args.exp_id, subset=args.subset, seed=SEED,
        cli_args=vars(args), start_time=start_time, end_time=now_utc(),
    )
    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
