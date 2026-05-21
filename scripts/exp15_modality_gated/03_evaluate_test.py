"""
Stage 03 — Evaluate OOF predictions: AUC, CI, DeLong, permutation tests.

Reads:
  outputs/predictions/final_exp15_oof_predictions.csv
  datasets/detector_processed/final_ucf_scores.csv   (baseline)

Writes:
  outputs/tables/final_exp15_results.csv  (+.tex)
  outputs/stats/final_exp15_delong_vs_ucf_only.json
  outputs/stats/final_exp15_delong_vs_ucf_quality.json   (if available)
  outputs/stats/final_exp15_permutation_tests.json

Run from project root:
  python scripts/exp15_modality_gated/03_evaluate_test.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils import (
    bootstrap_auc_ci,
    compute_auc,
    compute_eer,
    delong_test,
    get_output_dir,
    get_project_root,
    load_config,
    permutation_test_auc,
    require_file,
    set_seeds,
    setup_logger,
)

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def main():
    set_seeds(42)
    cfg = load_config(str(CONFIG_PATH))
    out_dir = ROOT / cfg["paths"]["output_root"]
    pred_dir = out_dir / "predictions"
    table_dir = out_dir / "tables"
    stats_dir = out_dir / "stats"
    log_dir = out_dir / "logs"
    for d in [table_dir, stats_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15.evaluate", str(log_dir / "run.log"))
    logger.info("=== Stage 03: Evaluate OOF Predictions ===")

    # ----------------------------------------------------------------
    # Load OOF predictions
    # ----------------------------------------------------------------
    oof_path = require_file(pred_dir / "final_exp15_oof_predictions.csv",
                            "Run 02_train_modality_gated.py first")
    oof = pd.read_csv(oof_path)
    y_true = oof["label"].values
    y_score = oof["prediction"].values
    n = len(oof)
    logger.info(f"OOF predictions loaded: {n} samples")

    # ----------------------------------------------------------------
    # Main metrics
    # ----------------------------------------------------------------
    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(y_true, y_score, n_iter=2000, seed=42)
    eer = compute_eer(y_true, y_score)
    thresh_met = threshold_metrics(y_true, y_score)

    logger.info(f"AUC={auc_mean:.4f} 95%CI=[{auc_lo:.4f},{auc_hi:.4f}]")
    logger.info(f"EER={eer:.4f}")
    logger.info(f"ACC={thresh_met['ACC']:.4f} F1={thresh_met['F1']:.4f}")

    # ----------------------------------------------------------------
    # UCF-only baseline
    # ----------------------------------------------------------------
    ucf_path = require_file(ROOT / cfg["paths"]["ucf_scores"], "UCF scores")
    ucf = pd.read_csv(ucf_path)

    # Align on video_id
    merged = oof[["video_id", "label", "prediction"]].merge(
        ucf[["video_id", "detector_score"]].rename(columns={"detector_score": "ucf_score"}),
        on="video_id", how="inner"
    )
    if len(merged) < len(oof):
        logger.warning(f"UCF merge lost {len(oof)-len(merged)} samples; proceeding with {len(merged)}")

    y_true_aligned = merged["label"].values
    y_mgf = merged["prediction"].values
    y_ucf = merged["ucf_score"].values

    auc_mgf_aligned = compute_auc(y_true_aligned, y_mgf)
    auc_ucf = compute_auc(y_true_aligned, y_ucf)
    delta_ucf = auc_mgf_aligned - auc_ucf

    z_ucf, p_ucf = delong_test(y_true_aligned, y_mgf, y_ucf)
    perm_ucf = permutation_test_auc(y_true_aligned, y_mgf, y_ucf, n_iter=10000, seed=42)

    delong_ucf_result = {
        "model": "ModalityGated",
        "baseline": "UCF_only",
        "auc_model": auc_mgf_aligned,
        "auc_baseline": auc_ucf,
        "delta_auc": delta_ucf,
        "delong_z": z_ucf,
        "delong_p": p_ucf,
        "permutation_p": perm_ucf["p_value"],
        "n": len(merged),
    }
    with open(stats_dir / "final_exp15_delong_vs_ucf_only.json", "w") as f:
        json.dump(delong_ucf_result, f, indent=2)
    logger.info(f"DeLong vs UCF only: ΔAUC={delta_ucf:+.4f} p={p_ucf:.4e}")

    # Permutation test
    perm_results = {"vs_ucf_only": perm_ucf}

    # ----------------------------------------------------------------
    # UCF+quality baseline (Exp.12) if available
    # ----------------------------------------------------------------
    exp12_candidates = [
        ROOT / "scripts/exp12_ucf_quality_fusion/outputs/predictions/final_exp12_oof_predictions.csv",
        ROOT / "outputs/results/exp12_oof_predictions.csv",
    ]
    ucf_quality_path = next((p for p in exp12_candidates if p.exists()), None)

    delong_quality_result = None
    if ucf_quality_path:
        logger.info(f"Found UCF+quality predictions at {ucf_quality_path}")
        exp12 = pd.read_csv(ucf_quality_path)
        score_col = next((c for c in ["prediction", "y_score", "pred_proba"] if c in exp12.columns), None)
        if score_col is None:
            logger.warning("Cannot find score column in Exp.12 predictions. Skipping comparison.")
        else:
            merged2 = oof[["video_id", "label", "prediction"]].merge(
                exp12[["video_id", score_col]].rename(columns={score_col: "ucf_qual_score"}),
                on="video_id", how="inner"
            )
            if len(merged2) > 10:
                y_mgf2 = merged2["prediction"].values
                y_ucfq = merged2["ucf_qual_score"].values
                y_true2 = merged2["label"].values
                auc_ucfq = compute_auc(y_true2, y_ucfq)
                auc_mgf2 = compute_auc(y_true2, y_mgf2)
                z_q, p_q = delong_test(y_true2, y_mgf2, y_ucfq)
                perm_q = permutation_test_auc(y_true2, y_mgf2, y_ucfq, n_iter=10000, seed=42)
                delong_quality_result = {
                    "model": "ModalityGated",
                    "baseline": "UCF+quality_Exp12",
                    "auc_model": float(auc_mgf2),
                    "auc_baseline": float(auc_ucfq),
                    "delta_auc": float(auc_mgf2 - auc_ucfq),
                    "delong_z": float(z_q),
                    "delong_p": float(p_q),
                    "permutation_p": float(perm_q["p_value"]),
                    "n": len(merged2),
                }
                with open(stats_dir / "final_exp15_delong_vs_ucf_quality.json", "w") as f:
                    json.dump(delong_quality_result, f, indent=2)
                logger.info(f"DeLong vs UCF+quality: ΔAUC={auc_mgf2-auc_ucfq:+.4f} p={p_q:.4e}")
                perm_results["vs_ucf_quality"] = perm_q
    else:
        logger.info("Exp.12 predictions not found. Skipping UCF+quality comparison.")
        # Write empty placeholder so the file is always present
        with open(stats_dir / "final_exp15_delong_vs_ucf_quality.json", "w") as f:
            json.dump({"note": "Exp.12 predictions not available"}, f, indent=2)

    with open(stats_dir / "final_exp15_permutation_tests.json", "w") as f:
        json.dump(perm_results, f, indent=2, default=float)

    # ----------------------------------------------------------------
    # Results table
    # ----------------------------------------------------------------
    rows = [
        {
            "model": "UCF_only",
            "AUC": round(auc_ucf, 4),
            "AUC_ci_low": None,
            "AUC_ci_high": None,
            "ACC": round(threshold_metrics(y_true_aligned, y_ucf)["ACC"], 4),
            "F1": round(threshold_metrics(y_true_aligned, y_ucf)["F1"], 4),
            "Precision": round(threshold_metrics(y_true_aligned, y_ucf)["Precision"], 4),
            "Recall": round(threshold_metrics(y_true_aligned, y_ucf)["Recall"], 4),
            "EER": round(compute_eer(y_true_aligned, y_ucf), 4),
            "n": len(merged),
        },
        {
            "model": "ModalityGated_Exp15",
            "AUC": round(auc_mean, 4),
            "AUC_ci_low": round(auc_lo, 4),
            "AUC_ci_high": round(auc_hi, 4),
            "ACC": round(thresh_met["ACC"], 4),
            "F1": round(thresh_met["F1"], 4),
            "Precision": round(thresh_met["Precision"], 4),
            "Recall": round(thresh_met["Recall"], 4),
            "EER": round(eer, 4),
            "n": n,
        },
    ]
    if delong_quality_result:
        rows.insert(1, {
            "model": "UCF+quality_Exp12",
            "AUC": round(delong_quality_result["auc_baseline"], 4),
            "AUC_ci_low": None,
            "AUC_ci_high": None,
            "ACC": None, "F1": None, "Precision": None, "Recall": None, "EER": None,
            "n": delong_quality_result["n"],
        })

    results_df = pd.DataFrame(rows)
    csv_out = table_dir / "final_exp15_results.csv"
    tex_out = table_dir / "final_exp15_results.tex"
    results_df.to_csv(csv_out, index=False)
    results_df.to_latex(tex_out, index=False, na_rep="—", float_format="%.4f")
    logger.info(f"Results table saved: {csv_out}")

    # ----------------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 64)
    print("Exp.15 — Evaluation Results")
    print("=" * 64)
    print(results_df.to_string(index=False))
    print("=" * 64)
    delta_str = f"{delta_ucf:+.4f}"
    print(f"\nvs UCF only:    ΔAUC = {delta_str}  (DeLong p = {p_ucf:.3e})")
    if delong_quality_result:
        d = delong_quality_result
        print(f"vs UCF+quality: ΔAUC = {d['delta_auc']:+.4f}  (DeLong p = {d['delong_p']:.3e})")
    print("=" * 64)


if __name__ == "__main__":
    main()
