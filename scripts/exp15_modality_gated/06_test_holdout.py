"""
Stage 06 — Evaluate ModalityGatedFusion ensemble on the predefined test holdout.

The test split (split=='test', 155 videos) is identity-disjoint from trainval.
All 5 fold checkpoints are used for ensemble averaging.

Reads:
  outputs/predictions/test_feature_matrix.parquet
  outputs/checkpoints/fold_{k}/best.pt  (k = 0..4)
  datasets/detector_processed/final_ucf_scores.csv  (UCF baseline on test)

Writes:
  outputs/predictions/test_exp15_predictions.csv
  outputs/tables/test_exp15_results.csv  (+.tex)
  outputs/stats/test_exp15_delong_vs_ucf_only.json

Run from project root:
  python scripts/exp15_modality_gated/06_test_holdout.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from dataset import ModalityDataset
from model import ModalityGatedFusion
from utils import (
    bootstrap_auc_ci,
    compute_auc,
    compute_eer,
    delong_test,
    get_project_root,
    load_config,
    permutation_test_auc,
    require_file,
    set_seeds,
    setup_logger,
)

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()


def run_inference(model, df, det_col, emo_cols, qual_cols, cfg, device):
    ds = ModalityDataset(df, det_col, emo_cols, qual_cols)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    model.eval()
    all_probs, all_gates, all_branches = [], [], []
    with torch.no_grad():
        for det, emo, qual, _ in loader:
            det, emo, qual = det.to(device), emo.to(device), qual.to(device)
            out = model(det, emo, qual)
            all_probs.append(torch.sigmoid(out["logit"]).cpu().numpy())
            all_gates.append(out["gate_weights"].cpu().numpy())
            all_branches.append(out["branch_logits"].cpu().numpy())
    return (
        np.concatenate(all_probs),
        np.concatenate(all_gates),
        np.concatenate(all_branches),
    )


def main():
    set_seeds(42)
    cfg = load_config(str(CONFIG_PATH))
    out_dir = ROOT / cfg["paths"]["output_root"]
    pred_dir = out_dir / "predictions"
    table_dir = out_dir / "tables"
    stats_dir = out_dir / "stats"
    log_dir = out_dir / "logs"
    ckpt_dir = out_dir / "checkpoints"
    for d in [pred_dir, table_dir, stats_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15.test_holdout", str(log_dir / "run.log"))
    logger.info("=== Stage 06: Test Holdout Evaluation ===")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    emo_cols = cfg["emotion_feature_cols"]
    qual_cols = cfg["quality_feature_cols"]
    det_col = "detector_score"
    all_feat_cols = [det_col] + emo_cols + qual_cols

    # ----------------------------------------------------------------
    # Load test feature matrix
    # ----------------------------------------------------------------
    test_path = require_file(
        pred_dir / "test_feature_matrix.parquet",
        "Run 01_prepare_features.py first"
    )
    test = pd.read_parquet(test_path)
    logger.info(f"Test holdout: {len(test)} videos  "
                f"(fake={int(test['label_int'].sum())}  real={int((test['label_int']==0).sum())})")
    logger.info(f"Forgery families: {test['forgery_family'].value_counts().to_dict()}")

    # ----------------------------------------------------------------
    # Ensemble inference over all 5 folds
    # ----------------------------------------------------------------
    fold_probs = []
    best_fold_val_auc = -1.0
    best_fold_idx = 0

    for k in range(cfg["n_folds"]):
        ckpt_path = ckpt_dir / f"fold_{k}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Run 02_train_modality_gated.py first."
            )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        scaler = ckpt["scaler"]

        test_scaled = test.copy()
        test_scaled[all_feat_cols] = scaler.transform(test[all_feat_cols])

        model = ModalityGatedFusion(
            emotion_dim=len(emo_cols),
            quality_dim=len(qual_cols),
            embed_dim=cfg["embed_dim"],
            dropout=cfg["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        probs, gates, branches = run_inference(
            model, test_scaled, det_col, emo_cols, qual_cols, cfg, device
        )
        fold_auc = compute_auc(test["label_int"].values, probs)
        fold_val_auc = float(ckpt.get("val_auc", 0.0))
        fold_probs.append(probs)

        if fold_val_auc > best_fold_val_auc:
            best_fold_val_auc = fold_val_auc
            best_fold_idx = k
            best_gates = gates
            best_branches = branches

        logger.info(f"  Fold {k}: val_AUC={fold_val_auc:.4f}  test_AUC={fold_auc:.4f}")

    ensemble_probs = np.stack(fold_probs, axis=0).mean(axis=0)
    y_true = test["label_int"].values
    logger.info(f"Ensemble (5 folds). Best single fold by val AUC: fold_{best_fold_idx}")

    # ----------------------------------------------------------------
    # Metrics — ensemble
    # ----------------------------------------------------------------
    auc_ens, auc_lo, auc_hi = bootstrap_auc_ci(y_true, ensemble_probs, n_iter=2000, seed=42)
    eer = compute_eer(y_true, ensemble_probs)
    y_pred = (ensemble_probs >= 0.5).astype(int)
    acc  = float(accuracy_score(y_true, y_pred))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))

    # ----------------------------------------------------------------
    # UCF-only baseline on the same test videos
    # ----------------------------------------------------------------
    ucf_all = pd.read_csv(require_file(ROOT / cfg["paths"]["ucf_scores"], "UCF scores"))
    merged = test[["video_id", "label_int"]].merge(
        ucf_all[["video_id", "detector_score"]].rename(columns={"detector_score": "ucf_score"}),
        on="video_id", how="inner"
    )
    if len(merged) < len(test):
        logger.warning(f"UCF merge: {len(test)-len(merged)} test videos missing UCF score")

    y_true_ucf = merged["label_int"].values
    y_ucf      = merged["ucf_score"].values
    y_mgf_ucf  = ensemble_probs[:len(merged)]  # same order since test is not shuffled

    auc_ucf = compute_auc(y_true_ucf, y_ucf)
    delta   = compute_auc(y_true_ucf, y_mgf_ucf) - auc_ucf
    z, p    = delong_test(y_true_ucf, y_mgf_ucf, y_ucf)
    perm    = permutation_test_auc(y_true_ucf, y_mgf_ucf, y_ucf, n_iter=10000, seed=42)

    delong_result = {
        "model": "ModalityGated_Exp15_ensemble",
        "baseline": "UCF_only",
        "auc_model": float(compute_auc(y_true_ucf, y_mgf_ucf)),
        "auc_baseline": float(auc_ucf),
        "delta_auc": float(delta),
        "delong_z": float(z),
        "delong_p": float(p),
        "permutation_p": float(perm["p_value"]),
        "n": len(merged),
    }
    with open(stats_dir / "test_exp15_delong_vs_ucf_only.json", "w") as f:
        json.dump(delong_result, f, indent=2)
    logger.info(f"DeLong vs UCF only: ΔAUC={delta:+.4f}  p={p:.3e}")

    # ----------------------------------------------------------------
    # Per-forgery AUC breakdown
    # ----------------------------------------------------------------
    test_out = test[["video_id", "label_int", "forgery_family", "dominant_emotion"]].copy()
    test_out = test_out.rename(columns={"label_int": "label"})
    test_out["prediction"]       = ensemble_probs
    test_out["gate_det"]         = best_gates[:, 0]
    test_out["gate_emo"]         = best_gates[:, 1]
    test_out["gate_qual"]        = best_gates[:, 2]
    test_out["branch_det_logit"] = best_branches[:, 0]
    test_out["branch_emo_logit"] = best_branches[:, 1]
    test_out["branch_qual_logit"]= best_branches[:, 2]

    # Per-forgery AUC: each family's fakes vs all real videos (standard per-forgery eval)
    # Real videos have forgery_family=NaN — they serve as negatives for every family.
    reals = test_out[test_out["forgery_family"].isna()].copy()
    merged_reals = merged[merged["label_int"] == 0].copy()  # real rows from UCF-merged frame

    per_forgery_rows = []
    for fam, fakes in test_out.groupby("forgery_family"):
        if len(fakes) < 5:
            continue

        # MGF AUC: this family's fakes + all reals
        sub = pd.concat([fakes, reals], ignore_index=True)
        fam_auc = compute_auc(sub["label"].values, sub["prediction"].values)

        # UCF AUC: same subset from the merged UCF frame
        ucf_fakes = merged[merged["video_id"].isin(fakes["video_id"])]
        ucf_sub = pd.concat([ucf_fakes, merged_reals], ignore_index=True)
        ucf_fam_auc = compute_auc(ucf_sub["label_int"].values, ucf_sub["ucf_score"].values) if len(ucf_sub) > 1 else None

        per_forgery_rows.append({
            "forgery_family": fam,
            "n_fake": len(fakes),
            "n_real": len(reals),
            "AUC_ModalityGated": round(fam_auc, 4),
            "AUC_UCF_only": round(ucf_fam_auc, 4) if ucf_fam_auc is not None else None,
            "delta_AUC": round(fam_auc - ucf_fam_auc, 4) if ucf_fam_auc is not None else None,
            "mean_gate_det": round(fakes["gate_det"].mean(), 4),
            "mean_gate_emo": round(fakes["gate_emo"].mean(), 4),
            "mean_gate_qual": round(fakes["gate_qual"].mean(), 4),
        })
    per_forgery_df = pd.DataFrame(per_forgery_rows)

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    test_out.to_csv(pred_dir / "test_exp15_predictions.csv", index=False)

    results = pd.DataFrame([
        {
            "model": "UCF_only_test",
            "AUC": round(auc_ucf, 4),
            "AUC_ci_low": None, "AUC_ci_high": None,
            "ACC": None, "F1": None, "Precision": None, "Recall": None, "EER": None,
            "n": len(merged),
        },
        {
            "model": "ModalityGated_Exp15_ensemble",
            "AUC": round(auc_ens, 4),
            "AUC_ci_low": round(auc_lo, 4),
            "AUC_ci_high": round(auc_hi, 4),
            "ACC": round(acc, 4),
            "F1": round(f1, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "EER": round(eer, 4),
            "n": len(test),
        },
    ])

    csv_out = table_dir / "test_exp15_results.csv"
    results.to_csv(csv_out, index=False)
    results.to_latex(table_dir / "test_exp15_results.tex", index=False, na_rep="—", float_format="%.4f")

    per_forgery_df.to_csv(table_dir / "test_exp15_per_forgery_auc.csv", index=False)
    per_forgery_df.to_latex(table_dir / "test_exp15_per_forgery_auc.tex", index=False,
                            na_rep="—", float_format="%.4f")

    logger.info(f"Results saved: {csv_out}")

    # ----------------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 68)
    print("Exp.15 — Test Holdout Results  (identity-disjoint, n=155)")
    print("=" * 68)
    print(results.to_string(index=False))
    print()
    print("Per-forgery family breakdown:")
    print(per_forgery_df.to_string(index=False))
    print("=" * 68)
    print(f"\nUCF-only test AUC:       {auc_ucf:.4f}")
    print(f"ModalityGated test AUC:  {auc_ens:.4f}  (95% CI: {auc_lo:.4f}–{auc_hi:.4f})")
    print(f"ΔAUC vs UCF only:        {delta:+.4f}  (DeLong p={p:.3e})")
    mean_g = test_out[["gate_det","gate_emo","gate_qual"]].mean()
    print(f"\nMean gate weights (test): "
          f"det={mean_g['gate_det']:.3f}  "
          f"emo={mean_g['gate_emo']:.3f}  "
          f"qual={mean_g['gate_qual']:.3f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
