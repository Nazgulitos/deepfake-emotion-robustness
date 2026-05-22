"""
Stage 03 — Evaluate ThreeModalityGated ensemble on test holdout.

Ensemble of 5 fold best.pt checkpoints on test_feature_matrix.parquet (155 videos).
Also computes OOF metrics on trainval_oof_predictions.csv.
Compares against UCF-only baseline using DeLong's test.

Reads:
  outputs/predictions/trainval_oof_predictions.csv
  outputs/predictions/test_feature_matrix.parquet
  outputs/checkpoints/fold_{k}/best.pt  (k = 0..4)
  datasets/detector_processed/final_ucf_scores.csv

Writes:
  outputs/predictions/test_exp15_predictions.csv
  outputs/tables/final_exp15_results.csv  (+.tex)
  outputs/stats/final_exp15_delong_vs_ucf.json

Run from project root:
  python scripts/exp15_three_modality/03_evaluate_test.py
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

from dataset import ThreeModalityDataset
from model import ThreeModalityGated
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


@torch.no_grad()
def run_inference(model, df, qual_cols, emo_static_cols, emo_temporal_cols, batch_size, device):
    ds = ThreeModalityDataset(df, qual_cols, emo_static_cols, emo_temporal_cols)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_probs, all_gates, all_branches = [], [], []
    for x_q, x_s, x_t, _ in loader:
        x_q, x_s, x_t = x_q.to(device), x_s.to(device), x_t.to(device)
        out = model(x_q, x_s, x_t)
        all_probs.append(torch.sigmoid(out["logit"]).cpu().numpy())
        all_gates.append(out["gate_weights"].cpu().numpy())
        all_branches.append(out["branch_logits"].cpu().numpy())
    return (
        np.concatenate(all_probs),
        np.concatenate(all_gates),
        np.concatenate(all_branches),
    )


def compute_metrics(y_true, y_score, n_bootstrap=2000):
    auc, auc_lo, auc_hi = bootstrap_auc_ci(y_true, y_score, n_iter=n_bootstrap, seed=42)
    eer = compute_eer(y_true, y_score)
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "AUC": round(auc, 4),
        "AUC_ci_low": round(auc_lo, 4),
        "AUC_ci_high": round(auc_hi, 4),
        "ACC": round(float(accuracy_score(y_true, y_pred)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "EER": round(eer, 4),
    }


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

    logger = setup_logger("exp15_tm.evaluate", str(log_dir / "run.log"))
    logger.info("=== Stage 03: Evaluate Test Holdout ===")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    qual_cols = cfg["quality_features"]
    emo_static_cols = cfg["emotion_static_features"]
    emo_temporal_base = [c for c in cfg["emotion_temporal_features"] if not c.startswith("std_score_")]
    emo_temporal_std = [c for c in cfg["emotion_temporal_features"] if c.startswith("std_score_")]
    emo_temporal_cols = emo_temporal_base + emo_temporal_std
    all_feat_cols = qual_cols + emo_static_cols + emo_temporal_cols

    # ── Load test set ──────────────────────────────────────────────────────────
    test = pd.read_parquet(
        require_file(pred_dir / "test_feature_matrix.parquet", "Run 01_prepare_features.py")
    )
    logger.info(f"Test: {len(test)} videos  fake={int(test['label_int'].sum())}  "
                f"real={int((test['label_int']==0).sum())}")

    # ── Ensemble inference ─────────────────────────────────────────────────────
    fold_probs = []
    best_fold_val_auc = -1.0
    best_fold_idx = 0

    for k in range(cfg["n_folds"]):
        ckpt_path = ckpt_dir / f"fold_{k}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\nRun 02_train_three_modality.py first."
            )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        scaler = ckpt["scaler"]
        fold_val_auc = float(ckpt.get("val_auc", 0.0))

        test_scaled = test.copy()
        test_scaled[all_feat_cols] = scaler.transform(test[all_feat_cols])

        model = ThreeModalityGated(
            quality_dim=len(qual_cols),
            emo_static_dim=len(emo_static_cols),
            emo_temporal_dim=len(emo_temporal_cols),
            embed_dim=cfg["embed_dim"],
            gate_hidden=cfg["gate_hidden"],
            dropout=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        probs, gates, branches = run_inference(
            model, test_scaled, qual_cols, emo_static_cols, emo_temporal_cols,
            cfg["batch_size"], device,
        )
        fold_auc = compute_auc(test["label_int"].values, probs)
        fold_probs.append(probs)

        if fold_val_auc > best_fold_val_auc:
            best_fold_val_auc = fold_val_auc
            best_fold_idx = k
            best_gates = gates
            best_branches = branches

        logger.info(f"  Fold {k}: val_AUC={fold_val_auc:.4f}  test_AUC={fold_auc:.4f}")

    ensemble_probs = np.stack(fold_probs, axis=0).mean(axis=0)
    y_true = test["label_int"].values
    logger.info(f"Best fold by val AUC: fold_{best_fold_idx}")

    # ── Test metrics ───────────────────────────────────────────────────────────
    test_metrics = compute_metrics(y_true, ensemble_probs)

    # ── OOF metrics ───────────────────────────────────────────────────────────
    oof = pd.read_csv(
        require_file(pred_dir / "trainval_oof_predictions.csv", "Run 02_train_three_modality.py")
    )
    oof_metrics = compute_metrics(oof["label_int"].values, oof["prediction"].values)

    # ── UCF baseline ───────────────────────────────────────────────────────────
    ucf_all = pd.read_csv(require_file(ROOT / cfg["paths"]["ucf_scores"], "UCF scores"))
    ucf_sub = ucf_all[["video_id", "detector_score"]].rename(
        columns={"detector_score": "ucf_score"}
    )

    # Test UCF
    test_merged = test[["video_id", "label_int"]].merge(ucf_sub, on="video_id", how="inner")
    if len(test_merged) < len(test):
        logger.warning(f"UCF merge: {len(test)-len(test_merged)} test videos missing UCF score")
    y_true_ucf = test_merged["label_int"].values
    y_ucf_test = test_merged["ucf_score"].values
    y_mgf_test = ensemble_probs[:len(test_merged)]
    auc_ucf_test = compute_auc(y_true_ucf, y_ucf_test)

    z_test, p_test = delong_test(y_true_ucf, y_mgf_test, y_ucf_test)
    perm_test = permutation_test_auc(y_true_ucf, y_mgf_test, y_ucf_test, n_iter=10000, seed=42)

    delong_result = {
        "model": "ThreeModalityGated_ensemble",
        "baseline": "UCF_only",
        "split": "test",
        "auc_model": float(compute_auc(y_true_ucf, y_mgf_test)),
        "auc_baseline": float(auc_ucf_test),
        "delta_auc": float(compute_auc(y_true_ucf, y_mgf_test) - auc_ucf_test),
        "delong_z": float(z_test),
        "delong_p": float(p_test),
        "permutation_p": float(perm_test["p_value"]),
        "n": len(test_merged),
    }
    with open(stats_dir / "final_exp15_delong_vs_ucf.json", "w") as f:
        json.dump(delong_result, f, indent=2)
    logger.info(f"DeLong test vs UCF: ΔAUC={delong_result['delta_auc']:+.4f}  p={p_test:.3e}")

    # OOF UCF
    oof_merged = oof[["video_id", "label_int"]].merge(ucf_sub, on="video_id", how="inner")
    auc_ucf_oof = compute_auc(oof_merged["label_int"].values, oof_merged["ucf_score"].values)

    # ── Per-forgery AUC ────────────────────────────────────────────────────────
    test_out = test[["video_id", "label_int", "forgery_family", "dominant_emotion"]].copy()
    test_out = test_out.rename(columns={"label_int": "label"})
    test_out["fold"] = "ensemble"
    test_out["split_type"] = "test_holdout"
    test_out["prediction"] = ensemble_probs
    test_out["gate_q"] = best_gates[:, 0]
    test_out["gate_s"] = best_gates[:, 1]
    test_out["gate_t"] = best_gates[:, 2]
    test_out["branch_q_logit"] = best_branches[:, 0]
    test_out["branch_s_logit"] = best_branches[:, 1]
    test_out["branch_t_logit"] = best_branches[:, 2]

    reals = test_out[test_out["forgery_family"].isna()].copy()
    ucf_reals = test_merged[test_merged["label_int"] == 0].copy()
    per_forgery_rows = []
    for fam, fakes in test_out.groupby("forgery_family"):
        if len(fakes) < 5:
            continue
        sub = pd.concat([fakes, reals], ignore_index=True)
        fam_auc = compute_auc(sub["label"].values, sub["prediction"].values)
        ucf_fakes = test_merged[test_merged["video_id"].isin(fakes["video_id"])]
        ucf_sub2 = pd.concat([ucf_fakes, ucf_reals], ignore_index=True)
        ucf_fam_auc = (
            compute_auc(ucf_sub2["label_int"].values, ucf_sub2["ucf_score"].values)
            if len(ucf_sub2) > 1 else None
        )
        per_forgery_rows.append({
            "forgery_family": fam,
            "n_fake": len(fakes),
            "n_real": len(reals),
            "AUC_ThreeModality": round(fam_auc, 4),
            "AUC_UCF_only": round(ucf_fam_auc, 4) if ucf_fam_auc is not None else None,
            "delta_AUC": round(fam_auc - ucf_fam_auc, 4) if ucf_fam_auc is not None else None,
            "mean_gate_q": round(fakes["gate_q"].mean(), 4),
            "mean_gate_s": round(fakes["gate_s"].mean(), 4),
            "mean_gate_t": round(fakes["gate_t"].mean(), 4),
        })
    per_forgery_df = pd.DataFrame(per_forgery_rows)
    per_forgery_df.to_csv(table_dir / "test_exp15_per_forgery_auc.csv", index=False)

    # ── Save outputs ───────────────────────────────────────────────────────────
    test_out.to_csv(pred_dir / "test_exp15_predictions.csv", index=False)

    results = pd.DataFrame([
        {"split": "trainval_oof", "model": "UCF_only",
         "AUC": round(auc_ucf_oof, 4), "AUC_ci_low": None, "AUC_ci_high": None,
         "ACC": None, "F1": None, "Precision": None, "Recall": None, "EER": None,
         "n": len(oof_merged)},
        {"split": "trainval_oof", "model": "ThreeModality_full",
         **oof_metrics, "n": len(oof)},
        {"split": "test_holdout", "model": "UCF_only",
         "AUC": round(auc_ucf_test, 4), "AUC_ci_low": None, "AUC_ci_high": None,
         "ACC": None, "F1": None, "Precision": None, "Recall": None, "EER": None,
         "n": len(test_merged)},
        {"split": "test_holdout", "model": "ThreeModality_full",
         **test_metrics, "n": len(test)},
    ])
    results.to_csv(table_dir / "final_exp15_results.csv", index=False)
    results.to_latex(table_dir / "final_exp15_results.tex", index=False,
                     na_rep="—", float_format="%.4f")

    logger.info(f"Results saved: {table_dir / 'final_exp15_results.csv'}")

    # ── Console summary ────────────────────────────────────────────────────────
    mean_g = test_out[["gate_q", "gate_s", "gate_t"]].mean()
    print(f"\n{'='*68}")
    print("Exp.15 v2 — Three-Modality Gated Fusion")
    print(f"{'='*68}")
    print(f"OOF  AUC: {oof_metrics['AUC']:.4f}  "
          f"(95%CI: {oof_metrics['AUC_ci_low']:.4f}–{oof_metrics['AUC_ci_high']:.4f})")
    print(f"Test AUC: {test_metrics['AUC']:.4f}  "
          f"(95%CI: {test_metrics['AUC_ci_low']:.4f}–{test_metrics['AUC_ci_high']:.4f})")
    print(f"UCF-only test AUC: {auc_ucf_test:.4f}")
    print(f"ΔAUC vs UCF:       {delong_result['delta_auc']:+.4f}  (DeLong p={p_test:.3e})")
    print(f"\nMean gate weights (test):")
    print(f"  quality={mean_g['gate_q']:.3f}  static={mean_g['gate_s']:.3f}  temporal={mean_g['gate_t']:.3f}")
    print(f"\nPer-forgery:")
    print(per_forgery_df.to_string(index=False))
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
