"""
Stage 06 — Evaluate ModalityGatedFusion ensemble on the pilot holdout set.

Loads best.pt from all 5 folds, ensembles predictions (mean of probabilities),
and evaluates against pilot ground truth. No retraining.

Reads:
  outputs/predictions/pilot_feature_matrix.parquet
  outputs/checkpoints/fold_{k}/best.pt  (for k in 0..4)
  datasets/detector_processed/pilot_ucf_scores.csv  (UCF baseline)

Writes:
  outputs/predictions/pilot_exp15_predictions.csv
  outputs/tables/pilot_exp15_results.csv  (+.tex)

Run from project root:
  python scripts/exp15_modality_gated/06_pilot_holdout.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import ModalityDataset
from model import ModalityGatedFusion
from utils import (
    bootstrap_auc_ci,
    compute_auc,
    compute_eer,
    get_project_root,
    load_config,
    require_file,
    set_seeds,
    setup_logger,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()


def run_inference(model, df, det_col, emo_cols, qual_cols, cfg, device) -> tuple:
    """Return (probs, gate_weights_array, branch_logits_array)."""
    ds = ModalityDataset(df, det_col, emo_cols, qual_cols)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    model.eval()

    all_probs, all_gates, all_branches = [], [], []
    with torch.no_grad():
        for det, emo, qual, _ in loader:
            det, emo, qual = det.to(device), emo.to(device), qual.to(device)
            out = model(det, emo, qual)
            probs = torch.sigmoid(out["logit"]).cpu().numpy()
            all_probs.append(probs)
            all_gates.append(out["gate_weights"].cpu().numpy())
            all_branches.append(out["branch_logits"].cpu().numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_gates, axis=0),
        np.concatenate(all_branches, axis=0),
    )


def main():
    set_seeds(42)
    cfg = load_config(str(CONFIG_PATH))
    out_dir = ROOT / cfg["paths"]["output_root"]
    pred_dir = out_dir / "predictions"
    table_dir = out_dir / "tables"
    log_dir = out_dir / "logs"
    ckpt_dir = out_dir / "checkpoints"
    for d in [pred_dir, table_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15.pilot", str(log_dir / "run.log"))
    logger.info("=== Stage 06: Pilot Holdout Evaluation ===")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    emo_cols = cfg["emotion_feature_cols"]
    qual_cols = cfg["quality_feature_cols"]
    det_col = "detector_score"
    all_feat_cols = [det_col] + emo_cols + qual_cols

    # ----------------------------------------------------------------
    # Load pilot feature matrix
    # ----------------------------------------------------------------
    pilot_path = require_file(pred_dir / "pilot_feature_matrix.parquet",
                              "Run 01_prepare_features.py first")
    pilot = pd.read_parquet(pilot_path)
    logger.info(f"Pilot dataset: {len(pilot)} videos")

    # ----------------------------------------------------------------
    # Load all fold checkpoints and run ensemble
    # ----------------------------------------------------------------
    fold_probs = []
    best_fold_auc = -1.0
    best_fold_idx = 0

    for k in range(cfg["n_folds"]):
        ckpt_path = ckpt_dir / f"fold_{k}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Run 02_train_modality_gated.py before this script."
            )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        scaler = ckpt["scaler"]

        # Scale pilot features using this fold's scaler
        pilot_scaled = pilot.copy()
        pilot_scaled[all_feat_cols] = scaler.transform(pilot[all_feat_cols])

        emotion_dim = len(emo_cols)
        quality_dim = len(qual_cols)
        model = ModalityGatedFusion(
            emotion_dim=emotion_dim,
            quality_dim=quality_dim,
            embed_dim=cfg["embed_dim"],
            dropout=cfg["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        probs, gates, branches = run_inference(
            model, pilot_scaled, det_col, emo_cols, qual_cols, cfg, device
        )
        fold_val_auc = float(ckpt.get("val_auc", 0.0))
        fold_probs.append(probs)

        if fold_val_auc > best_fold_auc:
            best_fold_auc = fold_val_auc
            best_fold_idx = k
            best_gates = gates
            best_branches = branches

        logger.info(f"  Fold {k}: val_AUC={fold_val_auc:.4f} pilot_AUC={compute_auc(pilot['label_int'].values, probs):.4f}")

    # Ensemble: mean of probabilities across all folds
    ensemble_probs = np.stack(fold_probs, axis=0).mean(axis=0)
    y_true = pilot["label_int"].values

    # Use gate weights from the best single fold for interpretability columns
    logger.info(f"Ensemble from {cfg['n_folds']} folds. Best fold: {best_fold_idx} (val AUC={best_fold_auc:.4f})")

    # ----------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------
    auc_ens, auc_lo, auc_hi = bootstrap_auc_ci(y_true, ensemble_probs, n_iter=2000, seed=42)
    eer = compute_eer(y_true, ensemble_probs)
    y_pred = (ensemble_probs >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    # UCF-only baseline on pilot
    ucf_pilot = pd.read_csv(require_file(ROOT / cfg["paths"]["ucf_scores_pilot"], "pilot UCF scores"))
    merged = pilot[["video_id", "label_int"]].merge(
        ucf_pilot[["video_id", "detector_score"]].rename(columns={"detector_score": "ucf_score"}),
        on="video_id", how="inner"
    )
    auc_ucf_pilot = compute_auc(merged["label_int"].values, merged["ucf_score"].values)

    # ----------------------------------------------------------------
    # Save predictions
    # ----------------------------------------------------------------
    out_df = pilot[["video_id", "label_int", "forgery_family", "dominant_emotion"]].copy()
    out_df = out_df.rename(columns={"label_int": "label"})
    out_df["prediction"] = ensemble_probs
    out_df["gate_det"] = best_gates[:, 0]
    out_df["gate_emo"] = best_gates[:, 1]
    out_df["gate_qual"] = best_gates[:, 2]
    out_df["branch_det_logit"] = best_branches[:, 0]
    out_df["branch_emo_logit"] = best_branches[:, 1]
    out_df["branch_qual_logit"] = best_branches[:, 2]

    out_df.to_csv(pred_dir / "pilot_exp15_predictions.csv", index=False)
    logger.info(f"Pilot predictions saved: {pred_dir / 'pilot_exp15_predictions.csv'}")

    # ----------------------------------------------------------------
    # Results table
    # ----------------------------------------------------------------
    results = pd.DataFrame([
        {
            "model": "UCF_only_pilot",
            "AUC": round(auc_ucf_pilot, 4),
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
            "n": len(pilot),
        },
    ])

    csv_out = table_dir / "pilot_exp15_results.csv"
    tex_out = table_dir / "pilot_exp15_results.tex"
    results.to_csv(csv_out, index=False)
    results.to_latex(tex_out, index=False, na_rep="—", float_format="%.4f")
    logger.info(f"Pilot results table: {csv_out}")

    # ----------------------------------------------------------------
    # Console output
    # ----------------------------------------------------------------
    print("\n" + "=" * 64)
    print("Exp.15 — Pilot Holdout Results")
    print("=" * 64)
    print(results.to_string(index=False))
    print("=" * 64)
    print(f"\nUCF-only pilot AUC:      {auc_ucf_pilot:.4f}")
    print(f"ModalityGated pilot AUC: {auc_ens:.4f} (95% CI: {auc_lo:.4f}–{auc_hi:.4f})")
    print(f"ΔAUC vs UCF only:        {auc_ens - auc_ucf_pilot:+.4f}")
    print("=" * 64)

    mean_gates = out_df[["gate_det", "gate_emo", "gate_qual"]].mean()
    print(f"\nMean gate weights (pilot): "
          f"det={mean_gates['gate_det']:.3f}  "
          f"emo={mean_gates['gate_emo']:.3f}  "
          f"qual={mean_gates['gate_qual']:.3f}")


if __name__ == "__main__":
    main()
