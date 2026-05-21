"""
Stage 06 — Evaluate ThreeModalityGated ensemble on pilot holdout.

Applies the 5-fold ensemble (trained on final trainval) to pilot videos
without any retraining.

Reads:
  outputs/predictions/pilot_feature_matrix.parquet
  outputs/checkpoints/fold_{k}/best.pt  (k = 0..4)
  datasets/detector_processed/pilot_ucf_scores.csv

Writes:
  outputs/predictions/pilot_exp15_predictions.csv
  outputs/tables/pilot_exp15_results.csv  (+.tex)
  outputs/stats/pilot_exp15_delong_vs_ucf.json

Run from project root:
  python scripts/exp15_three_modality/06_pilot_holdout.py
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

    logger = setup_logger("exp15_tm.pilot", str(log_dir / "run.log"))
    logger.info("=== Stage 06: Pilot Holdout Evaluation ===")

    pilot_path = pred_dir / "pilot_feature_matrix.parquet"
    if not pilot_path.exists():
        logger.error("pilot_feature_matrix.parquet not found — run 01_prepare_features.py first")
        print("pilot_feature_matrix.parquet not found. Run 01_prepare_features.py first.")
        return

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    qual_cols = cfg["quality_features"]
    emo_static_cols = cfg["emotion_static_features"]
    emo_temporal_base = [c for c in cfg["emotion_temporal_features"] if not c.startswith("std_score_")]
    emo_temporal_std = [c for c in cfg["emotion_temporal_features"] if c.startswith("std_score_")]
    emo_temporal_cols = emo_temporal_base + emo_temporal_std
    all_feat_cols = qual_cols + emo_static_cols + emo_temporal_cols

    pilot = pd.read_parquet(pilot_path)
    logger.info(f"Pilot: {len(pilot)} videos  fake={int(pilot['label_int'].sum())}  "
                f"real={int((pilot['label_int']==0).sum())}")

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

        pilot_scaled = pilot.copy()
        pilot_scaled[all_feat_cols] = scaler.transform(pilot[all_feat_cols])

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
            model, pilot_scaled, qual_cols, emo_static_cols, emo_temporal_cols,
            cfg["batch_size"], device,
        )
        fold_auc = compute_auc(pilot["label_int"].values, probs)
        fold_probs.append(probs)

        if fold_val_auc > best_fold_val_auc:
            best_fold_val_auc = fold_val_auc
            best_fold_idx = k
            best_gates = gates
            best_branches = branches

        logger.info(f"  Fold {k}: val_AUC={fold_val_auc:.4f}  pilot_AUC={fold_auc:.4f}")

    ensemble_probs = np.stack(fold_probs, axis=0).mean(axis=0)
    y_true = pilot["label_int"].values

    # ── Metrics ────────────────────────────────────────────────────────────────
    auc_ens, auc_lo, auc_hi = bootstrap_auc_ci(y_true, ensemble_probs, n_iter=2000, seed=42)
    eer = compute_eer(y_true, ensemble_probs)
    y_pred = (ensemble_probs >= 0.5).astype(int)

    # ── UCF baseline on pilot ─────────────────────────────────────────────────
    ucf_pilot = pd.read_csv(
        require_file(ROOT / cfg["paths"]["ucf_scores_pilot"], "pilot UCF scores")
    )
    merged = pilot[["video_id", "label_int"]].merge(
        ucf_pilot[["video_id", "detector_score"]].rename(columns={"detector_score": "ucf_score"}),
        on="video_id", how="inner",
    )
    if len(merged) < len(pilot):
        logger.warning(f"UCF pilot merge: {len(pilot)-len(merged)} videos missing UCF score")

    y_true_ucf = merged["label_int"].values
    y_ucf = merged["ucf_score"].values
    y_mgf_ucf = ensemble_probs[:len(merged)]

    auc_ucf = compute_auc(y_true_ucf, y_ucf)
    delta = compute_auc(y_true_ucf, y_mgf_ucf) - auc_ucf
    z, p = delong_test(y_true_ucf, y_mgf_ucf, y_ucf)

    delong_result = {
        "model": "ThreeModalityGated_ensemble",
        "baseline": "UCF_only",
        "split": "pilot",
        "auc_model": float(compute_auc(y_true_ucf, y_mgf_ucf)),
        "auc_baseline": float(auc_ucf),
        "delta_auc": float(delta),
        "delong_z": float(z),
        "delong_p": float(p),
        "n": len(merged),
    }
    with open(stats_dir / "pilot_exp15_delong_vs_ucf.json", "w") as f:
        json.dump(delong_result, f, indent=2)

    # ── Save predictions ───────────────────────────────────────────────────────
    out_df = pilot[["video_id", "label_int", "forgery_family", "dominant_emotion"]].copy()
    out_df["prediction"] = ensemble_probs
    out_df["gate_q"] = best_gates[:, 0]
    out_df["gate_s"] = best_gates[:, 1]
    out_df["gate_t"] = best_gates[:, 2]
    out_df.to_csv(pred_dir / "pilot_exp15_predictions.csv", index=False)

    results = pd.DataFrame([
        {"model": "UCF_only_pilot", "AUC": round(auc_ucf, 4),
         "AUC_ci_low": None, "AUC_ci_high": None,
         "ACC": None, "F1": None, "EER": None, "n": len(merged)},
        {"model": "ThreeModality_ensemble_pilot",
         "AUC": round(auc_ens, 4), "AUC_ci_low": round(auc_lo, 4),
         "AUC_ci_high": round(auc_hi, 4),
         "ACC": round(float(accuracy_score(y_true, y_pred)), 4),
         "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
         "EER": round(eer, 4), "n": len(pilot)},
    ])
    results.to_csv(table_dir / "pilot_exp15_results.csv", index=False)
    results.to_latex(table_dir / "pilot_exp15_results.tex", index=False,
                     na_rep="—", float_format="%.4f")

    print(f"\n{'='*60}")
    print("Exp.15 v2 — Pilot Holdout Results")
    print(f"{'='*60}")
    print(results.to_string(index=False))
    print(f"\nUCF-only pilot AUC:      {auc_ucf:.4f}")
    print(f"ThreeModality pilot AUC: {auc_ens:.4f}  (95% CI: {auc_lo:.4f}–{auc_hi:.4f})")
    print(f"ΔAUC vs UCF:             {delta:+.4f}  (DeLong p={p:.3e})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
