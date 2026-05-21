"""
Stage 07 — Ablation study: train four single/reduced-modality variants.

Variants:
  full          det + emo + qual  (already trained — loads existing checkpoints)
  no_detector   emo + qual
  det_only      det
  emo_only      emo
  qual_only     qual

Each variant: 5-fold GroupKFold on trainval, same seeds/splits as stage 02.
OOF AUC reported. Best checkpoint also evaluated on test holdout.

Writes:
  outputs/ablation/<variant>/fold_{k}/best.pt
  outputs/ablation/<variant>/oof_predictions.csv
  outputs/tables/final_exp15_ablation.csv  (+.tex)
  outputs/figures/final_exp15_ablation_roc.png

Run from project root:
  python scripts/exp15_modality_gated/07_ablation.py
"""

import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils import (
    bootstrap_auc_ci,
    compute_auc,
    compute_eer,
    get_project_root,
    hash_config,
    load_config,
    require_file,
    set_seeds,
    setup_logger,
)

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()

# ---------------------------------------------------------------------------
# Ablation model variants
# ---------------------------------------------------------------------------

class DetectorOnly(nn.Module):
    def __init__(self, embed_dim=16, dropout=0.2):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(1, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, det, emo, qual):
        h = self.embed(det)
        logit = self.head(h).squeeze(-1)
        gate = torch.zeros(det.size(0), 3, device=det.device)
        gate[:, 0] = 1.0
        return {"logit": logit, "gate_weights": gate,
                "branch_logits": torch.stack([logit, logit, logit], dim=-1)}


class EmoOnly(nn.Module):
    def __init__(self, emotion_dim=49, embed_dim=16, dropout=0.2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(emotion_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, embed_dim), nn.ReLU(),
        )
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, det, emo, qual):
        h = self.embed(emo)
        logit = self.head(h).squeeze(-1)
        gate = torch.zeros(emo.size(0), 3, device=emo.device)
        gate[:, 1] = 1.0
        return {"logit": logit, "gate_weights": gate,
                "branch_logits": torch.stack([logit, logit, logit], dim=-1)}


class QualOnly(nn.Module):
    def __init__(self, quality_dim=4, embed_dim=16, dropout=0.2):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(quality_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, det, emo, qual):
        h = self.embed(qual)
        logit = self.head(h).squeeze(-1)
        gate = torch.zeros(qual.size(0), 3, device=qual.device)
        gate[:, 2] = 1.0
        return {"logit": logit, "gate_weights": gate,
                "branch_logits": torch.stack([logit, logit, logit], dim=-1)}


class NoDetector(nn.Module):
    def __init__(self, emotion_dim=49, quality_dim=4, embed_dim=16, dropout=0.2):
        super().__init__()
        self.emo_embed = nn.Sequential(
            nn.Linear(emotion_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, embed_dim), nn.ReLU(),
        )
        self.qual_embed = nn.Sequential(
            nn.Linear(quality_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.emo_head  = nn.Linear(embed_dim, 1)
        self.qual_head = nn.Linear(embed_dim, 1)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, det, emo, qual):
        h_e = self.emo_embed(emo)
        h_q = self.qual_embed(qual)
        z_e = self.emo_head(h_e).squeeze(-1)
        z_q = self.qual_head(h_q).squeeze(-1)
        g = F.softmax(self.gate(torch.cat([h_e, h_q], dim=-1)), dim=-1)  # (B, 2)
        z = (g * torch.stack([z_e, z_q], dim=-1)).sum(dim=-1)
        # Pad gate to 3 cols for uniform downstream handling [det=0, emo, qual]
        gate3 = torch.cat([torch.zeros(g.size(0), 1, device=g.device), g], dim=-1)
        return {"logit": z, "gate_weights": gate3,
                "branch_logits": torch.stack([z, z_e, z_q], dim=-1)}


# ---------------------------------------------------------------------------
# Simple dataset (same interface as dataset.py)
# ---------------------------------------------------------------------------

class AblationDataset(Dataset):
    def __init__(self, df, det_col, emo_cols, qual_cols):
        self.det  = torch.tensor(df[det_col].values,  dtype=torch.float32).unsqueeze(1)
        self.emo  = torch.tensor(df[emo_cols].values, dtype=torch.float32)
        self.qual = torch.tensor(df[qual_cols].values, dtype=torch.float32)
        self.y    = torch.tensor(df["label_int"].values, dtype=torch.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return self.det[i], self.emo[i], self.qual[i], self.y[i]


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, all_logits, all_labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for det, emo, qual, y in loader:
            det, emo, qual, y = det.to(device), emo.to(device), qual.to(device), y.to(device)
            out  = model(det, emo, qual)
            loss = criterion(out["logit"], y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y)
            all_logits.append(out["logit"].detach().cpu())
            all_labels.append(y.detach().cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1 / (1 + np.exp(-logits))
    return total_loss / len(labels), compute_auc(labels, probs)


def train_variant_fold(k, model, train_df, test_df, cfg, device,
                       det_col, emo_cols, qual_cols, fold_dir):
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Scale
    all_feat = [det_col] + emo_cols + qual_cols
    scaler = StandardScaler()
    tr = train_df.copy(); va_idx = tr.sample(max(1, int(0.2*len(tr))),
                                              random_state=cfg["seed"]+k).index
    val_df = tr.loc[va_idx].reset_index(drop=True)
    tr_df  = tr.drop(va_idx).reset_index(drop=True)
    tr_df[all_feat]   = scaler.fit_transform(tr_df[all_feat])
    val_df[all_feat]  = scaler.transform(val_df[all_feat])
    te_df = test_df.copy()
    te_df[all_feat]   = scaler.transform(te_df[all_feat])

    n_pos = int((tr_df["label_int"] == 1).sum())
    n_neg = int((tr_df["label_int"] == 0).sum())
    pos_w = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                   weight_decay=cfg["weight_decay"])

    def _make_loader(df, shuffle):
        ds = AblationDataset(df, det_col, emo_cols, qual_cols)
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=0)

    tr_loader  = _make_loader(tr_df,  shuffle=True)
    val_loader = _make_loader(val_df, shuffle=False)
    te_loader  = _make_loader(te_df,  shuffle=False)

    best_val_auc, patience_counter, best_epoch = 0.0, 0, 0
    for epoch in range(cfg["n_epochs"]):
        run_epoch(model, tr_loader, criterion, optimizer, device, train=True)
        _, val_auc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        if val_auc > best_val_auc:
            best_val_auc = val_auc; best_epoch = epoch; patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "scaler": scaler, "val_auc": val_auc, "epoch": epoch},
                       fold_dir / "best.pt")
        else:
            patience_counter += 1
        if patience_counter >= cfg["patience"]:
            break

    # OOF predictions
    ckpt = torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    all_probs = []
    with torch.no_grad():
        for det, emo, qual, _ in te_loader:
            out = model(det.to(device), emo.to(device), qual.to(device))
            all_probs.append(torch.sigmoid(out["logit"]).cpu().numpy())
    probs = np.concatenate(all_probs)
    oof_auc = compute_auc(te_df["label_int"].values, probs)
    return probs, oof_auc, best_val_auc, scaler


# ---------------------------------------------------------------------------
# Run one variant across all folds
# ---------------------------------------------------------------------------

def run_variant(name, model_fn, trainval, test_df, cfg, device,
                det_col, emo_cols, qual_cols, abl_dir, logger):
    logger.info(f"--- Variant: {name} ---")
    gkf = GroupKFold(n_splits=cfg["n_folds"])
    groups = trainval["identity"].fillna(trainval["video_id"]).values
    X = trainval.index.values
    y = trainval["label_int"].values

    all_oof_probs, all_oof_labels, all_oof_vid, all_oof_fam = [], [], [], []
    fold_test_probs = []

    for k, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        set_seeds(cfg["seed"] + k)
        model = model_fn().to(device)
        fold_dir = abl_dir / name / f"fold_{k}"

        tr_df = trainval.iloc[tr_idx].reset_index(drop=True)
        te_df_fold = trainval.iloc[te_idx].reset_index(drop=True)

        probs, oof_auc, val_auc, scaler = train_variant_fold(
            k, model, tr_df, te_df_fold, cfg, device,
            det_col, emo_cols, qual_cols, fold_dir
        )
        all_oof_probs.extend(probs.tolist())
        all_oof_labels.extend(te_df_fold["label_int"].tolist())
        all_oof_vid.extend(te_df_fold["video_id"].tolist())
        all_oof_fam.extend(te_df_fold["forgery_family"].tolist())
        logger.info(f"  Fold {k}: val_AUC={val_auc:.4f}  OOF_AUC={oof_auc:.4f}")

        # Also run on test holdout
        all_feat = [det_col] + emo_cols + qual_cols
        test_scaled = test_df.copy()
        test_scaled[all_feat] = scaler.transform(test_df[all_feat])
        ds = AblationDataset(test_scaled, det_col, emo_cols, qual_cols)
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
        model.eval()
        tp = []
        with torch.no_grad():
            for det, emo, qual, _ in loader:
                out = model(det.to(device), emo.to(device), qual.to(device))
                tp.append(torch.sigmoid(out["logit"]).cpu().numpy())
        fold_test_probs.append(np.concatenate(tp))

    oof_auc_overall = compute_auc(np.array(all_oof_labels), np.array(all_oof_probs))
    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(
        np.array(all_oof_labels), np.array(all_oof_probs), n_iter=2000, seed=42
    )
    test_ensemble = np.stack(fold_test_probs).mean(axis=0)
    test_auc = compute_auc(test_df["label_int"].values, test_ensemble)
    test_auc_m, test_lo, test_hi = bootstrap_auc_ci(
        test_df["label_int"].values, test_ensemble, n_iter=2000, seed=42
    )
    eer = compute_eer(test_df["label_int"].values, test_ensemble)

    logger.info(f"  {name}: OOF AUC={oof_auc_overall:.4f}  Test AUC={test_auc:.4f}")

    oof_df = pd.DataFrame({
        "video_id": all_oof_vid,
        "label": all_oof_labels,
        "prediction": all_oof_probs,
        "forgery_family": all_oof_fam,
        "variant": name,
    })
    oof_df.to_csv(abl_dir / name / "oof_predictions.csv", index=False)

    return {
        "variant": name,
        "OOF_AUC": round(oof_auc_overall, 4),
        "OOF_AUC_ci_low": round(auc_lo, 4),
        "OOF_AUC_ci_high": round(auc_hi, 4),
        "Test_AUC": round(test_auc, 4),
        "Test_AUC_ci_low": round(test_lo, 4),
        "Test_AUC_ci_high": round(test_hi, 4),
        "Test_EER": round(eer, 4),
    }, test_ensemble


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seeds(42)
    cfg = load_config(str(CONFIG_PATH))
    out_dir = ROOT / cfg["paths"]["output_root"]
    pred_dir = out_dir / "predictions"
    table_dir = out_dir / "tables"
    fig_dir   = out_dir / "figures"
    log_dir   = out_dir / "logs"
    abl_dir   = out_dir / "ablation"
    for d in [table_dir, fig_dir, log_dir, abl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15.ablation", str(log_dir / "run.log"))
    logger.info("=== Stage 07: Ablation Study ===")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    emo_cols  = cfg["emotion_feature_cols"]
    qual_cols = cfg["quality_feature_cols"]
    det_col   = "detector_score"
    emotion_dim = len(emo_cols)
    quality_dim = len(qual_cols)

    trainval = pd.read_parquet(
        require_file(pred_dir / "trainval_feature_matrix.parquet", "Run 01 first")
    )
    test_df = pd.read_parquet(
        require_file(pred_dir / "test_feature_matrix.parquet", "Run 01 first")
    )

    # Define variants: name → model factory
    variants = [
        ("det_only",     lambda: DetectorOnly(embed_dim=cfg["embed_dim"], dropout=cfg["dropout"])),
        ("emo_only",     lambda: EmoOnly(emotion_dim=emotion_dim, embed_dim=cfg["embed_dim"], dropout=cfg["dropout"])),
        ("qual_only",    lambda: QualOnly(quality_dim=quality_dim, embed_dim=cfg["embed_dim"], dropout=cfg["dropout"])),
        ("no_detector",  lambda: NoDetector(emotion_dim=emotion_dim, quality_dim=quality_dim,
                                             embed_dim=cfg["embed_dim"], dropout=cfg["dropout"])),
    ]

    rows = []
    test_scores = {}   # variant → ensemble probs on test

    for name, model_fn in variants:
        t0 = time.time()
        row, test_ens = run_variant(
            name, model_fn, trainval, test_df, cfg, device,
            det_col, emo_cols, qual_cols, abl_dir, logger
        )
        elapsed = time.time() - t0
        row["time_sec"] = round(elapsed, 1)
        rows.append(row)
        test_scores[name] = test_ens
        print(f"[{name}] OOF={row['OOF_AUC']:.4f}  Test={row['Test_AUC']:.4f}  ({elapsed:.0f}s)")

    # Add full model row from existing OOF + test results
    oof_full = pd.read_csv(pred_dir / "final_exp15_oof_predictions.csv")
    test_full = pd.read_csv(pred_dir / "test_exp15_predictions.csv")
    full_oof_auc = compute_auc(oof_full["label"].values, oof_full["prediction"].values)
    full_oof_m, full_oof_lo, full_oof_hi = bootstrap_auc_ci(
        oof_full["label"].values, oof_full["prediction"].values, n_iter=2000, seed=42
    )
    full_test_auc = compute_auc(test_full["label"].values, test_full["prediction"].values)
    full_test_m, full_test_lo, full_test_hi = bootstrap_auc_ci(
        test_full["label"].values, test_full["prediction"].values, n_iter=2000, seed=42
    )
    full_eer = compute_eer(test_full["label"].values, test_full["prediction"].values)
    rows.insert(0, {
        "variant": "full (det+emo+qual)",
        "OOF_AUC": round(full_oof_auc, 4),
        "OOF_AUC_ci_low": round(full_oof_lo, 4),
        "OOF_AUC_ci_high": round(full_oof_hi, 4),
        "Test_AUC": round(full_test_auc, 4),
        "Test_AUC_ci_low": round(full_test_lo, 4),
        "Test_AUC_ci_high": round(full_test_hi, 4),
        "Test_EER": round(full_eer, 4),
        "time_sec": None,
    })
    test_scores["full (det+emo+qual)"] = test_full["prediction"].values

    # ----------------------------------------------------------------
    # Ablation table
    # ----------------------------------------------------------------
    abl_df = pd.DataFrame(rows)
    csv_out = table_dir / "final_exp15_ablation.csv"
    abl_df.to_csv(csv_out, index=False)
    abl_df.drop(columns=["time_sec"]).to_latex(
        table_dir / "final_exp15_ablation.tex", index=False,
        float_format="%.4f", na_rep="—"
    )
    logger.info(f"Ablation table saved: {csv_out}")

    # ----------------------------------------------------------------
    # ROC overlay figure
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {
        "full (det+emo+qual)": "steelblue",
        "no_detector": "darkorange",
        "emo_only": "forestgreen",
        "qual_only": "mediumpurple",
        "det_only": "crimson",
    }
    y_test = test_df["label_int"].values
    for row in rows:
        name = row["variant"]
        probs = test_scores[name]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = row["Test_AUC"]
        lw = 2.5 if name == "full (det+emo+qual)" else 1.5
        ls = "-" if name in ("full (det+emo+qual)", "no_detector") else "--"
        ax.plot(fpr, tpr, color=colors.get(name, "gray"), lw=lw, ls=ls,
                label=f"{name}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Ablation — ROC Curves on Test Holdout", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = fig_dir / "final_exp15_ablation_roc.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Ablation ROC saved: {fig_path}")

    # ----------------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Exp.15 — Ablation Study")
    print("=" * 72)
    print(abl_df[["variant", "OOF_AUC", "Test_AUC", "Test_EER"]].to_string(index=False))
    print("=" * 72)

    full_test = next(r for r in rows if r["variant"] == "full (det+emo+qual)")["Test_AUC"]
    no_det    = next((r for r in rows if r["variant"] == "no_detector"), None)
    det_only  = next((r for r in rows if r["variant"] == "det_only"), None)
    if no_det:
        delta = full_test - no_det["Test_AUC"]
        print(f"\nFull vs No-Detector: ΔAUC = {delta:+.4f}")
        if abs(delta) < 0.01:
            print("→ Detector is REDUNDANT given emotion+quality features.")
        elif delta > 0.02:
            print("→ Detector contributes meaningfully despite low gate weight.")
        else:
            print("→ Marginal detector contribution.")
    if det_only:
        print(f"Detector alone: AUC = {det_only['Test_AUC']:.4f}  "
              f"(UCF baseline was ~0.73)")
    print("=" * 72)


if __name__ == "__main__":
    main()
