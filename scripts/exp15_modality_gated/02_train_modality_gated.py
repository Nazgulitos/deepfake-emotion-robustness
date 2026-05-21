"""
Stage 02 — Train ModalityGatedFusion with 5-fold GroupKFold + full checkpointing/resume.

Reads:  outputs/predictions/final_feature_matrix.parquet
Writes: outputs/checkpoints/fold_{k}/  (best.pt, last.pt, state.json, DONE)
        outputs/predictions/final_exp15_oof_predictions.csv
        outputs/logs/training_curves.csv
        outputs/figures/final_exp15_training_curves.png
        outputs/tensorboard/fold_{k}/

Run from project root:
  python scripts/exp15_modality_gated/02_train_modality_gated.py
"""

import csv
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from dataset import make_loaders
from model import ModalityGatedFusion
from utils import (
    compute_auc,
    get_output_dir,
    get_project_root,
    hash_config,
    load_config,
    log_run_metadata,
    require_file,
    set_seeds,
    setup_logger,
)

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available (import error). CSV log and figures will still be saved.")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.rename(tmp, path)


def _save_last(fold_dir: Path, model, optimizer, scaler_sk, best_val_auc: float,
               patience_counter: int, epoch: int, state: dict) -> None:
    rng_states = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng_states["cuda"] = torch.cuda.get_rng_state()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler": scaler_sk,
            "rng_states": rng_states,
        },
        fold_dir / "last.pt",
    )
    state.update(
        {
            "current_epoch": epoch,
            "best_val_auc": best_val_auc,
            "patience_counter": patience_counter,
            "completed": False,
            "timestamp_last_save": datetime.now(timezone.utc).isoformat(),
        }
    )
    _atomic_json(fold_dir / "state.json", state)


def _save_best(fold_dir: Path, model, scaler_sk, val_auc: float, epoch: int) -> None:
    backup_dir = fold_dir.parent / "backup"
    backup_dir.mkdir(exist_ok=True)
    data = {
        "model_state_dict": model.state_dict(),
        "scaler": scaler_sk,
        "val_auc": val_auc,
        "epoch": epoch,
    }
    torch.save(data, fold_dir / "best.pt")
    torch.save(data, backup_dir / f"{fold_dir.name}_best.pt")


# ---------------------------------------------------------------------------
# Training one epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    all_logits, all_labels = [], []
    all_gate_weights = []

    with torch.set_grad_enabled(train):
        for det, emo, qual, labels in loader:
            det, emo, qual, labels = (
                det.to(device), emo.to(device), qual.to(device), labels.to(device)
            )
            out = model(det, emo, qual)
            loss = criterion(out["logit"], labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_logits.append(out["logit"].detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_gate_weights.append(out["gate_weights"].detach().cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    gate_w = torch.cat(all_gate_weights).numpy()

    probs = 1 / (1 + np.exp(-logits))
    auc = compute_auc(labels, probs)
    mean_loss = total_loss / len(labels)
    mean_gates = gate_w.mean(axis=0)  # [det, emo, qual]

    return mean_loss, auc, mean_gates


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def train_fold(
    k: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
    logger,
    curves_writer,
    config_hash: str,
) -> pd.DataFrame:
    fold_dir = out_dir / "checkpoints" / f"fold_{k}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = out_dir / "tensorboard" / f"fold_{k}"
    tb_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    emo_cols = cfg["emotion_feature_cols"]
    qual_cols = cfg["quality_feature_cols"]
    det_col = "detector_score"

    # ----------------------------------------------------------------
    # Resume logic
    # ----------------------------------------------------------------
    start_epoch = 0
    best_val_auc = 0.0
    patience_counter = 0
    best_epoch = 0

    state = {
        "fold": k,
        "config_hash": config_hash,
        "torch_version": torch.__version__,
        "completed": False,
        "current_epoch": -1,
        "best_val_auc": 0.0,
        "best_epoch": 0,
        "patience_counter": 0,
    }

    # Scaler fitted on train; needs to be built before any resume path checks model
    # We always refit scaler on the same train data (deterministic), so it's safe
    rng_seed = cfg["seed"] + k
    set_seeds(rng_seed)

    # 80/20 split within train portion
    val_size = max(1, int(0.2 * len(train_df)))
    val_idx = train_df.sample(val_size, random_state=cfg["seed"] + k).index
    val_df_fold = train_df.loc[val_idx].reset_index(drop=True)
    train_df_fold = train_df.drop(val_idx).reset_index(drop=True)

    # StandardScaler — fit on train only
    all_feat_cols = [det_col] + emo_cols + qual_cols
    scaler = StandardScaler()
    train_scaled = train_df_fold.copy()
    val_scaled = val_df_fold.copy()
    test_scaled = test_df.copy()

    train_scaled[all_feat_cols] = scaler.fit_transform(train_df_fold[all_feat_cols])
    val_scaled[all_feat_cols] = scaler.transform(val_df_fold[all_feat_cols])
    test_scaled[all_feat_cols] = scaler.transform(test_df[all_feat_cols])

    # Model
    emotion_dim = len(emo_cols)
    quality_dim = len(qual_cols)
    model = ModalityGatedFusion(
        emotion_dim=emotion_dim,
        quality_dim=quality_dim,
        embed_dim=cfg["embed_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    # Pos weight
    n_pos = int((train_scaled["label_int"] == 1).sum())
    n_neg = int((train_scaled["label_int"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    # Check for existing checkpoint
    state_path = fold_dir / "state.json"
    if state_path.exists():
        saved_state = json.load(open(state_path))
        if saved_state.get("config_hash") != config_hash:
            logger.warning(f"[Fold {k}] Config changed. Restarting fold from scratch.")
            shutil.rmtree(fold_dir)
            fold_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Try to resume
            resume_ok = False
            try:
                ckpt = torch.load(fold_dir / "last.pt", map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                # Restore RNG
                torch.set_rng_state(ckpt["rng_states"]["torch"])
                if torch.cuda.is_available() and "cuda" in ckpt["rng_states"]:
                    torch.cuda.set_rng_state(ckpt["rng_states"]["cuda"])
                np.random.set_state(ckpt["rng_states"]["numpy"])
                random.setstate(ckpt["rng_states"]["python"])
                start_epoch = saved_state["current_epoch"] + 1
                best_val_auc = saved_state["best_val_auc"]
                patience_counter = saved_state["patience_counter"]
                best_epoch = saved_state.get("best_epoch", 0)
                state = saved_state
                resume_ok = True
                logger.info(f"[Fold {k}] Resumed from epoch {start_epoch}")
            except Exception as e:
                logger.warning(f"[Fold {k}] last.pt corrupted: {e}. Trying best.pt...")
                try:
                    ckpt = torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
                    model.load_state_dict(ckpt["model_state_dict"])
                    start_epoch = ckpt["epoch"] + 1
                    best_val_auc = ckpt["val_auc"]
                    best_epoch = ckpt["epoch"]
                    resume_ok = True
                    logger.info(f"[Fold {k}] Resumed from best.pt (epoch {start_epoch})")
                except Exception as e2:
                    logger.error(f"[Fold {k}] Both checkpoints corrupted: {e2}. Restarting.")
                    shutil.rmtree(fold_dir)
                    fold_dir.mkdir(parents=True, exist_ok=True)

    # DataLoaders
    train_loader, val_loader = make_loaders(
        train_scaled, val_scaled, det_col, emo_cols, qual_cols,
        batch_size=cfg["batch_size"], seed=cfg["seed"]
    )

    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=str(tb_dir))

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    t0 = time.time()
    n_epochs = cfg["n_epochs"]
    patience = cfg["patience"]

    for epoch in range(start_epoch, n_epochs):
        tr_loss, tr_auc, tr_gates = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_auc, vl_gates = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        improved = vl_auc > best_val_auc
        if improved:
            best_val_auc = vl_auc
            best_epoch = epoch
            patience_counter = 0
            _save_best(fold_dir, model, scaler, vl_auc, epoch)
            state["best_epoch"] = epoch
        else:
            patience_counter += 1

        _save_last(fold_dir, model, optimizer, scaler, best_val_auc, patience_counter, epoch, state)

        # CSV log
        lr_now = optimizer.param_groups[0]["lr"]
        curves_writer.writerow([
            k, epoch, f"{tr_loss:.6f}", f"{tr_auc:.6f}",
            f"{vl_loss:.6f}", f"{vl_auc:.6f}", f"{best_val_auc:.6f}",
            f"{lr_now:.2e}",
            f"{tr_gates[0]:.4f}", f"{tr_gates[1]:.4f}", f"{tr_gates[2]:.4f}",
        ])

        if writer:
            writer.add_scalar("train/loss", tr_loss, epoch)
            writer.add_scalar("train/auc", tr_auc, epoch)
            writer.add_scalar("val/loss", vl_loss, epoch)
            writer.add_scalar("val/auc", vl_auc, epoch)
            writer.add_scalar("val/auc_best", best_val_auc, epoch)
            writer.add_scalar("gate_weights/detector", vl_gates[0], epoch)
            writer.add_scalar("gate_weights/emotion", vl_gates[1], epoch)
            writer.add_scalar("gate_weights/quality", vl_gates[2], epoch)

        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(
                f"[Fold {k}] Epoch {epoch+1:3d}/{n_epochs} | "
                f"train_loss={tr_loss:.4f} train_auc={tr_auc:.4f} | "
                f"val_loss={vl_loss:.4f} val_auc={vl_auc:.4f} | "
                f"gate=[d:{vl_gates[0]:.2f} e:{vl_gates[1]:.2f} q:{vl_gates[2]:.2f}] | "
                f"lr={lr_now:.2e}"
            )

        if patience_counter >= patience:
            print(
                f"[Fold {k}] Early stopping at epoch {epoch+1} | "
                f"best_val_auc={best_val_auc:.4f} at epoch {best_epoch+1}"
            )
            break

    elapsed_total = time.time() - t0

    # ----------------------------------------------------------------
    # OOF predictions on test fold
    # ----------------------------------------------------------------
    best_ckpt = torch.load(fold_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels_oof = [], []
    all_gates_oof, all_branch_logits_oof = [], []

    from torch.utils.data import DataLoader
    from dataset import ModalityDataset
    test_ds = ModalityDataset(test_scaled, det_col, emo_cols, qual_cols)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    with torch.no_grad():
        for det, emo, qual, labels in test_loader:
            det, emo, qual = det.to(device), emo.to(device), qual.to(device)
            out = model(det, emo, qual)
            probs = torch.sigmoid(out["logit"]).cpu().numpy()
            all_preds.extend(probs.tolist())
            all_labels_oof.extend(labels.numpy().tolist())
            all_gates_oof.append(out["gate_weights"].cpu().numpy())
            all_branch_logits_oof.append(out["branch_logits"].cpu().numpy())

    gates_arr = np.concatenate(all_gates_oof, axis=0)
    branches_arr = np.concatenate(all_branch_logits_oof, axis=0)
    oof_auc = compute_auc(np.array(all_labels_oof), np.array(all_preds))

    oof_df = test_df[["video_id", "label_int", "forgery_family", "dominant_emotion"]].copy()
    oof_df = oof_df.rename(columns={"label_int": "label"})
    oof_df["prediction"] = all_preds
    oof_df["gate_det"] = gates_arr[:, 0]
    oof_df["gate_emo"] = gates_arr[:, 1]
    oof_df["gate_qual"] = gates_arr[:, 2]
    oof_df["branch_det_logit"] = branches_arr[:, 0]
    oof_df["branch_emo_logit"] = branches_arr[:, 1]
    oof_df["branch_qual_logit"] = branches_arr[:, 2]
    oof_df["fold"] = k

    # Save per-fold OOF
    oof_df.to_csv(fold_dir / "oof_predictions.csv", index=False)

    # TensorBoard fold-level scalars
    if writer:
        writer.add_scalar(f"fold_{k}/best_val_auc", best_val_auc, 0)
        writer.add_scalar(f"fold_{k}/test_oof_auc", oof_auc, 0)
        writer.add_scalar(f"fold_{k}/n_epochs_trained", epoch + 1, 0)
        writer.close()

    # Mark fold done
    (fold_dir / "DONE").touch()

    mean_gates = gates_arr.mean(axis=0)
    mins, secs = divmod(int(elapsed_total), 60)
    print(f"\n{'='*32}")
    print(f"[Fold {k+1}/5] Training complete")
    print(f"{'='*32}")
    print(f"  Best val AUC:       {best_val_auc:.4f} (epoch {best_epoch+1})")
    print(f"  Test OOF AUC:       {oof_auc:.4f}")
    print(f"  Epochs trained:     {epoch+1}/{n_epochs}")
    print(f"  Total time:         {mins}m {secs}s")
    print(f"  Mean gate weights:  det={mean_gates[0]:.2f}  emo={mean_gates[1]:.2f}  qual={mean_gates[2]:.2f}")
    print(f"{'='*32}\n")
    logger.info(f"[Fold {k}] OOF AUC={oof_auc:.4f} best_val_AUC={best_val_auc:.4f}")

    return oof_df


# ---------------------------------------------------------------------------
# Post-training figures
# ---------------------------------------------------------------------------

def plot_training_curves(curves_csv: Path, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(curves_csv)
    folds = sorted(df["fold"].unique())
    cmap = plt.cm.tab10

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exp.15 Training Curves — ModalityGatedFusion", fontsize=13)

    # (a) loss
    ax = axes[0, 0]
    for i, k in enumerate(folds):
        sub = df[df["fold"] == k]
        ax.plot(sub["epoch"], sub["train_loss"], color=cmap(i), alpha=0.6, linestyle="--")
        ax.plot(sub["epoch"], sub["val_loss"], color=cmap(i), alpha=0.9, label=f"fold {k}")
    ax.set_title("(a) Train/Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend(fontsize=7)

    # (b) AUC
    ax = axes[0, 1]
    for i, k in enumerate(folds):
        sub = df[df["fold"] == k]
        ax.plot(sub["epoch"], sub["train_auc"], color=cmap(i), alpha=0.6, linestyle="--")
        ax.plot(sub["epoch"], sub["val_auc"], color=cmap(i), alpha=0.9, label=f"fold {k}")
    ax.set_title("(b) Train/Val AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=7)

    # (c) Best val AUC progression
    ax = axes[1, 0]
    for i, k in enumerate(folds):
        sub = df[df["fold"] == k]
        ax.plot(sub["epoch"], sub["val_auc_best"], color=cmap(i), alpha=0.9, label=f"fold {k}")
    ax.set_title("(c) Best Val AUC Progression")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Best Val AUC")
    ax.legend(fontsize=7)

    # (d) Mean gate weights
    ax = axes[1, 1]
    # Average across folds per epoch (use epoch as common x-axis)
    max_ep = df.groupby("fold")["epoch"].max().min()
    sub = df[df["epoch"] <= max_ep]
    gate_mean = sub.groupby("epoch")[["gate_det_mean", "gate_emo_mean", "gate_qual_mean"]].mean()
    ax.plot(gate_mean.index, gate_mean["gate_det_mean"], label="Detector", color="steelblue")
    ax.plot(gate_mean.index, gate_mean["gate_emo_mean"], label="Emotion", color="darkorange")
    ax.plot(gate_mean.index, gate_mean["gate_qual_mean"], label="Quality", color="forestgreen")
    ax.set_title("(d) Mean Gate Weights over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gate Weight")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Training curves saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seeds(42)
    cfg = load_config(str(CONFIG_PATH))
    out_dir = ROOT / cfg["paths"]["output_root"]
    pred_dir = out_dir / "predictions"
    log_dir = out_dir / "logs"
    fig_dir = out_dir / "figures"
    for d in [pred_dir, log_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15.train", str(log_dir / "run.log"))
    log_run_metadata(logger, cfg, str(CONFIG_PATH))
    logger.info("=== Stage 02: Train ModalityGatedFusion ===")

    config_hash = hash_config(cfg)

    # Feature matrix — trainval only (test holdout is kept separate)
    fm_path = require_file(pred_dir / "trainval_feature_matrix.parquet",
                           "Run 01_prepare_features.py first")
    df = pd.read_parquet(fm_path)
    logger.info(f"Loaded trainval feature matrix: {df.shape}")

    emo_cols = cfg["emotion_feature_cols"]
    qual_cols = cfg["quality_feature_cols"]

    # Verify columns
    for col in (["video_id", "label_int", "forgery_family", "dominant_emotion", "identity", "detector_score"]
                + emo_cols + qual_cols):
        if col not in df.columns:
            raise RuntimeError(f"Missing column in feature matrix: '{col}'")

    # CSV log writer (append mode for resume)
    curves_csv = log_dir / "training_curves.csv"
    curves_file_exists = curves_csv.exists()
    curves_file = open(curves_csv, "a", newline="")
    curves_writer = csv.writer(curves_file)
    if not curves_file_exists:
        curves_writer.writerow([
            "fold", "epoch", "train_loss", "train_auc",
            "val_loss", "val_auc", "val_auc_best", "lr",
            "gate_det_mean", "gate_emo_mean", "gate_qual_mean",
        ])

    # GroupKFold
    # Real videos have no identity (NaN) — fill with video_id so each real video
    # is its own group, ensuring they distribute across folds without leaking fakes.
    groups = df["identity"].fillna(df["video_id"]).values
    gkf = GroupKFold(n_splits=cfg["n_folds"])
    X = df.index.values
    y = df["label_int"].values

    all_oof = []

    for k, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        fold_dir = out_dir / "checkpoints" / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Skip completed folds — load existing OOF
        if (fold_dir / "DONE").exists():
            logger.info(f"[Fold {k}] Already completed. Loading OOF predictions.")
            existing_oof = pd.read_csv(fold_dir / "oof_predictions.csv")
            all_oof.append(existing_oof)
            continue

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        logger.info(f"[Fold {k}] train={len(train_df)} test={len(test_df)}")

        set_seeds(cfg["seed"] + k)
        oof_df = train_fold(k, train_df, test_df, cfg, out_dir, logger, curves_writer, config_hash)
        all_oof.append(oof_df)

    curves_file.close()

    # ----------------------------------------------------------------
    # Consolidation
    # ----------------------------------------------------------------
    for k in range(cfg["n_folds"]):
        done_path = out_dir / "checkpoints" / f"fold_{k}" / "DONE"
        if not done_path.exists():
            raise RuntimeError(
                f"Fold {k} did not complete. Re-run this script to resume."
            )

    final_oof = pd.concat(all_oof, ignore_index=True)
    oof_path = pred_dir / "final_exp15_oof_predictions.csv"
    final_oof.to_csv(oof_path, index=False)
    logger.info(f"OOF predictions saved: {oof_path} ({len(final_oof)} rows)")

    overall_auc = compute_auc(final_oof["label"].values, final_oof["prediction"].values)
    per_fold_auc = final_oof.groupby("fold").apply(
        lambda g: compute_auc(g["label"].values, g["prediction"].values)
    )
    print("\nPer-fold AUC:")
    for k, auc in per_fold_auc.items():
        print(f"  Fold {k}: {auc:.4f}")
    print(f"Overall OOF AUC: {overall_auc:.4f}")
    logger.info(f"Overall OOF AUC: {overall_auc:.4f}")

    # Training curves figure
    plot_training_curves(
        curves_csv,
        fig_dir / "final_exp15_training_curves.png",
    )


if __name__ == "__main__":
    main()
