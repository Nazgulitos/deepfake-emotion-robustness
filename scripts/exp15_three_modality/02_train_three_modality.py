"""
Stage 02 — Train ThreeModalityGated with 5-fold GroupKFold CV.

Uses predefined train/val/test split:
  - trainval (635 videos, split∈{train,val}) → 5-fold GroupKFold by identity
  - test (155 videos) → never seen during training

Per-fold checkpointing with full resume support:
  outputs/checkpoints/fold_{k}/best.pt   — best val-AUC model
  outputs/checkpoints/fold_{k}/last.pt   — latest epoch (with RNG state)
  outputs/checkpoints/fold_{k}/state.json — atomic progress file
  outputs/checkpoints/fold_{k}/DONE       — sentinel written after fold completes

After all 5 folds saves OOF predictions:
  outputs/predictions/trainval_oof_predictions.csv

Run from project root:
  python scripts/exp15_three_modality/02_train_three_modality.py
"""

import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import ThreeModalityDataset
from model import ThreeModalityGated
from utils import (
    bootstrap_auc_ci,
    compute_auc,
    get_project_root,
    hash_config,
    load_config,
    require_file,
    set_seeds,
    setup_logger,
)

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()


# ── Training helpers ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x_q, x_s, x_t, y in loader:
        x_q, x_s, x_t, y = x_q.to(device), x_s.to(device), x_t.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x_q, x_s, x_t)
        loss = criterion(out["logit"], y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for x_q, x_s, x_t, y in loader:
        x_q, x_s, x_t, y = x_q.to(device), x_s.to(device), x_t.to(device), y.to(device)
        out = model(x_q, x_s, x_t)
        loss = criterion(out["logit"], y)
        total_loss += loss.item() * len(y)
        all_probs.append(torch.sigmoid(out["logit"]).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auc = compute_auc(labels, probs)
    return total_loss / len(loader.dataset), auc, probs, labels


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


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_state(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


def load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def save_checkpoint(path: Path, model, optimizer, scaler, epoch, best_auc, val_auc,
                    include_rng: bool = False) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler": scaler,
        "best_val_auc": best_auc,
        "val_auc": val_auc,
    }
    if include_rng:
        ckpt["rng_state"] = {
            "python": None,
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    torch.save(ckpt, path)


def load_checkpoint(path: Path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "rng_state" in ckpt and ckpt["rng_state"] is not None:
        rs = ckpt["rng_state"]
        if rs.get("numpy") is not None:
            np.random.set_state(rs["numpy"])
        if rs.get("torch") is not None:
            torch.set_rng_state(rs["torch"])
        if rs.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rs["cuda"])
    return ckpt


# ── Main ───────────────────────────────────────────────────────────────────────

def train_fold(
    k: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    qual_cols: list,
    emo_static_cols: list,
    emo_temporal_cols: list,
    all_feat_cols: list,
    cfg: dict,
    ckpt_dir: Path,
    log_dir: Path,
    device: torch.device,
    logger,
    cfg_hash: str,
) -> dict:
    """Train one fold. Returns dict with val_auc, best_epoch, probs, gates, branches."""

    fold_dir = ckpt_dir / f"fold_{k}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    done_path = fold_dir / "DONE"
    state_path = fold_dir / "state.json"
    best_path = fold_dir / "best.pt"
    last_path = fold_dir / "last.pt"
    curves_path = log_dir / f"training_curves_fold_{k}.csv"

    # ── Already done? ──────────────────────────────────────────────────────────
    if done_path.exists():
        state = load_state(state_path)
        logger.info(f"Fold {k}: DONE sentinel found — loading best checkpoint")
        scaler = torch.load(best_path, map_location=device, weights_only=False)["scaler"]
        val_scaled = val_df.copy()
        val_scaled[all_feat_cols] = scaler.transform(val_df[all_feat_cols])
        model = ThreeModalityGated(
            quality_dim=len(qual_cols),
            emo_static_dim=len(emo_static_cols),
            emo_temporal_dim=len(emo_temporal_cols),
            embed_dim=cfg["embed_dim"],
            gate_hidden=cfg["gate_hidden"],
            dropout=0.0,
        ).to(device)
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"]
        )
        probs, gates, branches = run_inference(
            model, val_scaled, qual_cols, emo_static_cols, emo_temporal_cols,
            cfg["batch_size"], device,
        )
        return {
            "val_auc": state["best_val_auc"] if state else compute_auc(val_df["label_int"].values, probs),
            "probs": probs,
            "gates": gates,
            "branches": branches,
        }

    # ── Scaler ────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    train_scaled[all_feat_cols] = scaler.fit_transform(train_df[all_feat_cols])
    val_scaled[all_feat_cols] = scaler.transform(val_df[all_feat_cols])

    # ── Model / optimizer ─────────────────────────────────────────────────────
    model = ThreeModalityGated(
        quality_dim=len(qual_cols),
        emo_static_dim=len(emo_static_cols),
        emo_temporal_dim=len(emo_temporal_cols),
        embed_dim=cfg["embed_dim"],
        gate_hidden=cfg["gate_hidden"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    # pos_weight
    n_neg = int((train_df["label_int"] == 0).sum())
    n_pos = int((train_df["label_int"] == 1).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_auc = -1.0
    patience_count = 0

    state = load_state(state_path)
    if state and state.get("config_hash") == cfg_hash and last_path.exists():
        try:
            ckpt = load_checkpoint(last_path, model, optimizer, device)
            start_epoch = ckpt["epoch"] + 1
            best_val_auc = ckpt["best_val_auc"]
            logger.info(
                f"Fold {k}: Resuming from epoch {start_epoch}, best_val_auc={best_val_auc:.4f}"
            )
        except Exception as e:
            logger.warning(f"Fold {k}: Resume failed ({e}), restarting")
            start_epoch = 0
            best_val_auc = -1.0
            model = ThreeModalityGated(
                quality_dim=len(qual_cols), emo_static_dim=len(emo_static_cols),
                emo_temporal_dim=len(emo_temporal_cols), embed_dim=cfg["embed_dim"],
                gate_hidden=cfg["gate_hidden"], dropout=cfg["dropout"],
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
            )

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_ds = ThreeModalityDataset(train_scaled, qual_cols, emo_static_cols, emo_temporal_cols)
    val_ds = ThreeModalityDataset(val_scaled, qual_cols, emo_static_cols, emo_temporal_cols)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = HERE / "outputs" / "tensorboard" / f"fold_{k}"
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(str(tb_dir))
    except Exception:
        pass

    # ── Training loop ─────────────────────────────────────────────────────────
    csv_header_written = curves_path.exists() and start_epoch > 0
    with open(curves_path, "a", newline="") as csvf:
        writer = csv.writer(csvf)
        if not csv_header_written:
            writer.writerow(["fold", "epoch", "train_loss", "val_loss", "val_auc",
                              "gate_q", "gate_s", "gate_t"])

        for epoch in range(start_epoch, cfg["n_epochs"]):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_auc, val_probs, val_labels = eval_epoch(model, val_loader, criterion, device)

            # Mean gate weights on val set
            model.eval()
            gates_list = []
            with torch.no_grad():
                for x_q, x_s, x_t, _ in val_loader:
                    x_q, x_s, x_t = x_q.to(device), x_s.to(device), x_t.to(device)
                    out = model(x_q, x_s, x_t)
                    gates_list.append(out["gate_weights"].cpu().numpy())
            mean_gates = np.concatenate(gates_list).mean(axis=0)

            writer.writerow([k, epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                              f"{val_auc:.6f}"] + [f"{g:.4f}" for g in mean_gates])

            if tb_writer:
                try:
                    tb_writer.add_scalar(f"fold{k}/train_loss", train_loss, epoch)
                    tb_writer.add_scalar(f"fold{k}/val_loss", val_loss, epoch)
                    tb_writer.add_scalar(f"fold{k}/val_auc", val_auc, epoch)
                    tb_writer.add_scalar(f"fold{k}/gate_q", mean_gates[0], epoch)
                    tb_writer.add_scalar(f"fold{k}/gate_s", mean_gates[1], epoch)
                    tb_writer.add_scalar(f"fold{k}/gate_t", mean_gates[2], epoch)
                except Exception:
                    pass

            # Best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_count = 0
                save_checkpoint(best_path, model, optimizer, scaler, epoch, best_val_auc, val_auc)
            else:
                patience_count += 1

            # Last checkpoint (every epoch, with RNG state)
            save_checkpoint(last_path, model, optimizer, scaler, epoch, best_val_auc, val_auc,
                             include_rng=True)
            save_state(state_path, {
                "fold": k,
                "epoch": epoch,
                "best_val_auc": best_val_auc,
                "patience": patience_count,
                "config_hash": cfg_hash,
            })

            if epoch % 5 == 0 or epoch == cfg["n_epochs"] - 1:
                elapsed = time.time() - t0
                logger.info(
                    f"  Fold {k} | Ep {epoch:3d} | "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"val_AUC={val_auc:.4f}  best={best_val_auc:.4f}  "
                    f"gates=[q={mean_gates[0]:.3f} s={mean_gates[1]:.3f} t={mean_gates[2]:.3f}]  "
                    f"patience={patience_count}/{cfg['patience']}  {elapsed:.1f}s"
                )

            if patience_count >= cfg["patience"]:
                logger.info(f"  Fold {k}: Early stopping at epoch {epoch}")
                break

    if tb_writer:
        try:
            tb_writer.close()
        except Exception:
            pass

    # ── Load best model for inference ─────────────────────────────────────────
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.dropout = 0.0  # disable dropout for inference

    probs, gates, branches = run_inference(
        model, val_scaled, qual_cols, emo_static_cols, emo_temporal_cols,
        cfg["batch_size"], device,
    )

    done_path.touch()
    logger.info(f"  Fold {k}: DONE. best_val_AUC={best_val_auc:.4f}")

    return {"val_auc": best_val_auc, "probs": probs, "gates": gates, "branches": branches}


def main():
    set_seeds(42)
    cfg = load_config(str(CONFIG_PATH))
    cfg_hash = hash_config(cfg)

    out_dir = ROOT / cfg["paths"]["output_root"]
    pred_dir = out_dir / "predictions"
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    fig_dir = out_dir / "figures"
    for d in [pred_dir, ckpt_dir, log_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15_tm.train", str(log_dir / "run.log"))
    logger.info("=== Stage 02: Train ThreeModalityGated ===")
    logger.info(f"Config hash: {cfg_hash}")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    qual_cols = cfg["quality_features"]
    emo_static_cols = cfg["emotion_static_features"]
    emo_temporal_base = [c for c in cfg["emotion_temporal_features"] if not c.startswith("std_score_")]
    emo_temporal_std = [c for c in cfg["emotion_temporal_features"] if c.startswith("std_score_")]
    emo_temporal_cols = emo_temporal_base + emo_temporal_std
    all_feat_cols = qual_cols + emo_static_cols + emo_temporal_cols

    # Load trainval
    trainval_path = require_file(pred_dir / "trainval_feature_matrix.parquet",
                                  "Run 01_prepare_features.py first")
    df = pd.read_parquet(trainval_path)
    logger.info(f"TrainVal: {len(df)} videos  (fake={int(df['label_int'].sum())}  "
                f"real={int((df['label_int']==0).sum())})")

    # Identity-aware groups (real videos get unique group = video_id)
    groups = df["identity"].fillna(df["video_id"]).values

    kf = GroupKFold(n_splits=cfg["n_folds"])
    oof_records = []

    for k, (train_idx, val_idx) in enumerate(kf.split(df, groups=groups)):
        logger.info(f"\n{'='*60}\nFold {k}  (train={len(train_idx)}  val={len(val_idx)})")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        result = train_fold(
            k, train_df, val_df,
            qual_cols, emo_static_cols, emo_temporal_cols, all_feat_cols,
            cfg, ckpt_dir, log_dir, device, logger, cfg_hash,
        )

        # Collect OOF predictions
        for i, idx in enumerate(val_idx):
            row = df.iloc[idx]
            oof_records.append({
                "video_id": row["video_id"],
                "label": row["label"],
                "label_int": row["label_int"],
                "forgery_family": row.get("forgery_family", None),
                "dominant_emotion": row.get("dominant_emotion", None),
                "fold": k,
                "split_type": "trainval_oof",
                "prediction": float(result["probs"][i]),
                "gate_q": float(result["gates"][i, 0]),
                "gate_s": float(result["gates"][i, 1]),
                "gate_t": float(result["gates"][i, 2]),
                "branch_q_logit": float(result["branches"][i, 0]),
                "branch_s_logit": float(result["branches"][i, 1]),
                "branch_t_logit": float(result["branches"][i, 2]),
            })

        logger.info(f"Fold {k} val AUC: {result['val_auc']:.4f}")

    # ── OOF predictions ────────────────────────────────────────────────────────
    oof_df = pd.DataFrame(oof_records)
    oof_df.to_csv(pred_dir / "trainval_oof_predictions.csv", index=False)

    oof_auc = compute_auc(oof_df["label_int"].values, oof_df["prediction"].values)
    logger.info(f"\nOOF AUC (all folds): {oof_auc:.4f}")

    # ── Training curves figure ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        curves_files = list(log_dir.glob("training_curves_fold_*.csv"))
        if curves_files:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for cf in sorted(curves_files):
                cdf = pd.read_csv(cf)
                fold_id = cdf["fold"].iloc[0]
                axes[0].plot(cdf["epoch"], cdf["train_loss"], alpha=0.7, label=f"fold {fold_id}")
                axes[1].plot(cdf["epoch"], cdf["val_auc"], alpha=0.7, label=f"fold {fold_id}")
                for i, (gate_col, color) in enumerate(
                    zip(["gate_q", "gate_s", "gate_t"], ["tab:blue", "tab:orange", "tab:green"])
                ):
                    if gate_col in cdf.columns:
                        axes[2].plot(cdf["epoch"], cdf[gate_col], alpha=0.7,
                                     color=color, label=f"{gate_col} f{fold_id}" if fold_id == 0 else "")
            axes[0].set_title("Training Loss", fontsize=11)
            axes[1].set_title("Validation AUC", fontsize=11)
            axes[2].set_title("Gate Weights", fontsize=11)
            for ax in axes:
                ax.set_xlabel("Epoch")
                ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(fig_dir / "final_exp15_training_curves.png", dpi=300)
            plt.close(fig)
            logger.info("Training curves figure saved")
    except Exception as e:
        logger.warning(f"Training curves figure failed: {e}")

    print(f"\n{'='*60}")
    print(f"Training complete. OOF AUC: {oof_auc:.4f}")
    print(f"OOF predictions: {pred_dir / 'trainval_oof_predictions.csv'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
