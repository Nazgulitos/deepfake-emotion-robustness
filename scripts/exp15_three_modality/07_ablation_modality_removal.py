"""
Stage 07 — Ablation study: remove one modality at a time.

Trains 4 model configurations × 5 folds on trainval, evaluates on test holdout:
  full                   : quality + emotion_static + emotion_temporal (3-branch)
  no_quality             : emotion_static + emotion_temporal (2-branch)
  no_emotion_static      : quality + emotion_temporal (2-branch)
  no_emotion_temporal    : quality + emotion_static (2-branch)

Results saved to:
  outputs/tables/final_exp15_ablation_summary.csv  (+.tex)
  outputs/figures/final_exp15_ablation_bars.png
  outputs/stats/final_exp15_permutation_full_vs_ablation.json

Run from project root:
  python scripts/exp15_three_modality/07_ablation_modality_removal.py
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

from dataset import ThreeModalityDataset, TwoModalityDataset
from model import ThreeModalityGated, TwoModalityGated
from utils import (
    bootstrap_auc_ci,
    compute_auc,
    get_project_root,
    hash_config,
    load_config,
    permutation_test_auc,
    require_file,
    set_seeds,
    setup_logger,
)

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()

# Ablation configs: name → (modality_a, modality_b, missing)
ABLATION_CONFIGS = ["full", "no_quality", "no_emotion_static", "no_emotion_temporal"]


def get_cols_for_config(config_name: str, qual_cols, emo_static_cols, emo_temporal_cols):
    if config_name == "full":
        return qual_cols, emo_static_cols, emo_temporal_cols
    elif config_name == "no_quality":
        return emo_static_cols, emo_temporal_cols, None
    elif config_name == "no_emotion_static":
        return qual_cols, emo_temporal_cols, None
    elif config_name == "no_emotion_temporal":
        return qual_cols, emo_static_cols, None
    else:
        raise ValueError(f"Unknown config: {config_name}")


def build_model_for_config(config_name: str, qual_dim, emo_static_dim, emo_temporal_dim,
                            embed_dim, gate_hidden, dropout):
    from model import build_ablation_model
    return build_ablation_model(config_name, qual_dim, emo_static_dim, emo_temporal_dim,
                                 embed_dim=embed_dim, gate_hidden=gate_hidden, dropout=dropout)


def make_dataset(config_name, df, qual_cols, emo_static_cols, emo_temporal_cols):
    if config_name == "full":
        return ThreeModalityDataset(df, qual_cols, emo_static_cols, emo_temporal_cols)
    cols_a, cols_b, _ = get_cols_for_config(config_name, qual_cols, emo_static_cols, emo_temporal_cols)
    return TwoModalityDataset(df, cols_a, cols_b)


def train_epoch_generic(model, loader, optimizer, criterion, device, is_three=True):
    model.train()
    total_loss = 0.0
    for batch in loader:
        if is_three:
            x_a, x_b, x_c, y = batch
            x_a, x_b, x_c, y = x_a.to(device), x_b.to(device), x_c.to(device), y.to(device)
            out = model(x_a, x_b, x_c)
        else:
            x_a, x_b, y = batch
            x_a, x_b, y = x_a.to(device), x_b.to(device), y.to(device)
            out = model(x_a, x_b)
        optimizer.zero_grad()
        loss = criterion(out["logit"], y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch_generic(model, loader, criterion, device, is_three=True):
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for batch in loader:
        if is_three:
            x_a, x_b, x_c, y = batch
            x_a, x_b, x_c, y = x_a.to(device), x_b.to(device), x_c.to(device), y.to(device)
            out = model(x_a, x_b, x_c)
        else:
            x_a, x_b, y = batch
            x_a, x_b, y = x_a.to(device), x_b.to(device), y.to(device)
            out = model(x_a, x_b)
        loss = criterion(out["logit"], y)
        total_loss += loss.item() * len(y)
        all_probs.append(torch.sigmoid(out["logit"]).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return total_loss / len(loader.dataset), compute_auc(labels, probs), probs


@torch.no_grad()
def run_inference_generic(model, df, config_name, qual_cols, emo_static_cols,
                           emo_temporal_cols, batch_size, device):
    ds = make_dataset(config_name, df, qual_cols, emo_static_cols, emo_temporal_cols)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    is_three = (config_name == "full")
    model.eval()
    all_probs = []
    for batch in loader:
        if is_three:
            x_a, x_b, x_c, _ = batch
            x_a, x_b, x_c = x_a.to(device), x_b.to(device), x_c.to(device)
            out = model(x_a, x_b, x_c)
        else:
            x_a, x_b, _ = batch
            x_a, x_b = x_a.to(device), x_b.to(device)
            out = model(x_a, x_b)
        all_probs.append(torch.sigmoid(out["logit"]).cpu().numpy())
    return np.concatenate(all_probs)


def get_feat_cols(config_name, qual_cols, emo_static_cols, emo_temporal_cols):
    if config_name == "full":
        return qual_cols + emo_static_cols + emo_temporal_cols
    elif config_name == "no_quality":
        return emo_static_cols + emo_temporal_cols
    elif config_name == "no_emotion_static":
        return qual_cols + emo_temporal_cols
    elif config_name == "no_emotion_temporal":
        return qual_cols + emo_static_cols


def train_config(
    config_name: str,
    df_trainval: pd.DataFrame,
    df_test: pd.DataFrame,
    qual_cols, emo_static_cols, emo_temporal_cols,
    cfg: dict,
    ckpt_dir: Path,
    log_dir: Path,
    device: torch.device,
    logger,
) -> dict:
    """Train config over 5 folds, return OOF + test AUCs."""

    config_ckpt_dir = ckpt_dir / "ablation" / config_name
    config_ckpt_dir.mkdir(parents=True, exist_ok=True)

    is_three = (config_name == "full")
    feat_cols = get_feat_cols(config_name, qual_cols, emo_static_cols, emo_temporal_cols)

    q_dim = len(qual_cols)
    s_dim = len(emo_static_cols)
    t_dim = len(emo_temporal_cols)

    groups = df_trainval["identity"].fillna(df_trainval["video_id"]).values
    kf = GroupKFold(n_splits=cfg["n_folds"])

    oof_probs = np.zeros(len(df_trainval))
    fold_test_probs = []

    for k, (train_idx, val_idx) in enumerate(kf.split(df_trainval, groups=groups)):
        done_path = config_ckpt_dir / f"fold_{k}_DONE"
        best_path = config_ckpt_dir / f"fold_{k}_best.pt"

        train_df = df_trainval.iloc[train_idx].reset_index(drop=True)
        val_df = df_trainval.iloc[val_idx].reset_index(drop=True)

        if done_path.exists() and best_path.exists():
            logger.info(f"  [{config_name}] Fold {k}: already done, loading")
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            scaler = ckpt["scaler"]
            val_scaled = val_df.copy()
            val_scaled[feat_cols] = scaler.transform(val_df[feat_cols])
            test_scaled = df_test.copy()
            test_scaled[feat_cols] = scaler.transform(df_test[feat_cols])

            model = build_model_for_config(config_name, q_dim, s_dim, t_dim,
                                            cfg["embed_dim"], cfg["gate_hidden"], 0.0)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)
            val_probs = run_inference_generic(model, val_scaled, config_name,
                                              qual_cols, emo_static_cols, emo_temporal_cols,
                                              cfg["batch_size"], device)
            test_fold_probs = run_inference_generic(model, test_scaled, config_name,
                                                     qual_cols, emo_static_cols, emo_temporal_cols,
                                                     cfg["batch_size"], device)
            oof_probs[val_idx] = val_probs
            fold_test_probs.append(test_fold_probs)
            continue

        # ── Scale ──────────────────────────────────────────────────────────────
        scaler = StandardScaler()
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        test_scaled = df_test.copy()
        train_scaled[feat_cols] = scaler.fit_transform(train_df[feat_cols])
        val_scaled[feat_cols] = scaler.transform(val_df[feat_cols])
        test_scaled[feat_cols] = scaler.transform(df_test[feat_cols])

        # ── Model ──────────────────────────────────────────────────────────────
        model = build_model_for_config(config_name, q_dim, s_dim, t_dim,
                                        cfg["embed_dim"], cfg["gate_hidden"], cfg["dropout"])
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                       weight_decay=cfg["weight_decay"])
        n_neg = int((train_df["label_int"] == 0).sum())
        n_pos = int((train_df["label_int"] == 1).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # ── DataLoaders ────────────────────────────────────────────────────────
        train_ds = make_dataset(config_name, train_scaled, qual_cols, emo_static_cols, emo_temporal_cols)
        val_ds = make_dataset(config_name, val_scaled, qual_cols, emo_static_cols, emo_temporal_cols)
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

        # ── Training loop ──────────────────────────────────────────────────────
        best_val_auc = -1.0
        patience = 0

        for epoch in range(cfg["n_epochs"]):
            train_loss = train_epoch_generic(model, train_loader, optimizer, criterion,
                                              device, is_three=is_three)
            val_loss, val_auc, _ = eval_epoch_generic(model, val_loader, criterion,
                                                       device, is_three=is_three)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "scaler": scaler,
                    "val_auc": val_auc,
                }, best_path)
            else:
                patience += 1

            if patience >= cfg["patience"]:
                break

        logger.info(f"  [{config_name}] Fold {k}: best_val_AUC={best_val_auc:.4f}  (ep{epoch})")

        # ── Inference from best ────────────────────────────────────────────────
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()

        val_probs = run_inference_generic(model, val_scaled, config_name,
                                          qual_cols, emo_static_cols, emo_temporal_cols,
                                          cfg["batch_size"], device)
        test_fold_probs = run_inference_generic(model, test_scaled, config_name,
                                                 qual_cols, emo_static_cols, emo_temporal_cols,
                                                 cfg["batch_size"], device)
        oof_probs[val_idx] = val_probs
        fold_test_probs.append(test_fold_probs)
        done_path.touch()

    test_ensemble_probs = np.stack(fold_test_probs, axis=0).mean(axis=0)
    oof_auc = compute_auc(df_trainval["label_int"].values, oof_probs)
    test_auc = compute_auc(df_test["label_int"].values, test_ensemble_probs)

    logger.info(f"[{config_name}] OOF_AUC={oof_auc:.4f}  Test_AUC={test_auc:.4f}")
    return {
        "config": config_name,
        "oof_auc": oof_auc,
        "test_auc": test_auc,
        "oof_probs": oof_probs,
        "test_probs": test_ensemble_probs,
    }


def main():
    set_seeds(42)
    cfg = load_config(str(CONFIG_PATH))
    out_dir = ROOT / cfg["paths"]["output_root"]
    pred_dir = out_dir / "predictions"
    ckpt_dir = out_dir / "checkpoints"
    table_dir = out_dir / "tables"
    stats_dir = out_dir / "stats"
    fig_dir = out_dir / "figures"
    log_dir = out_dir / "logs"
    for d in [table_dir, stats_dir, fig_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15_tm.ablation", str(log_dir / "run.log"))
    logger.info("=== Stage 07: Ablation — Modality Removal ===")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    qual_cols = cfg["quality_features"]
    emo_static_cols = cfg["emotion_static_features"]
    emo_temporal_base = [c for c in cfg["emotion_temporal_features"] if not c.startswith("std_score_")]
    emo_temporal_std = [c for c in cfg["emotion_temporal_features"] if c.startswith("std_score_")]
    emo_temporal_cols = emo_temporal_base + emo_temporal_std

    df_tv = pd.read_parquet(
        require_file(pred_dir / "trainval_feature_matrix.parquet", "Run 01_prepare_features.py")
    )
    df_test = pd.read_parquet(
        require_file(pred_dir / "test_feature_matrix.parquet", "Run 01_prepare_features.py")
    )

    results = []
    for config_name in ABLATION_CONFIGS:
        logger.info(f"\n{'─'*50}\nConfig: {config_name}")
        res = train_config(
            config_name, df_tv, df_test,
            qual_cols, emo_static_cols, emo_temporal_cols,
            cfg, ckpt_dir, log_dir, device, logger,
        )
        results.append(res)

    # ── Reference AUCs ────────────────────────────────────────────────────────
    full_res = next(r for r in results if r["config"] == "full")
    full_oof_auc = full_res["oof_auc"]
    full_test_auc = full_res["test_auc"]

    # ── Permutation tests vs full ──────────────────────────────────────────────
    perm_results = {}
    y_true_test = df_test["label_int"].values
    y_true_oof = df_tv["label_int"].values

    for res in results:
        if res["config"] == "full":
            continue
        perm_test = permutation_test_auc(
            y_true_test, full_res["test_probs"], res["test_probs"],
            n_iter=10000, seed=42,
        )
        perm_results[res["config"]] = perm_test

    with open(stats_dir / "final_exp15_permutation_full_vs_ablation.json", "w") as f:
        json.dump(perm_results, f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_rows = []
    for res in results:
        config_name = res["config"]
        delta_oof = res["oof_auc"] - full_oof_auc if config_name != "full" else 0.0
        delta_test = res["test_auc"] - full_test_auc if config_name != "full" else 0.0
        perm_p = perm_results[config_name]["p_value"] if config_name != "full" else None
        summary_rows.append({
            "config": config_name,
            "trainval_oof_auc": round(res["oof_auc"], 4),
            "test_auc": round(res["test_auc"], 4),
            "delta_oof_vs_full": round(delta_oof, 4) if config_name != "full" else "---",
            "delta_test_vs_full": round(delta_test, 4) if config_name != "full" else "---",
            "permutation_p": round(perm_p, 4) if perm_p is not None else "---",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(table_dir / "final_exp15_ablation_summary.csv", index=False)
    summary_df.to_latex(table_dir / "final_exp15_ablation_summary.tex", index=False,
                        na_rep="—", float_format="%.4f")

    # ── Ablation bars figure ───────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        configs = [r["config"] for r in results]
        test_aucs = [r["test_auc"] for r in results]
        colors = ["#2ca02c" if c == "full" else "#d62728" for c in configs]

        bars = ax.bar(configs, test_aucs, color=colors, width=0.5)
        ax.axhline(y=test_aucs[0], color="green", linestyle="--", alpha=0.5, label="full")
        for bar, auc_val in zip(bars, test_aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{auc_val:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(max(0, min(test_aucs) - 0.05), min(1.0, max(test_aucs) + 0.05))
        ax.set_xlabel("Configuration")
        ax.set_ylabel("AUC")
        ax.tick_params(axis="x", rotation=15)

        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(fig_dir / "final_exp15_ablation_bars.png", dpi=300)
        plt.close(fig)
        logger.info("Ablation bars figure saved")
    except Exception as e:
        logger.warning(f"Ablation figure failed: {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("Ablation Summary")
    print(f"{'='*68}")
    print(summary_df.to_string(index=False))
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
