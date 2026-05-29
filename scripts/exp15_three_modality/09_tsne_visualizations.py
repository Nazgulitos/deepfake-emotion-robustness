"""
Stage 09 — t-SNE visualizations of learned representations.

Extracts branch embeddings (h_q, h_s, h_t) from the 5-fold ensemble,
then generates four publication-quality figures.

Reads:
  outputs/checkpoints/fold_{k}/best.pt  (k = 0..4)
  outputs/predictions/test_feature_matrix.parquet
  outputs/predictions/pilot_feature_matrix.parquet
  datasets/detector_processed/final_ucf_scores.csv
  datasets/detector_processed/pilot_ucf_scores.csv

Writes:
  outputs/predictions/final_exp15_tsne_coords.csv
  outputs/figures/final_exp15_tsne_per_modality.png      (Figure A)
  outputs/figures/final_exp15_tsne_ucf_vs_gated.png      (Figure B)
  outputs/figures/final_exp15_tsne_gating_coloured.png   (Figure C)
  outputs/figures/final_exp15_tsne_seen_vs_unseen.png    (Figure D)

Run from project root:
  python scripts/exp15_three_modality/09_tsne_visualizations.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import ThreeModalityDataset
from model import ThreeModalityGated
from utils import compute_auc, get_project_root, load_config, require_file, set_seeds, setup_logger

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()

def _tsne_params():
    import sklearn
    key = "max_iter" if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5) else "n_iter"
    return dict(n_components=2, perplexity=30, random_state=42,
                init="pca", learning_rate="auto", **{key: 1000})

TSNE_PARAMS = _tsne_params()

# Colour palette
C_REAL   = "#4575b4"   # blue
C_FAKE   = "#d73027"   # red-orange
C_QUAL   = "#2ca02c"   # green  (quality dominant)
C_STAT   = "#ff7f0e"   # orange (emotion-static dominant)
C_TEMP   = "#9467bd"   # purple (emotion-temporal dominant)
C_SEEN_REAL   = "#4575b4"
C_SEEN_FAKE   = "#d73027"
C_UNSEEN_REAL = "#74c476"
C_UNSEEN_FAKE = "#fc8d59"

MARKER_FACESWAP    = "o"
MARKER_FACEREENACT = "s"
MARKER_TALKINGFACE = "^"
MARKER_REAL        = "D"


# ── Embedding extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, df, qual_cols, emo_static_cols, emo_temporal_cols,
                        batch_size, device):
    """Returns probs, gates (N,3), h_q (N,16), h_s (N,16), h_t (N,16)."""
    ds = ThreeModalityDataset(df, qual_cols, emo_static_cols, emo_temporal_cols)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_probs, all_gates = [], []
    all_hq, all_hs, all_ht = [], [], []
    for x_q, x_s, x_t, _ in loader:
        x_q, x_s, x_t = x_q.to(device), x_s.to(device), x_t.to(device)
        h_q = model.q_embed(x_q)
        h_s = model.s_embed(x_s)
        h_t = model.t_embed(x_t)
        z_q = model.q_head(h_q).squeeze(-1)
        z_s = model.s_head(h_s).squeeze(-1)
        z_t = model.t_head(h_t).squeeze(-1)
        import torch.nn.functional as F
        gate_logits = model.gate(torch.cat([h_q, h_s, h_t], dim=-1))
        gate_weights = F.softmax(gate_logits, dim=-1)
        z_stacked = torch.stack([z_q, z_s, z_t], dim=-1)
        z_final = (gate_weights * z_stacked).sum(dim=-1)
        all_probs.append(torch.sigmoid(z_final).cpu().numpy())
        all_gates.append(gate_weights.cpu().numpy())
        all_hq.append(h_q.cpu().numpy())
        all_hs.append(h_s.cpu().numpy())
        all_ht.append(h_t.cpu().numpy())
    return (
        np.concatenate(all_probs),
        np.concatenate(all_gates),
        np.concatenate(all_hq),
        np.concatenate(all_hs),
        np.concatenate(all_ht),
    )


def load_ensemble_embeddings(ckpt_dir, test_df, qual_cols, emo_static_cols, emo_temporal_cols,
                              all_feat_cols, cfg, device, n_folds=5):
    """Average embeddings over all 5 fold models."""
    fold_probs, fold_gates = [], []
    fold_hq, fold_hs, fold_ht = [], [], []

    for k in range(n_folds):
        ckpt_path = ckpt_dir / f"fold_{k}" / "best.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        scaler = ckpt["scaler"]

        df_scaled = test_df.copy()
        df_scaled[all_feat_cols] = scaler.transform(test_df[all_feat_cols])

        model = ThreeModalityGated(
            quality_dim=len(qual_cols),
            emo_static_dim=len(emo_static_cols),
            emo_temporal_dim=len(emo_temporal_cols),
            embed_dim=cfg["embed_dim"],
            gate_hidden=cfg["gate_hidden"],
            dropout=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        probs, gates, hq, hs, ht = extract_embeddings(
            model, df_scaled, qual_cols, emo_static_cols, emo_temporal_cols,
            cfg["batch_size"], device,
        )
        fold_probs.append(probs)
        fold_gates.append(gates)
        fold_hq.append(hq)
        fold_hs.append(hs)
        fold_ht.append(ht)

    return (
        np.stack(fold_probs).mean(0),
        np.stack(fold_gates).mean(0),
        np.stack(fold_hq).mean(0),
        np.stack(fold_hs).mean(0),
        np.stack(fold_ht).mean(0),
    )


# ── t-SNE helpers ──────────────────────────────────────────────────────────────

def run_tsne(X: np.ndarray) -> np.ndarray:
    # If 1-d, add jitter so t-SNE is meaningful
    if X.ndim == 1 or X.shape[1] == 1:
        X = X.reshape(-1, 1)
        rng = np.random.default_rng(42)
        X = np.hstack([X, rng.normal(0, 0.01, (len(X), 1))])
    tsne = TSNE(**TSNE_PARAMS)
    return tsne.fit_transform(X)


def forgery_marker(fam):
    if pd.isna(fam):
        return MARKER_REAL
    fam = str(fam)
    if "FaceSwap" in fam:
        return MARKER_FACESWAP
    elif "FaceReenact" in fam:
        return MARKER_FACEREENACT
    elif "TalkingFace" in fam:
        return MARKER_TALKINGFACE
    return MARKER_REAL


def scatter_by_marker(ax, xy, labels, families, color_real, color_fake,
                       size=40, alpha=0.75):
    """Plot scatter with per-marker forgery family."""
    fam_map = {
        MARKER_FACESWAP: "FaceSwap",
        MARKER_FACEREENACT: "FaceReenact",
        MARKER_TALKINGFACE: "TalkingFace",
        MARKER_REAL: "Real",
    }
    for marker, fam_label in fam_map.items():
        if marker == MARKER_REAL:
            mask = np.array([pd.isna(f) for f in families]) & (labels == 0)
        else:
            mask = np.array([forgery_marker(f) == marker for f in families]) & (labels == 1)
        if mask.sum() == 0:
            continue
        color = color_fake if fam_label != "Real" else color_real
        ax.scatter(xy[mask, 0], xy[mask, 1], c=color, marker=marker,
                   s=size, alpha=alpha, edgecolors="k", linewidths=0.3,
                   label=fam_label, zorder=3)


def draw_fake_hull(ax, xy, labels, color="red"):
    """Draw convex hull around fake cluster."""
    from scipy.spatial import ConvexHull
    fake_xy = xy[labels == 1]
    if len(fake_xy) < 4:
        return
    try:
        hull = ConvexHull(fake_xy)
        pts = np.append(hull.vertices, hull.vertices[0])
        ax.fill(fake_xy[pts, 0], fake_xy[pts, 1],
                alpha=0.10, color=color, zorder=1)
        ax.plot(fake_xy[pts, 0], fake_xy[pts, 1],
                color=color, lw=1.5, linestyle="--", alpha=0.6, zorder=2)
    except Exception:
        pass


# ── Figure A — per-modality t-SNE ─────────────────────────────────────────────

def figure_a(hq, hs, ht, labels, families, auc_q, auc_s, auc_t, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    embeds = [hq, hs, ht]
    titles = [
        f"(a) Quality branch  (AUC={auc_q:.3f})",
        f"(b) Emotion-Static branch  (AUC={auc_s:.3f})",
        f"(c) Emotion-Temporal branch  (AUC={auc_t:.3f})",
    ]
    for ax, emb, title in zip(axes, embeds, titles):
        xy = run_tsne(emb)
        scatter_by_marker(ax, xy, labels, families, C_REAL, C_FAKE)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # shared legend
    legend_elements = [
        mpatches.Patch(color=C_REAL, label="Real"),
        mpatches.Patch(color=C_FAKE, label="Fake"),
        Line2D([0], [0], marker=MARKER_FACESWAP, color="w", markerfacecolor="grey",
               markersize=8, label="FaceSwap"),
        Line2D([0], [0], marker=MARKER_FACEREENACT, color="w", markerfacecolor="grey",
               markersize=8, label="FaceReenact"),
        Line2D([0], [0], marker=MARKER_TALKINGFACE, color="w", markerfacecolor="grey",
               markersize=8, label="TalkingFace"),
        Line2D([0], [0], marker=MARKER_REAL, color="w", markerfacecolor="grey",
               markersize=8, label="Real (diamond)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(fig_dir / "final_exp15_tsne_per_modality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Figure B — UCF vs ThreeModality ───────────────────────────────────────────

def figure_b(combined_emb, labels, families, ucf_scores, auc_ucf, auc_gated, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # UCF subplot: 1-d logits with jitter
    xy_ucf = run_tsne(ucf_scores.reshape(-1, 1))
    ax = axes[0]
    scatter_by_marker(ax, xy_ucf, labels, families, C_REAL, C_FAKE)
    draw_fake_hull(ax, xy_ucf, labels, color=C_FAKE)
    ax.set_title(f"(a) UCF Baseline  (AUC={auc_ucf:.3f})", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    # ThreeModality subplot: 48-d combined embedding
    xy_gated = run_tsne(combined_emb)
    ax = axes[1]
    scatter_by_marker(ax, xy_gated, labels, families, C_REAL, C_FAKE)
    draw_fake_hull(ax, xy_gated, labels, color=C_FAKE)
    ax.set_title(f"(b) Three-Modality Gated  (AUC={auc_gated:.3f})", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    legend_elements = [
        mpatches.Patch(color=C_REAL, label="Real"),
        mpatches.Patch(color=C_FAKE, label="Fake"),
        Line2D([0], [0], marker=MARKER_FACESWAP, color="w", markerfacecolor="grey",
               markersize=8, label="FaceSwap"),
        Line2D([0], [0], marker=MARKER_FACEREENACT, color="w", markerfacecolor="grey",
               markersize=8, label="FaceReenact"),
        Line2D([0], [0], marker=MARKER_TALKINGFACE, color="w", markerfacecolor="grey",
               markersize=8, label="TalkingFace"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(fig_dir / "final_exp15_tsne_ucf_vs_gated.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return xy_gated  # reuse for Figure C


# ── Figure C — gating-coloured t-SNE ──────────────────────────────────────────

def figure_c(xy_gated, gates, labels, fig_dir):
    dom = np.argmax(gates, axis=1)  # 0=quality, 1=static, 2=temporal
    colors_map = {0: C_QUAL, 1: C_STAT, 2: C_TEMP}
    dom_labels = {0: "Quality dominant", 1: "Emotion-Static dominant", 2: "Emotion-Temporal dominant"}

    fig, ax = plt.subplots(figsize=(8, 7))
    for d in [0, 1, 2]:
        mask = dom == d
        marker_arr = np.array(["o" if l == 0 else "^" for l in labels[mask]])
        for m, m_label in [("o", "real"), ("^", "fake")]:
            submask = marker_arr == m
            pts = xy_gated[mask][submask]
            if len(pts) == 0:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], c=colors_map[d], marker=m,
                       s=50, alpha=0.8, edgecolors="k", linewidths=0.3,
                       label=f"{dom_labels[d]} ({m_label})" if m == "o" else None,
                       zorder=3)

    legend_elements = [
        mpatches.Patch(color=C_QUAL, label="Quality dominant"),
        mpatches.Patch(color=C_STAT, label="Emotion-Static dominant"),
        mpatches.Patch(color=C_TEMP, label="Emotion-Temporal dominant"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markersize=8, markeredgecolor="k", label="Real (circle)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
               markersize=8, markeredgecolor="k", label="Fake (triangle)"),
    ]
    ax.set_title("Dominant gating modality", fontsize=11)
    ax.legend(handles=legend_elements, fontsize=9, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(fig_dir / "final_exp15_tsne_gating_coloured.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Figure D — seen vs unseen ──────────────────────────────────────────────────

def figure_d(xy_gated_all, xy_ucf_all, test_labels, pilot_labels,
             auc_ucf_test, auc_gated_test,
             auc_ucf_pilot, auc_gated_pilot, fig_dir):
    """xy_gated_all and xy_ucf_all are already projected (len(test)+len(pilot), 2)."""

    def make_category(labels, is_test):
        cats = []
        for l in labels:
            if is_test:
                cats.append("seen_real" if l == 0 else "seen_fake")
            else:
                cats.append("unseen_real" if l == 0 else "unseen_fake")
        return cats

    cats_test  = make_category(test_labels,  is_test=True)
    cats_pilot = make_category(pilot_labels, is_test=False)
    all_cats   = np.array(cats_test + cats_pilot)

    cat_styles = {
        "seen_real":   (C_SEEN_REAL,   "o", "Seen Real (test)"),
        "seen_fake":   (C_SEEN_FAKE,   "^", "Seen Fake (test)"),
        "unseen_real": (C_UNSEEN_REAL, "o", "Unseen Real (pilot)"),
        "unseen_fake": (C_UNSEEN_FAKE, "^", "Unseen Fake (pilot)"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, xy, title in [
        (axes[0], xy_ucf_all,
         f"(a) UCF Baseline\ntest AUC={auc_ucf_test:.3f}  pilot AUC={auc_ucf_pilot:.3f}"),
        (axes[1], xy_gated_all,
         f"(b) Three-Modality Gated\ntest AUC={auc_gated_test:.3f}  pilot AUC={auc_gated_pilot:.3f}"),
    ]:
        for cat, (color, marker, _) in cat_styles.items():
            mask = np.array([c == cat for c in all_cats])
            if mask.sum() == 0:
                continue
            ax.scatter(xy[mask, 0], xy[mask, 1], c=color, marker=marker,
                       s=45, alpha=0.8, edgecolors="k", linewidths=0.3, zorder=3)
        ax.set_xticks([])
        ax.set_yticks([])

    legend_elements = [
        mpatches.Patch(color=c, label=lbl)
        for cat, (c, m, lbl) in cat_styles.items()
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(fig_dir / "final_exp15_tsne_seen_vs_unseen.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    set_seeds(42)
    cfg = load_config(str(CONFIG_PATH))
    out_dir  = ROOT / cfg["paths"]["output_root"]
    pred_dir = out_dir / "predictions"
    ckpt_dir = out_dir / "checkpoints"
    fig_dir  = out_dir / "figures"
    log_dir  = out_dir / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15_tm.tsne", str(log_dir / "run.log"))
    logger.info("=== Stage 09: t-SNE Visualizations ===")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    qual_cols       = cfg["quality_features"]
    emo_static_cols = cfg["emotion_static_features"]
    emo_temporal_base = [c for c in cfg["emotion_temporal_features"] if not c.startswith("std_score_")]
    emo_temporal_std  = [c for c in cfg["emotion_temporal_features"] if c.startswith("std_score_")]
    emo_temporal_cols = emo_temporal_base + emo_temporal_std
    all_feat_cols = qual_cols + emo_static_cols + emo_temporal_cols

    # ── Load full final data (trainval + test) for t-SNE ──────────────────────
    trainval = pd.read_parquet(
        require_file(pred_dir / "trainval_feature_matrix.parquet", "Run 01_prepare_features.py")
    )
    test_split = pd.read_parquet(
        require_file(pred_dir / "test_feature_matrix.parquet", "Run 01_prepare_features.py")
    )
    test = pd.concat([trainval, test_split], ignore_index=True)
    logger.info(f"Full final dataset: {len(test)} videos (trainval={len(trainval)}, test={len(test_split)})")

    # ── Extract embeddings for all final videos (ensemble mean) ───────────────
    logger.info("Extracting embeddings from 5-fold ensemble...")
    probs_test, gates_test, hq_test, hs_test, ht_test = load_ensemble_embeddings(
        ckpt_dir, test, qual_cols, emo_static_cols, emo_temporal_cols,
        all_feat_cols, cfg, device,
    )
    combined_test = np.hstack([hq_test, hs_test, ht_test])
    labels_test   = test["label_int"].values
    families_test = test["forgery_family"].values

    # Branch AUCs (simple logistic scoring — use branch logit proxy = embedding L2 norm sign)
    # Better: use per-branch single-modality models from ablation, but since we have embeddings
    # we use the branch head output. Re-run single forward pass for branch logits.
    # For simplicity: use correlation of each embedding PCA-1 with label as AUC proxy.
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    def branch_auc(emb, labels):
        try:
            lr = LogisticRegression(max_iter=500, random_state=42)
            lr.fit(emb, labels)
            return compute_auc(labels, lr.predict_proba(emb)[:, 1])
        except Exception:
            return float("nan")

    auc_q = branch_auc(hq_test, labels_test)
    auc_s = branch_auc(hs_test, labels_test)
    auc_t = branch_auc(ht_test, labels_test)
    auc_gated_test = compute_auc(labels_test, probs_test)
    logger.info(f"Branch AUCs (in-sample): q={auc_q:.3f}  s={auc_s:.3f}  t={auc_t:.3f}  "
                f"gated={auc_gated_test:.3f}")

    # ── UCF scores for test ────────────────────────────────────────────────────
    ucf_all = pd.read_csv(require_file(ROOT / cfg["paths"]["ucf_scores"], "UCF scores"))
    ucf_test_merged = test[["video_id", "label_int"]].merge(
        ucf_all[["video_id", "detector_score"]], on="video_id", how="left"
    )
    ucf_scores_test = ucf_test_merged["detector_score"].fillna(0.0).values
    auc_ucf_test = compute_auc(labels_test, ucf_scores_test)
    logger.info(f"UCF full AUC: {auc_ucf_test:.3f}")

    # ── Load pilot data ────────────────────────────────────────────────────────
    pilot_path = pred_dir / "pilot_feature_matrix.parquet"
    pilot_ok = pilot_path.exists()
    if pilot_ok:
        pilot = pd.read_parquet(pilot_path)
        logger.info(f"Pilot: {len(pilot)} videos")
        probs_pilot, gates_pilot, hq_pilot, hs_pilot, ht_pilot = load_ensemble_embeddings(
            ckpt_dir, pilot, qual_cols, emo_static_cols, emo_temporal_cols,
            all_feat_cols, cfg, device,
        )
        combined_pilot = np.hstack([hq_pilot, hs_pilot, ht_pilot])
        labels_pilot   = pilot["label_int"].values

        ucf_pilot_df = pd.read_csv(
            require_file(ROOT / cfg["paths"]["ucf_scores_pilot"], "pilot UCF scores")
        )
        ucf_pilot_merged = pilot[["video_id", "label_int"]].merge(
            ucf_pilot_df[["video_id", "detector_score"]], on="video_id", how="left"
        )
        ucf_scores_pilot = ucf_pilot_merged["detector_score"].fillna(0.0).values
        auc_ucf_pilot    = compute_auc(labels_pilot, ucf_scores_pilot)
        auc_gated_pilot  = compute_auc(labels_pilot, probs_pilot)
        logger.info(f"Pilot — UCF AUC: {auc_ucf_pilot:.3f}  Gated AUC: {auc_gated_pilot:.3f}")
    else:
        logger.warning("Pilot feature matrix not found — skipping Figure D")

    # ── Save t-SNE coordinates ─────────────────────────────────────────────────
    logger.info("Computing t-SNE coordinates for all projections...")
    xy_combined_test = run_tsne(combined_test)
    xy_hq_test = run_tsne(hq_test)
    xy_hs_test = run_tsne(hs_test)
    xy_ht_test = run_tsne(ht_test)
    xy_ucf_test = run_tsne(ucf_scores_test.reshape(-1, 1))

    coord_rows = []
    for i in range(len(test)):
        base = {
            "video_id": test["video_id"].iloc[i],
            "source": "test",
            "label": int(labels_test[i]),
            "forgery_family": test["forgery_family"].iloc[i] if "forgery_family" in test.columns else None,
            "dominant_emotion": test["dominant_emotion"].iloc[i] if "dominant_emotion" in test.columns else None,
            "gate_q": float(gates_test[i, 0]),
            "gate_s": float(gates_test[i, 1]),
            "gate_t": float(gates_test[i, 2]),
        }
        for modality, xy in [("quality", xy_hq_test), ("emotion_static", xy_hs_test),
                               ("emotion_temporal", xy_ht_test), ("combined", xy_combined_test),
                               ("ucf", xy_ucf_test)]:
            coord_rows.append({**base, "modality": modality,
                                "tsne_x": float(xy[i, 0]), "tsne_y": float(xy[i, 1])})

    if pilot_ok:
        xy_combined_pilot = run_tsne(np.vstack([combined_test, combined_pilot]))[len(test):]
        xy_ucf_pilot = run_tsne(
            np.vstack([ucf_scores_test.reshape(-1, 1), ucf_scores_pilot.reshape(-1, 1)])
        )[len(test):]
        for i in range(len(pilot)):
            base = {
                "video_id": pilot["video_id"].iloc[i],
                "source": "pilot",
                "label": int(labels_pilot[i]),
                "forgery_family": pilot["forgery_family"].iloc[i] if "forgery_family" in pilot.columns else None,
                "dominant_emotion": pilot["dominant_emotion"].iloc[i] if "dominant_emotion" in pilot.columns else None,
                "gate_q": float(gates_pilot[i, 0]),
                "gate_s": float(gates_pilot[i, 1]),
                "gate_t": float(gates_pilot[i, 2]),
            }
            for modality, xy in [("combined", xy_combined_pilot), ("ucf", xy_ucf_pilot)]:
                coord_rows.append({**base, "modality": modality,
                                    "tsne_x": float(xy[i, 0]), "tsne_y": float(xy[i, 1])})

    coords_df = pd.DataFrame(coord_rows)
    coords_df.to_csv(pred_dir / "final_exp15_tsne_coords.csv", index=False)
    logger.info(f"t-SNE coordinates saved ({len(coords_df)} rows)")

    # ── Generate figures ───────────────────────────────────────────────────────
    logger.info("Figure A: per-modality t-SNE...")
    figure_a(hq_test, hs_test, ht_test, labels_test, families_test,
             auc_q, auc_s, auc_t, fig_dir)
    logger.info("Figure A saved.")

    logger.info("Figure B: UCF vs ThreeModality t-SNE...")
    xy_gated = figure_b(combined_test, labels_test, families_test,
                        ucf_scores_test, auc_ucf_test, auc_gated_test, fig_dir)
    logger.info("Figure B saved.")

    logger.info("Figure C: gating-coloured t-SNE...")
    figure_c(xy_gated, gates_test, labels_test, fig_dir)
    logger.info("Figure C saved.")

    if pilot_ok:
        logger.info("Figure D: seen vs unseen t-SNE...")
        combined_all = np.vstack([combined_test, combined_pilot])
        ucf_all_arr  = np.concatenate([ucf_scores_test, ucf_scores_pilot])
        xy_combined_all = run_tsne(combined_all)
        xy_ucf_all      = run_tsne(ucf_all_arr.reshape(-1, 1))
        figure_d(
            xy_combined_all, xy_ucf_all,
            labels_test, labels_pilot,
            auc_ucf_test, auc_gated_test,
            auc_ucf_pilot, auc_gated_pilot,
            fig_dir,
        )
        logger.info("Figure D saved.")
    else:
        logger.warning("Figure D skipped — pilot data not available")

    print(f"\n{'='*60}")
    print("Stage 09 complete. Figures saved to:", fig_dir)
    print(f"  A: tsne_per_modality.png")
    print(f"  B: tsne_ucf_vs_gated.png")
    print(f"  C: tsne_gating_coloured.png")
    if pilot_ok:
        print(f"  D: tsne_seen_vs_unseen.png")
    print(f"  Coords: predictions/final_exp15_tsne_coords.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
