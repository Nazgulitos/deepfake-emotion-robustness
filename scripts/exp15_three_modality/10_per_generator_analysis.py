"""
Stage 10 — Per-generator analysis.

Answers the supervisor's question: "show how it works on different deepfake methods".

Generator method is taken directly from `manipulation_type` column in face manifest
(e.g. SimSwap, FSRT, AniTalker, SadTalker, ...).

Reads:
  outputs/predictions/trainval_oof_predictions.csv
  outputs/predictions/test_exp15_predictions.csv
  datasets/metadata/final_face_manifest.csv
  datasets/detector_processed/final_ucf_scores.csv

Writes:
  outputs/tables/final_exp15_per_generator_stats.csv
  outputs/tables/final_exp15_per_generator_top_findings.csv
  outputs/figures/final_exp15_per_generator_heatmap.png
  outputs/figures/final_exp15_per_generator_auc_comparison.png
  outputs/figures/final_exp15_per_generator_modality_dominance.png
  outputs/logs/per_generator_analysis.log

Run from project root:
  python scripts/exp15_three_modality/10_per_generator_analysis.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils import compute_auc, get_project_root, load_config, setup_logger

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()

# Canonical generator names (normalize variants)
GENERATOR_CANONICAL = {
    "tps-mm": "TPSMM", "tpsmm": "TPSMM", "tps_mm": "TPSMM",
    "real-3d-portrait": "Real3DPortrait", "real3d-portrait": "Real3DPortrait",
    "ip-lap": "IP_LAP", "ip_lap": "IP_LAP", "iplap": "IP_LAP",
    "celeb-df-v2": "CelebDFv2", "celebdf": "CelebDFv2", "celeb_df_v2": "CelebDFv2",
}

RAW_GATE_COLS  = ["gate_q", "gate_s", "gate_t"]        # columns in OOF/test predictions
STATS_GATE_COLS = ["mean_gate_q", "mean_gate_s", "mean_gate_t"]  # columns in per-generator stats
GATE_LABELS = ["Quality", "Emo-Static", "Emo-Temporal"]
GATE_COLORS = ["#2ca02c", "#ff7f0e", "#9467bd"]


def normalize_generator(name: str) -> str:
    if pd.isna(name):
        return "Unknown"
    key = str(name).lower().replace(" ", "").replace("-", "").replace("_", "")
    # check canonical map with normalized key
    for k, v in GENERATOR_CANONICAL.items():
        if k.replace("-", "").replace("_", "") == key:
            return v
    return str(name).strip()


def load_predictions(pred_dir: Path) -> pd.DataFrame:
    oof  = pd.read_csv(pred_dir / "trainval_oof_predictions.csv")
    test = pd.read_csv(pred_dir / "test_exp15_predictions.csv")

    # Normalize label column names
    for df in [oof, test]:
        if "label_int" not in df.columns and "label" in df.columns:
            df["label_int"] = df["label"]
        if "label" not in df.columns and "label_int" in df.columns:
            df["label"] = df["label_int"]

    # test predictions use "label" (0/1 from label_int)
    oof["split_source"]  = "oof"
    test["split_source"] = "test"
    combined = pd.concat([oof, test], ignore_index=True)
    return combined


def build_generator_map(face_manifest_path: Path, logger) -> pd.DataFrame:
    """Return DataFrame[video_id, generator, forgery_family] for all fake videos."""
    face = pd.read_csv(face_manifest_path)

    if "manipulation_type" in face.columns:
        logger.info("Generator column found: manipulation_type")
        gen_map = (
            face[face["label"] == "fake"][["video_id", "manipulation_type", "manipulation_family"]]
            .drop_duplicates("video_id")
            .rename(columns={"manipulation_type": "generator",
                              "manipulation_family": "forgery_family_src"})
        )
        gen_map["generator"] = gen_map["generator"].apply(normalize_generator)
        return gen_map
    else:
        # Fallback: parse from video_id
        logger.warning("manipulation_type column not found — parsing generator from video_id")
        KNOWN = [
            "SimSwap", "BlendFace", "GHOST", "InSwapper", "HifiFace",
            "MobileFaceSwap", "UniFace", "CelebDFv2",
            "DaGAN", "FSRT", "HyperReenact", "LIA", "LivePortrait",
            "MCNET", "TPSMM", "Real3DPortrait",
            "AniTalker", "EDTalk", "EchoMimic", "FLOAT", "IP_LAP", "SadTalker",
        ]
        def _parse(vid):
            for g in KNOWN:
                if g.lower() in str(vid).lower():
                    return g
            return "Unknown"

        fake_rows = face[face["label"] == "fake"][["video_id", "manipulation_family"]].drop_duplicates("video_id")
        fake_rows = fake_rows.rename(columns={"manipulation_family": "forgery_family_src"})
        fake_rows["generator"] = fake_rows["video_id"].apply(_parse)
        return fake_rows


def compute_per_generator_auc(df_fakes, df_reals, score_col="prediction"):
    if len(df_fakes) < 3 or len(df_reals) < 3:
        return float("nan")
    subset = pd.concat([df_fakes, df_reals], ignore_index=True)
    try:
        return float(roc_auc_score(subset["label_int"], subset[score_col]))
    except Exception:
        return float("nan")


def main():
    cfg     = load_config(str(CONFIG_PATH))
    out_dir = ROOT / cfg["paths"]["output_root"]
    pred_dir  = out_dir / "predictions"
    table_dir = out_dir / "tables"
    fig_dir   = out_dir / "figures"
    log_dir   = out_dir / "logs"
    for d in [table_dir, fig_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15_tm.per_generator", str(log_dir / "per_generator_analysis.log"))
    logger.info("=== Stage 10: Per-Generator Analysis ===")

    # ── Load predictions ───────────────────────────────────────────────────────
    combined = load_predictions(pred_dir)
    logger.info(f"Loaded predictions: {len(combined)} rows  "
                f"(fake={int((combined['label_int']==1).sum())}  "
                f"real={int((combined['label_int']==0).sum())})")

    # ── Build generator map ────────────────────────────────────────────────────
    gen_map = build_generator_map(ROOT / cfg["paths"]["face_manifest"], logger)
    logger.info(f"Generator map: {len(gen_map)} fake videos, "
                f"{gen_map['generator'].nunique()} unique generators")
    logger.info(f"Generators: {sorted(gen_map['generator'].unique())}")

    # Merge generator into combined predictions
    combined = combined.merge(
        gen_map[["video_id", "generator", "forgery_family_src"]],
        on="video_id", how="left",
    )
    # For real videos generator = NaN — that's correct
    fake_mask = combined["label_int"] == 1
    unknown = (combined.loc[fake_mask, "generator"].isna() |
               (combined.loc[fake_mask, "generator"] == "Unknown")).sum()
    if unknown > 0:
        logger.warning(f"{unknown} fake videos have unknown generator")

    # ── UCF scores ────────────────────────────────────────────────────────────
    ucf_df = pd.read_csv(ROOT / cfg["paths"]["ucf_scores"])[["video_id", "detector_score"]]
    combined = combined.merge(ucf_df, on="video_id", how="left")
    combined["ucf_score"] = combined["detector_score"].fillna(0.0)

    # ── Reals pool ────────────────────────────────────────────────────────────
    all_reals = combined[combined["label_int"] == 0].copy()
    logger.info(f"Real videos pool: {len(all_reals)}")

    # ── Per-generator stats ────────────────────────────────────────────────────
    rows = []
    excluded = []

    for gen, grp in combined[combined["label_int"] == 1].groupby("generator"):
        if pd.isna(gen) or gen == "Unknown":
            continue
        n_fake = len(grp)
        if n_fake < 5:
            excluded.append((gen, n_fake))
            logger.info(f"  Excluded {gen}: n_fake={n_fake} < 5")
            continue

        # AUC ThreeModality
        auc_tm  = compute_per_generator_auc(grp, all_reals, "prediction")
        # AUC UCF
        auc_ucf = compute_per_generator_auc(grp, all_reals, "ucf_score")
        delta   = (auc_tm - auc_ucf) if not (np.isnan(auc_tm) or np.isnan(auc_ucf)) else float("nan")

        # Gate weights (only rows that have gate cols)
        gate_data = grp[RAW_GATE_COLS].dropna()
        mean_gq = float(gate_data["gate_q"].mean()) if len(gate_data) else float("nan")
        mean_gs = float(gate_data["gate_s"].mean()) if len(gate_data) else float("nan")
        mean_gt = float(gate_data["gate_t"].mean()) if len(gate_data) else float("nan")

        # Dominant modality
        if not any(np.isnan(v) for v in [mean_gq, mean_gs, mean_gt]):
            dom_idx = int(np.argmax([mean_gq, mean_gs, mean_gt]))
            dominant = ["quality", "emotion_static", "emotion_temporal"][dom_idx]
        else:
            dominant = "unknown"

        family = grp["forgery_family_src"].mode().iloc[0] if "forgery_family_src" in grp.columns else \
                 grp["forgery_family"].mode().iloc[0] if "forgery_family" in grp.columns else "Unknown"

        rows.append({
            "generator":          gen,
            "forgery_family":     family,
            "n_fake":             n_fake,
            "n_real_paired":      len(all_reals),
            "auc_threemodality":  round(auc_tm, 4) if not np.isnan(auc_tm) else None,
            "auc_ucf":            round(auc_ucf, 4) if not np.isnan(auc_ucf) else None,
            "delta_auc":          round(delta, 4) if not np.isnan(delta) else None,
            "mean_gate_q":        round(mean_gq, 4),
            "mean_gate_s":        round(mean_gs, 4),
            "mean_gate_t":        round(mean_gt, 4),
            "dominant_modality":  dominant,
            "mean_pred_score":    round(float(grp["prediction"].mean()), 4),
        })

    stats = pd.DataFrame(rows)
    stats = stats.sort_values(["forgery_family", "delta_auc"], ascending=[True, False])
    stats.to_csv(table_dir / "final_exp15_per_generator_stats.csv", index=False)
    logger.info(f"Per-generator stats: {len(stats)} generators included, "
                f"{len(excluded)} excluded (n<5)")
    if excluded:
        logger.info(f"Excluded: {excluded}")

    # ── Top findings table ─────────────────────────────────────────────────────
    findings = []
    stats_valid = stats.dropna(subset=["delta_auc", "auc_threemodality", "auc_ucf"])

    if len(stats_valid):
        r = stats_valid.loc[stats_valid["delta_auc"].idxmax()]
        findings.append({"finding": "biggest_improvement_over_ucf",
                         "generator": r["generator"], "family": r["forgery_family"],
                         "detail": f"delta_auc={r['delta_auc']:+.3f}"})

        r = stats_valid.loc[stats_valid["auc_ucf"].idxmin()]
        findings.append({"finding": "hardest_for_ucf",
                         "generator": r["generator"], "family": r["forgery_family"],
                         "detail": f"ucf_auc={r['auc_ucf']:.3f}"})

        r = stats_valid.loc[stats_valid["auc_threemodality"].idxmax()]
        findings.append({"finding": "easiest_for_threemodality",
                         "generator": r["generator"], "family": r["forgery_family"],
                         "detail": f"auc={r['auc_threemodality']:.3f}"})

        r = stats_valid.loc[stats_valid["auc_threemodality"].idxmin()]
        findings.append({"finding": "hardest_for_threemodality",
                         "generator": r["generator"], "family": r["forgery_family"],
                         "detail": f"auc={r['auc_threemodality']:.3f}"})

    stats_gates = stats.dropna(subset=["mean_gate_q", "mean_gate_s", "mean_gate_t"])
    if len(stats_gates):
        r = stats_gates.loc[stats_gates["mean_gate_t"].idxmax()]
        findings.append({"finding": "most_temporal_dominant",
                         "generator": r["generator"], "family": r["forgery_family"],
                         "detail": f"gate_t={r['mean_gate_t']:.3f}"})

        r = stats_gates.loc[stats_gates["mean_gate_s"].idxmax()]
        findings.append({"finding": "most_static_dominant",
                         "generator": r["generator"], "family": r["forgery_family"],
                         "detail": f"gate_s={r['mean_gate_s']:.3f}"})

        r = stats_gates.loc[stats_gates["mean_gate_q"].idxmax()]
        findings.append({"finding": "most_quality_dominant",
                         "generator": r["generator"], "family": r["forgery_family"],
                         "detail": f"gate_q={r['mean_gate_q']:.3f}"})

    findings_df = pd.DataFrame(findings)
    findings_df.to_csv(table_dir / "final_exp15_per_generator_top_findings.csv", index=False)

    # ── Figure 1: Heatmap ──────────────────────────────────────────────────────
    try:
        import seaborn as sns

        pivot = stats.set_index("generator")[["mean_gate_q", "mean_gate_s", "mean_gate_t"]]
        pivot.columns = ["Quality", "Emo-Static", "Emo-Temporal"]

        fig, (ax_heat, ax_bar) = plt.subplots(
            1, 2, figsize=(10, max(7, len(stats) * 0.45)),
            gridspec_kw={"width_ratios": [5, 1]},
        )

        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap="viridis",
            vmin=0, vmax=1, cbar_kws={"label": "Mean Gate Weight"},
            linewidths=0.5, ax=ax_heat,
        )
        ax_heat.set_ylabel("")

        # Family separators
        families_order = stats["forgery_family"].values
        breaks = []
        for i in range(1, len(families_order)):
            if families_order[i] != families_order[i - 1]:
                breaks.append(i)
        for b in breaks:
            ax_heat.axhline(y=b, color="white", linewidth=2.5)

        # n_fake bar
        ax_bar.barh(range(len(stats)), stats["n_fake"].values,
                    color="#4C72B0", alpha=0.7)
        ax_bar.set_yticks(range(len(stats)))
        ax_bar.set_yticklabels([])
        ax_bar.set_xlabel("n fake")
        ax_bar.invert_yaxis()
        for b in breaks:
            ax_bar.axhline(y=b, color="grey", linewidth=1.5)

        fig.tight_layout()
        fig.savefig(fig_dir / "final_exp15_per_generator_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure 1 saved: per_generator_heatmap.png")
    except Exception as e:
        logger.warning(f"Figure 1 failed: {e}")

    # ── Figure 2: AUC comparison bars ─────────────────────────────────────────
    try:
        stats_plot = stats.dropna(subset=["auc_threemodality", "auc_ucf"])
        n = len(stats_plot)
        x = np.arange(n)
        width = 0.38

        fig, ax = plt.subplots(figsize=(max(12, n * 0.75), 6))
        bars_ucf = ax.bar(x - width / 2, stats_plot["auc_ucf"], width,
                          label="UCF baseline", color="#9E9E9E", alpha=0.85)
        bars_tm  = ax.bar(x + width / 2, stats_plot["auc_threemodality"], width,
                          label="ThreeModality", color="#4C72B0", alpha=0.85)

        # ΔAUC annotations above each group
        for i, (_, row) in enumerate(stats_plot.iterrows()):
            d = row["delta_auc"]
            if not np.isnan(d):
                color = "#2ca02c" if d >= 0 else "#d62728"
                ax.text(x[i], max(row["auc_threemodality"], row["auc_ucf"]) + 0.015,
                        f"{d:+.2f}", ha="center", fontsize=7, color=color, fontweight="bold")

        ax.axhline(0.5, color="red", linestyle="--", alpha=0.4, lw=1, label="Chance (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r['generator']}\n({r['forgery_family'][:4]})"
             for _, r in stats_plot.iterrows()],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("AUC")
        ax.legend(fontsize=9)

        # Family separators
        families_order2 = stats_plot["forgery_family"].values
        for i in range(1, len(families_order2)):
            if families_order2[i] != families_order2[i - 1]:
                ax.axvline(x=i - 0.5, color="black", linewidth=1.5, alpha=0.4)

        fig.tight_layout()
        fig.savefig(fig_dir / "final_exp15_per_generator_auc_comparison.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure 2 saved: per_generator_auc_comparison.png")
    except Exception as e:
        logger.warning(f"Figure 2 failed: {e}")

    # ── Figure 3: Stacked horizontal bars (modality dominance) ────────────────
    try:
        stats_gates_plot = stats.dropna(subset=["mean_gate_q", "mean_gate_s", "mean_gate_t"])
        labels_y = stats_gates_plot["generator"].tolist()
        y = np.arange(len(labels_y))

        fig, ax = plt.subplots(figsize=(9, max(6, len(labels_y) * 0.42)))
        bottom = np.zeros(len(labels_y))
        for col, label, color in zip(STATS_GATE_COLS, GATE_LABELS, GATE_COLORS):
            vals = stats_gates_plot[col].values
            ax.barh(y, vals, left=bottom, label=label, color=color, height=0.65)
            for j, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0.08:
                    ax.text(b + v / 2, y[j], f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            bottom += vals

        # Family separators
        fam_list = stats_gates_plot["forgery_family"].values
        for i in range(1, len(fam_list)):
            if fam_list[i] != fam_list[i - 1]:
                ax.axhline(y=i - 0.5, color="black", linewidth=1.5, alpha=0.5)

        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{r['generator']}  ({r['forgery_family'][:4]})"
             for _, r in stats_gates_plot.iterrows()],
            fontsize=8,
        )
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Mean gate weight")
        ax.legend(loc="lower right", fontsize=9)
        ax.invert_yaxis()

        fig.tight_layout()
        fig.savefig(fig_dir / "final_exp15_per_generator_modality_dominance.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure 3 saved: per_generator_modality_dominance.png")
    except Exception as e:
        logger.warning(f"Figure 3 failed: {e}")

    # ── Console summary ────────────────────────────────────────────────────────
    stats_valid2 = stats.dropna(subset=["auc_threemodality", "auc_ucf", "delta_auc"])
    print(f"\n{'='*64}")
    print("Per-Generator Analysis — Summary")
    print(f"{'='*64}")
    print(f"Total generators analyzed : {len(stats)}")
    if len(stats_valid2):
        print(f"Avg ThreeModality AUC     : {stats_valid2['auc_threemodality'].mean():.3f}")
        print(f"Avg UCF AUC               : {stats_valid2['auc_ucf'].mean():.3f}")
        print(f"Avg ΔAUC                  : {stats_valid2['delta_auc'].mean():+.3f}")
        print(f"\nTop 3 biggest improvements over UCF:")
        for i, row in stats_valid2.nlargest(3, "delta_auc").iterrows():
            print(f"  {row['generator']:<22} | family={row['forgery_family']:<12} | ΔAUC={row['delta_auc']:+.3f}")
        print(f"\nModality dominance by family:")
        for fam, grp in stats.groupby("forgery_family"):
            dom = ["quality", "emotion_static", "emotion_temporal"][
                int(np.argmax([grp["mean_gate_q"].mean(),
                               grp["mean_gate_s"].mean(),
                               grp["mean_gate_t"].mean()]))
            ]
            print(f"  {fam:<14} → {dom} dominant  "
                  f"(q={grp['mean_gate_q'].mean():.2f}  "
                  f"s={grp['mean_gate_s'].mean():.2f}  "
                  f"t={grp['mean_gate_t'].mean():.2f})")
    print(f"\nAll outputs saved to {table_dir} and {fig_dir}")
    print(f"{'='*64}")

    print(f"\nPer-generator stats table:")
    print(stats.to_string(index=False))

    print(f"\nTop findings:")
    print(findings_df.to_string(index=False))


if __name__ == "__main__":
    main()
