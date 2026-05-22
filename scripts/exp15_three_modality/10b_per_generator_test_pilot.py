"""
Stage 10b — Per-generator analysis on TEST + PILOT only (no OOF leakage).

Uses only:
  - test_exp15_predictions.csv   (155 videos, predefined held-out test split)
  - pilot_exp15_predictions.csv  (196 videos, separate pilot holdout)

Both subsets were never seen during training → honest evaluation.

Writes (prefixed testpilot_ to avoid overwriting stage 10 results):
  outputs/tables/testpilot_exp15_per_generator_stats.csv
  outputs/tables/testpilot_exp15_per_generator_top_findings.csv
  outputs/figures/testpilot_exp15_per_generator_heatmap.png
  outputs/figures/testpilot_exp15_per_generator_auc_comparison.png
  outputs/figures/testpilot_exp15_per_generator_modality_dominance.png
  outputs/logs/per_generator_testpilot.log

Run from project root:
  python scripts/exp15_three_modality/10b_per_generator_test_pilot.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils import get_project_root, load_config, setup_logger

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()

GENERATOR_CANONICAL = {
    "tps-mm": "TPSMM", "tpsmm": "TPSMM", "tps_mm": "TPSMM",
    "real-3d-portrait": "Real3DPortrait", "real3d-portrait": "Real3DPortrait",
    "ip-lap": "IP_LAP", "ip_lap": "IP_LAP", "iplap": "IP_LAP",
    "celeb-df-v2": "CelebDFv2", "celebdf": "CelebDFv2", "celeb_df_v2": "CelebDFv2",
}

RAW_GATE_COLS   = ["gate_q", "gate_s", "gate_t"]
STATS_GATE_COLS = ["mean_gate_q", "mean_gate_s", "mean_gate_t"]
GATE_LABELS     = ["Quality", "Emo-Static", "Emo-Temporal"]
GATE_COLORS     = ["#2ca02c", "#ff7f0e", "#9467bd"]


def normalize_generator(name):
    if pd.isna(name):
        return "Unknown"
    key = str(name).lower().replace(" ", "").replace("-", "").replace("_", "")
    for k, v in GENERATOR_CANONICAL.items():
        if k.replace("-", "").replace("_", "") == key:
            return v
    return str(name).strip()


def _normalize_df(df, source_tag):
    if "label_int" not in df.columns and "label" in df.columns:
        df["label_int"] = df["label"].astype(int)
    if "label" not in df.columns and "label_int" in df.columns:
        df["label"] = df["label_int"]
    df["label_int"] = df["label_int"].astype(int)
    df["source"] = source_tag
    return df


def build_generator_map(face_manifest_path, pilot_manifest_path, logger):
    rows = []
    for path, tag in [(face_manifest_path, "final"), (pilot_manifest_path, "pilot")]:
        if not path.exists():
            logger.warning(f"Manifest not found: {path}")
            continue
        face = pd.read_csv(path)
        if "manipulation_type" not in face.columns:
            logger.warning(f"No manipulation_type in {path.name}")
            continue
        sub = (
            face[face["label"] == "fake"][["video_id", "manipulation_type", "manipulation_family"]]
            .drop_duplicates("video_id")
            .copy()
        )
        sub["generator"] = sub["manipulation_type"].apply(normalize_generator)
        sub = sub.rename(columns={"manipulation_family": "forgery_family_src"})
        sub["manifest_source"] = tag
        rows.append(sub[["video_id", "generator", "forgery_family_src"]])

    if not rows:
        raise RuntimeError("Could not build generator map from any manifest")
    return pd.concat(rows, ignore_index=True).drop_duplicates("video_id")


def compute_per_generator_auc(df_fakes, df_reals, score_col):
    if len(df_fakes) < 3 or len(df_reals) < 3:
        return float("nan")
    subset = pd.concat([df_fakes, df_reals], ignore_index=True)
    try:
        return float(roc_auc_score(subset["label_int"], subset[score_col]))
    except Exception:
        return float("nan")


def make_figures(stats, fig_dir, prefix, logger):
    # ── Figure 1: Heatmap ─────────────────────────────────────────────────────
    try:
        import seaborn as sns
        pivot = stats.set_index("generator")[STATS_GATE_COLS].copy()
        pivot.columns = GATE_LABELS

        fig, (ax_heat, ax_bar) = plt.subplots(
            1, 2, figsize=(10, max(7, len(stats) * 0.45)),
            gridspec_kw={"width_ratios": [5, 1]},
        )
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis",
                    vmin=0, vmax=1, cbar_kws={"label": "Mean Gate Weight"},
                    linewidths=0.5, ax=ax_heat)
        ax_heat.set_ylabel("")
        ax_heat.set_title("Modality dominance per generator (test+pilot)", pad=12)

        families_order = stats["forgery_family"].values
        breaks = [i for i in range(1, len(families_order))
                  if families_order[i] != families_order[i - 1]]
        for b in breaks:
            ax_heat.axhline(y=b, color="white", linewidth=2.5)

        ax_bar.barh(range(len(stats)), stats["n_fake"].values, color="#4C72B0", alpha=0.7)
        ax_bar.set_yticks(range(len(stats)))
        ax_bar.set_yticklabels([])
        ax_bar.set_xlabel("n fake")
        ax_bar.set_title("n", pad=12)
        ax_bar.invert_yaxis()
        for b in breaks:
            ax_bar.axhline(y=b, color="grey", linewidth=1.5)

        fig.tight_layout()
        out = fig_dir / f"{prefix}_per_generator_heatmap.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure 1 saved: {out.name}")
    except Exception as e:
        logger.warning(f"Figure 1 failed: {e}")

    # ── Figure 2: AUC comparison bars ─────────────────────────────────────────
    try:
        stats_plot = stats.dropna(subset=["auc_threemodality", "auc_ucf"])
        n = len(stats_plot)
        x = np.arange(n)
        width = 0.38

        fig, ax = plt.subplots(figsize=(max(12, n * 0.75), 6))
        ax.bar(x - width / 2, stats_plot["auc_ucf"], width,
               label="UCF baseline", color="#9E9E9E", alpha=0.85)
        ax.bar(x + width / 2, stats_plot["auc_threemodality"], width,
               label="ThreeModality", color="#4C72B0", alpha=0.85)

        for i, (_, row) in enumerate(stats_plot.iterrows()):
            d = row["delta_auc"]
            if not np.isnan(d):
                c = "#2ca02c" if d >= 0 else "#d62728"
                ax.text(x[i], max(row["auc_threemodality"], row["auc_ucf"]) + 0.015,
                        f"{d:+.2f}", ha="center", fontsize=7, color=c, fontweight="bold")

        ax.axhline(0.5, color="red", linestyle="--", alpha=0.4, lw=1, label="Chance (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r['generator']}\n({r['forgery_family'][:4]})"
             for _, r in stats_plot.iterrows()],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("AUC")
        ax.set_title("Per-generator AUC: ThreeModality vs UCF  [test+pilot only]")
        ax.legend(fontsize=9)

        families_order2 = stats_plot["forgery_family"].values
        for i in range(1, len(families_order2)):
            if families_order2[i] != families_order2[i - 1]:
                ax.axvline(x=i - 0.5, color="black", linewidth=1.5, alpha=0.4)

        fig.tight_layout()
        out = fig_dir / f"{prefix}_per_generator_auc_comparison.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure 2 saved: {out.name}")
    except Exception as e:
        logger.warning(f"Figure 2 failed: {e}")

    # ── Figure 3: Stacked horizontal bars ─────────────────────────────────────
    try:
        sp = stats.dropna(subset=STATS_GATE_COLS)
        y = np.arange(len(sp))

        fig, ax = plt.subplots(figsize=(9, max(6, len(sp) * 0.42)))
        bottom = np.zeros(len(sp))
        for col, lbl, color in zip(STATS_GATE_COLS, GATE_LABELS, GATE_COLORS):
            vals = sp[col].values
            ax.barh(y, vals, left=bottom, label=lbl, color=color, height=0.65)
            for j, (v, b) in enumerate(zip(vals, bottom)):
                if v > 0.08:
                    ax.text(b + v / 2, y[j], f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
            bottom += vals

        fam_list = sp["forgery_family"].values
        for i in range(1, len(fam_list)):
            if fam_list[i] != fam_list[i - 1]:
                ax.axhline(y=i - 0.5, color="black", linewidth=1.5, alpha=0.5)

        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{r['generator']}  ({r['forgery_family'][:4]})" for _, r in sp.iterrows()],
            fontsize=8,
        )
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Mean gate weight")
        ax.set_title("Modality dominance per generator [test+pilot]")
        ax.legend(loc="lower right", fontsize=9)
        ax.invert_yaxis()

        fig.tight_layout()
        out = fig_dir / f"{prefix}_per_generator_modality_dominance.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure 3 saved: {out.name}")
    except Exception as e:
        logger.warning(f"Figure 3 failed: {e}")


def main():
    cfg       = load_config(str(CONFIG_PATH))
    out_dir   = ROOT / cfg["paths"]["output_root"]
    pred_dir  = out_dir / "predictions"
    table_dir = out_dir / "tables"
    fig_dir   = out_dir / "figures"
    log_dir   = out_dir / "logs"
    for d in [table_dir, fig_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    PREFIX = "testpilot_exp15"
    logger = setup_logger("exp15_tm.per_gen_tp", str(log_dir / "per_generator_testpilot.log"))
    logger.info("=== Stage 10b: Per-Generator Analysis (test + pilot only) ===")

    # ── Load test predictions ──────────────────────────────────────────────────
    test  = _normalize_df(pd.read_csv(pred_dir / "test_exp15_predictions.csv"),  "test")
    pilot = _normalize_df(pd.read_csv(pred_dir / "pilot_exp15_predictions.csv"), "pilot")
    combined = pd.concat([test, pilot], ignore_index=True)
    logger.info(f"Loaded: test={len(test)}  pilot={len(pilot)}  total={len(combined)}")
    logger.info(f"  fake={int((combined['label_int']==1).sum())}  "
                f"real={int((combined['label_int']==0).sum())}")

    # ── Build generator map (final + pilot manifests) ─────────────────────────
    pilot_manifest = ROOT / cfg["paths"].get(
        "pilot_face_manifest",
        "datasets/metadata/pilot_face_manifest.csv"
    )
    gen_map = build_generator_map(
        ROOT / cfg["paths"]["face_manifest"],
        pilot_manifest,
        logger,
    )
    logger.info(f"Generator map: {len(gen_map)} fake videos, "
                f"{gen_map['generator'].nunique()} unique generators")

    combined = combined.merge(
        gen_map[["video_id", "generator", "forgery_family_src"]],
        on="video_id", how="left",
    )
    fake_mask = combined["label_int"] == 1
    unknown = (combined.loc[fake_mask, "generator"].isna() |
               (combined.loc[fake_mask, "generator"] == "Unknown")).sum()
    if unknown:
        logger.warning(f"{unknown} fake videos with unknown generator")

    # ── UCF scores ────────────────────────────────────────────────────────────
    ucf_final = pd.read_csv(ROOT / cfg["paths"]["ucf_scores"])[["video_id", "detector_score"]]
    ucf_pilot_path = ROOT / "datasets/detector_processed/pilot_ucf_scores.csv"
    if ucf_pilot_path.exists():
        ucf_pilot = pd.read_csv(ucf_pilot_path)[["video_id", "detector_score"]]
        ucf_all = pd.concat([ucf_final, ucf_pilot], ignore_index=True).drop_duplicates("video_id")
    else:
        logger.warning("pilot_ucf_scores.csv not found, using final UCF only")
        ucf_all = ucf_final
    combined = combined.merge(ucf_all, on="video_id", how="left")
    combined["ucf_score"] = combined["detector_score"].fillna(0.0)

    # ── Reals pool ────────────────────────────────────────────────────────────
    all_reals = combined[combined["label_int"] == 0].copy()
    logger.info(f"Real videos pool: {len(all_reals)}")

    # ── Per-generator stats ────────────────────────────────────────────────────
    rows, excluded = [], []
    for gen, grp in combined[combined["label_int"] == 1].groupby("generator"):
        if pd.isna(gen) or gen == "Unknown":
            continue
        n_fake = len(grp)
        if n_fake < 3:
            excluded.append((gen, n_fake))
            logger.info(f"  Excluded {gen}: n_fake={n_fake} < 3")
            continue

        auc_tm  = compute_per_generator_auc(grp, all_reals, "prediction")
        auc_ucf = compute_per_generator_auc(grp, all_reals, "ucf_score")
        delta   = (auc_tm - auc_ucf) if not (np.isnan(auc_tm) or np.isnan(auc_ucf)) else float("nan")

        gate_data = grp[RAW_GATE_COLS].dropna()
        mean_gq = float(gate_data["gate_q"].mean()) if len(gate_data) else float("nan")
        mean_gs = float(gate_data["gate_s"].mean()) if len(gate_data) else float("nan")
        mean_gt = float(gate_data["gate_t"].mean()) if len(gate_data) else float("nan")

        if not any(np.isnan(v) for v in [mean_gq, mean_gs, mean_gt]):
            dominant = ["quality", "emotion_static", "emotion_temporal"][
                int(np.argmax([mean_gq, mean_gs, mean_gt]))]
        else:
            dominant = "unknown"

        family = (grp["forgery_family_src"].mode().iloc[0]
                  if "forgery_family_src" in grp.columns and grp["forgery_family_src"].notna().any()
                  else grp["forgery_family"].mode().iloc[0]
                  if "forgery_family" in grp.columns else "Unknown")

        rows.append({
            "generator":         gen,
            "forgery_family":    family,
            "n_fake":            n_fake,
            "n_real_paired":     len(all_reals),
            "auc_threemodality": round(auc_tm,  4) if not np.isnan(auc_tm)  else None,
            "auc_ucf":           round(auc_ucf, 4) if not np.isnan(auc_ucf) else None,
            "delta_auc":         round(delta,   4) if not np.isnan(delta)   else None,
            "mean_gate_q":       round(mean_gq, 4),
            "mean_gate_s":       round(mean_gs, 4),
            "mean_gate_t":       round(mean_gt, 4),
            "dominant_modality": dominant,
            "mean_pred_score":   round(float(grp["prediction"].mean()), 4),
        })

    stats = pd.DataFrame(rows).sort_values(["forgery_family", "delta_auc"],
                                            ascending=[True, False])
    stats.to_csv(table_dir / f"{PREFIX}_per_generator_stats.csv", index=False)
    logger.info(f"Stats: {len(stats)} generators, {len(excluded)} excluded (n<3)")

    # ── Top findings ──────────────────────────────────────────────────────────
    findings = []
    sv = stats.dropna(subset=["delta_auc", "auc_threemodality", "auc_ucf"])
    if len(sv):
        for key, col, asc, tmpl in [
            ("biggest_improvement_over_ucf", "delta_auc",         False, "delta_auc={:.3f}"),
            ("hardest_for_ucf",              "auc_ucf",           True,  "ucf_auc={:.3f}"),
            ("easiest_for_threemodality",    "auc_threemodality", False, "auc={:.3f}"),
            ("hardest_for_threemodality",    "auc_threemodality", True,  "auc={:.3f}"),
        ]:
            r = sv.loc[sv[col].idxmin() if asc else sv[col].idxmax()]
            findings.append({"finding": key, "generator": r["generator"],
                             "family": r["forgery_family"],
                             "detail": tmpl.format(r[col])})
    sg = stats.dropna(subset=STATS_GATE_COLS)
    if len(sg):
        for key, col, tmpl in [
            ("most_temporal_dominant", "mean_gate_t", "gate_t={:.3f}"),
            ("most_static_dominant",   "mean_gate_s", "gate_s={:.3f}"),
            ("most_quality_dominant",  "mean_gate_q", "gate_q={:.3f}"),
        ]:
            r = sg.loc[sg[col].idxmax()]
            findings.append({"finding": key, "generator": r["generator"],
                             "family": r["forgery_family"],
                             "detail": tmpl.format(r[col])})

    findings_df = pd.DataFrame(findings)
    findings_df.to_csv(table_dir / f"{PREFIX}_per_generator_top_findings.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    make_figures(stats, fig_dir, PREFIX, logger)

    # ── Console summary ───────────────────────────────────────────────────────
    sv2 = stats.dropna(subset=["auc_threemodality", "auc_ucf", "delta_auc"])
    print(f"\n{'='*64}")
    print("Per-Generator Analysis [TEST + PILOT only] — Summary")
    print(f"{'='*64}")
    print(f"Data source : test ({len(test)} videos) + pilot ({len(pilot)} videos)")
    print(f"Total generators analyzed : {len(stats)}")
    if len(sv2):
        print(f"Avg ThreeModality AUC     : {sv2['auc_threemodality'].mean():.3f}")
        print(f"Avg UCF AUC               : {sv2['auc_ucf'].mean():.3f}")
        print(f"Avg ΔAUC                  : {sv2['delta_auc'].mean():+.3f}")
        print(f"\nTop 3 biggest improvements over UCF:")
        for _, row in sv2.nlargest(3, "delta_auc").iterrows():
            print(f"  {row['generator']:<22} | family={row['forgery_family']:<12} | ΔAUC={row['delta_auc']:+.3f}")
        print(f"\nModality dominance by family:")
        for fam, grp in stats.groupby("forgery_family"):
            dom = ["quality", "emotion_static", "emotion_temporal"][
                int(np.argmax([grp["mean_gate_q"].mean(),
                               grp["mean_gate_s"].mean(),
                               grp["mean_gate_t"].mean()]))]
            print(f"  {fam:<14} → {dom} dominant  "
                  f"(q={grp['mean_gate_q'].mean():.2f}  "
                  f"s={grp['mean_gate_s'].mean():.2f}  "
                  f"t={grp['mean_gate_t'].mean():.2f})")
    print(f"\nOutputs → {table_dir}  and  {fig_dir}")
    print(f"{'='*64}")
    print(f"\nPer-generator stats table:")
    print(stats.to_string(index=False))
    print(f"\nTop findings:")
    print(findings_df.to_string(index=False))


if __name__ == "__main__":
    main()
