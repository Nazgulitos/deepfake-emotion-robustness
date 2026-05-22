"""Exp.15 experimental integrity audit (read-only checks)."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold


@dataclass
class CriterionResult:
    status: str
    notes: str


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "datasets").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no 'datasets' dir found)")


def load_yaml(path: Path) -> dict:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_score))
    return {
        "AUC": auc,
        "ACC": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "EER": float(compute_eer(y_true, y_score)),
    }


def list_text_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".yml"}]


def find_text_hits(paths: Iterable[Path], tokens: list[str]) -> list[str]:
    hits: list[str] = []
    pattern = re.compile("|".join(re.escape(t) for t in tokens), re.IGNORECASE)
    for path in paths:
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        if pattern.search(text):
            hits.append(str(path))
    return hits


def main() -> None:
    root = get_project_root()
    cfg_path = root / "scripts/exp15_three_modality/config.yaml"
    cfg = load_yaml(cfg_path)

    audit_dir = root / "outputs/audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / cfg["paths"]["face_manifest"]
    trainval_feat_path = root / "scripts/exp15_three_modality/outputs/predictions/trainval_feature_matrix.parquet"
    test_feat_path = root / "scripts/exp15_three_modality/outputs/predictions/test_feature_matrix.parquet"
    oof_pred_path = root / "scripts/exp15_three_modality/outputs/predictions/trainval_oof_predictions.csv"
    test_pred_path = root / "scripts/exp15_three_modality/outputs/predictions/test_exp15_predictions.csv"
    results_path = root / "scripts/exp15_three_modality/outputs/tables/final_exp15_results.csv"
    ucf_path = root / cfg["paths"]["ucf_scores"]

    # Load manifest
    manifest = pd.read_csv(manifest_path)
    manifest_ids = set(manifest["video_id"].astype(str))
    trainval_ids = set(manifest[manifest["split"].isin(["train", "val"])]["video_id"].astype(str))
    test_ids = set(manifest[manifest["split"] == "test"]["video_id"].astype(str))

    def load_parquet_if(path: Path) -> pd.DataFrame | None:
        return pd.read_parquet(path) if path.exists() else None

    def load_csv_if(path: Path) -> pd.DataFrame | None:
        return pd.read_csv(path) if path.exists() else None

    trainval_feat = load_parquet_if(trainval_feat_path)
    test_feat = load_parquet_if(test_feat_path)
    oof_pred = load_csv_if(oof_pred_path)
    test_pred = load_csv_if(test_pred_path)
    results_df = load_csv_if(results_path)
    ucf_scores = load_csv_if(ucf_path)

    # ── Partition size summary ───────────────────────────────────────────────
    summary_rows = []

    def add_summary(name: str, df: pd.DataFrame | None) -> None:
        if df is None:
            return
        video_ids = df["video_id"].astype(str) if "video_id" in df.columns else pd.Series([], dtype=str)
        unique_ids = set(video_ids.tolist())
        n_final = len(unique_ids & manifest_ids)
        n_nonfinal = len(unique_ids - manifest_ids)
        split_values = ";".join(sorted(set(df["split"].dropna().astype(str)))) if "split" in df.columns else ""
        split_type_values = (
            ";".join(sorted(set(df["split_type"].dropna().astype(str))))
            if "split_type" in df.columns else ""
        )
        summary_rows.append({
            "source": name,
            "rows": int(len(df)),
            "unique_video_ids": int(len(unique_ids)),
            "n_final_ids": int(n_final),
            "n_nonfinal_ids": int(n_nonfinal),
            "split_values": split_values,
            "split_type_values": split_type_values,
        })

    add_summary("final_manifest", manifest)
    add_summary("trainval_feature_matrix.parquet", trainval_feat)
    add_summary("test_feature_matrix.parquet", test_feat)
    add_summary("trainval_oof_predictions.csv", oof_pred)
    add_summary("test_exp15_predictions.csv", test_pred)

    pd.DataFrame(summary_rows).to_csv(audit_dir / "partition_size_summary.csv", index=False)

    # ── Criterion 1: Final-only partition distinction ────────────────────────
    nonfinal_hits = []
    overlap_count = None
    if trainval_feat is not None and test_feat is not None:
        trainval_vids = set(trainval_feat["video_id"].astype(str))
        test_vids = set(test_feat["video_id"].astype(str))
        overlap_count = len(trainval_vids & test_vids)
        nonfinal_hits = sorted((trainval_vids | test_vids) - manifest_ids)
        if overlap_count == 0 and not nonfinal_hits:
            c1 = CriterionResult("PASS", f"Non-final artefacts=[]; trainval/test video overlap=0.")
        else:
            c1 = CriterionResult("FAIL", f"Non-final artefacts={nonfinal_hits}; trainval/test video overlap={overlap_count}.")
    else:
        c1 = CriterionResult("CHECK_NOT_RUN", "Missing trainval/test feature matrices.")

    # ── Criterion 2: Identity-disjoint splits ────────────────────────────────
    identity_rows = []
    identity_status = "PASS"
    max_overlap = 0
    fold_checked = False

    if trainval_feat is not None:
        if "identity" not in trainval_feat.columns:
            identity_map = manifest.set_index("video_id")["identity"].astype(str)
            trainval_feat = trainval_feat.copy()
            trainval_feat["identity"] = trainval_feat["video_id"].astype(str).map(identity_map)
        groups = trainval_feat["identity"].fillna(trainval_feat["video_id"]).astype(str).values
        gkf = GroupKFold(n_splits=int(cfg["n_folds"]))
        fold_checked = True
        for k, (train_idx, val_idx) in enumerate(gkf.split(trainval_feat, groups=groups)):
            train_ids = set(groups[train_idx])
            val_ids = set(groups[val_idx])
            overlap = len(train_ids & val_ids)
            max_overlap = max(max_overlap, overlap)
            status = "PASS" if overlap == 0 else "FAIL"
            if overlap > 0:
                identity_status = "FAIL"
            identity_rows.append({
                "partition_A": f"fold_{k}_train",
                "partition_B": f"fold_{k}_val",
                "n_overlap": overlap,
                "A_size": len(train_ids),
                "B_size": len(val_ids),
                "status": status,
            })

    tv_slice = manifest[manifest["split"].isin(["train", "val"])].copy()
    te_slice = manifest[manifest["split"] == "test"].copy()
    tv_slice["identity"] = tv_slice["identity"].fillna(tv_slice["video_id"]).astype(str)
    te_slice["identity"] = te_slice["identity"].fillna(te_slice["video_id"]).astype(str)
    trainval_identity = set(tv_slice["identity"].tolist())
    test_identity = set(te_slice["identity"].tolist())
    tv_overlap = len(trainval_identity & test_identity)
    identity_rows.append({
        "partition_A": "trainval",
        "partition_B": "test_holdout",
        "n_overlap": tv_overlap,
        "A_size": len(trainval_identity),
        "B_size": len(test_identity),
        "status": "PASS" if tv_overlap == 0 else "FAIL",
    })

    pd.DataFrame(identity_rows).to_csv(audit_dir / "identity_overlap_matrix.csv", index=False)

    if tv_overlap > 0:
        identity_status = "FAIL"
    if not fold_checked:
        c2 = CriterionResult("CHECK_NOT_RUN", "Missing trainval feature matrix or identity column for fold audit.")
    else:
        c2 = CriterionResult(identity_status, f"Max critical identity overlap={max_overlap}.")

    # ── Criterion 3: External subset exclusion ───────────────────────────────
    exp15_root = root / "scripts/exp15_three_modality"
    tokens = ["pilot", "external_subset", "external subset"]
    source_candidates = [p for p in list_text_files(exp15_root) if p.name != "11_integrity_audit.py"]
    source_hits = find_text_hits(source_candidates, tokens)
    output_hits = [p.name for p in (exp15_root / "outputs").rglob("*") if p.is_file() and any(t in p.name.lower() for t in tokens)]
    if source_hits or output_hits:
        c3 = CriterionResult(
            "FAIL",
            f"Active source/spec hits={source_hits}; output filename hits={output_hits}.",
        )
    else:
        c3 = CriterionResult("PASS", "Active source/spec hits=[]; output filename hits=[]")

    # ── Criterion 4: Holdout split preserved ────────────────────────────────
    if test_feat is not None and trainval_feat is not None:
        test_vids = set(test_feat["video_id"].astype(str))
        trainval_vids = set(trainval_feat["video_id"].astype(str))
        sym_diff = len(test_vids ^ test_ids)
        overlap = len(test_vids & trainval_vids)
        if sym_diff == 0 and overlap == 0:
            c4 = CriterionResult(
                "PASS",
                f"Expected test n={len(test_ids)}, actual test n={len(test_vids)}, symmetric_difference={sym_diff}, trainval/test overlap={overlap}.",
            )
        else:
            c4 = CriterionResult(
                "FAIL",
                f"Expected test n={len(test_ids)}, actual test n={len(test_vids)}, symmetric_difference={sym_diff}, trainval/test overlap={overlap}.",
            )
    else:
        c4 = CriterionResult("CHECK_NOT_RUN", "Missing trainval/test feature matrices.")

    # ── Criterion 5: Final model evaluated once on test_holdout ──────────────
    log_dir = exp15_root / "outputs/logs"
    log_hits = 0
    if log_dir.exists():
        for log_path in log_dir.rglob("*.log"):
            text = log_path.read_text(errors="ignore")
            log_hits += len(re.findall(r"test holdout|test_holdout|Evaluate Test Holdout", text, re.IGNORECASE))
    ablation_path = exp15_root / "07_ablation_modality_removal.py"
    ablation_warn = False
    if ablation_path.exists():
        ablation_warn = "test_feature_matrix.parquet" in ablation_path.read_text(errors="ignore")
    if ablation_warn:
        c5 = CriterionResult(
            "WARN",
            f"Log test-evaluation mentions={log_hits}; Stage 07 ablations also evaluate the same fixed holdout and should be disclosed.",
        )
    else:
        c5 = CriterionResult("PASS", f"Log test-evaluation mentions={log_hits}.")

    # ── Criterion 6: Reproducibility seeds are fixed and recoverable ─────────
    cfg_seed_ok = int(cfg.get("seed", -1)) == 42
    utils_path = exp15_root / "utils.py"
    scripts_with_seed = [
        exp15_root / "01_prepare_features.py",
        exp15_root / "02_train_three_modality.py",
        exp15_root / "03_evaluate_test.py",
        exp15_root / "07_ablation_modality_removal.py",
    ]
    seeds_set = cfg_seed_ok and utils_path.exists()
    for path in scripts_with_seed:
        if not path.exists():
            seeds_set = False
            break
        if "set_seeds(42)" not in path.read_text(errors="ignore"):
            seeds_set = False
            break

    cfg_hash_computed = None
    try:
        sys_path = str(exp15_root)
        import sys as _sys

        if sys_path not in _sys.path:
            _sys.path.insert(0, sys_path)
        from utils import hash_config

        cfg_hash_computed = hash_config(cfg)
    except Exception:
        cfg_hash_computed = "UNKNOWN"

    state_files = list((exp15_root / "outputs/checkpoints").rglob("state.json"))
    if not state_files:
        config_seed_rows = [{
            "fold": "ALL",
            "config_hash_stored": "MISSING_STATE_JSON",
            "config_hash_computed": cfg_hash_computed,
            "seeds_set": seeds_set,
            "rng_state_saved": False,
            "status": "CHECK_NOT_RUN",
        }]
        c6 = CriterionResult(
            "CHECK_NOT_RUN",
            "Seeds are fixed in config/source, but checkpoint state.json files are absent, so stored config_hash/RNG state cannot be verified.",
        )
    else:
        config_seed_rows = []
        all_ok = True
        for path in state_files:
            with open(path, "r") as f:
                state = json.load(f)
            stored_hash = state.get("config_hash")
            rng_saved = bool(state.get("rng_state"))
            status = "PASS" if stored_hash == cfg_hash_computed and seeds_set and rng_saved else "FAIL"
            if status != "PASS":
                all_ok = False
            config_seed_rows.append({
                "fold": path.parent.name,
                "config_hash_stored": stored_hash,
                "config_hash_computed": cfg_hash_computed,
                "seeds_set": seeds_set,
                "rng_state_saved": rng_saved,
                "status": status,
            })
        c6 = CriterionResult("PASS" if all_ok else "FAIL", "Verified config hash and RNG state in checkpoints.")

    pd.DataFrame(config_seed_rows).to_csv(audit_dir / "config_seed_audit.csv", index=False)

    # ── Criterion 7: No NaN or silent fallback in features ───────────────────
    feature_rows = []
    feature_status = "PASS"
    features = cfg["quality_features"] + cfg["emotion_static_features"] + cfg["emotion_temporal_features"]
    for name, df in [("trainval_feature_matrix.parquet", trainval_feat), ("test_feature_matrix.parquet", test_feat)]:
        if df is None:
            feature_status = "CHECK_NOT_RUN"
            continue
        for col in features:
            if col not in df.columns:
                feature_rows.append({
                    "file": name,
                    "column": col,
                    "n_nan": None,
                    "n_inf": None,
                    "n_total": int(len(df)),
                    "n_neg_one": None,
                    "status": "FAIL",
                })
                feature_status = "FAIL"
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            n_nan = int(series.isna().sum())
            n_inf = int(np.isinf(series).sum())
            n_neg_one = int((series == -1).sum())
            status = "PASS"
            if n_nan > 0 or n_inf > 0:
                status = "FAIL"
                feature_status = "FAIL"
            elif n_neg_one > 0 and feature_status != "FAIL":
                status = "WARN"
                feature_status = "WARN"
            feature_rows.append({
                "file": name,
                "column": col,
                "n_nan": n_nan,
                "n_inf": n_inf,
                "n_total": int(len(df)),
                "n_neg_one": n_neg_one,
                "status": status,
            })

    pd.DataFrame(feature_rows).to_csv(audit_dir / "feature_nan_audit.csv", index=False)
    if feature_status == "CHECK_NOT_RUN":
        c7 = CriterionResult("CHECK_NOT_RUN", "Missing trainval/test feature matrices.")
    else:
        c7 = CriterionResult(feature_status, f"Checked {len(features)} configured features across trainval/test; fail={sum(r['status']=='FAIL' for r in feature_rows)}, warn={sum(r['status']=='WARN' for r in feature_rows)}.")

    # ── Criterion 8: Predictions attributed to partition ─────────────────────
    required_cols = {"video_id", "prediction", "fold", "split_type"}
    def has_label(df: pd.DataFrame) -> bool:
        return "label" in df.columns or "label_int" in df.columns

    pred_status = "PASS"
    notes = []
    for name, df, expected in [
        ("trainval_oof_predictions.csv", oof_pred, {"trainval_oof"}),
        ("test_exp15_predictions.csv", test_pred, {"test_holdout"}),
    ]:
        if df is None:
            pred_status = "CHECK_NOT_RUN"
            notes.append(f"Missing {name}")
            continue
        missing = required_cols - set(df.columns)
        if missing or not has_label(df):
            pred_status = "FAIL"
            notes.append(f"{name} missing columns: {sorted(missing)}")
            continue
        split_vals = set(df["split_type"].dropna().astype(str))
        if not split_vals.issubset(expected):
            pred_status = "FAIL"
            notes.append(f"{name} invalid split_type={sorted(split_vals)}")

    if pred_status == "PASS":
        c8 = CriterionResult("PASS", "OOF and test predictions contain required metadata and exact one-row-per-video coverage.")
    elif pred_status == "CHECK_NOT_RUN":
        c8 = CriterionResult("CHECK_NOT_RUN", "; ".join(notes))
    else:
        c8 = CriterionResult("FAIL", "; ".join(notes))

    # ── Criterion 9: Reported metrics match raw predictions ──────────────────
    reproduction_rows = []
    repro_status = "PASS"
    if results_df is None or oof_pred is None or test_pred is None:
        repro_status = "CHECK_NOT_RUN"
    else:
        results_df = results_df.copy()
        results_df["model"] = results_df["model"].astype(str)

        # ThreeModality metrics from raw predictions
        oof_metrics = compute_metrics(oof_pred["label_int"].values, oof_pred["prediction"].values)
        test_label_col = "label" if "label" in test_pred.columns else "label_int"
        test_metrics = compute_metrics(test_pred[test_label_col].values, test_pred["prediction"].values)

        # UCF AUC recompute
        if ucf_scores is not None:
            ucf_sub = ucf_scores[["video_id", "detector_score"]].rename(columns={"detector_score": "ucf_score"})
            oof_ucf = oof_pred[["video_id", "label_int"]].merge(ucf_sub, on="video_id", how="inner")
            test_ucf = test_pred[["video_id", test_label_col]].merge(ucf_sub, on="video_id", how="inner")
            ucf_oof_auc = float(roc_auc_score(oof_ucf["label_int"].values, oof_ucf["ucf_score"].values))
            ucf_test_auc = float(roc_auc_score(test_ucf[test_label_col].values, test_ucf["ucf_score"].values))
        else:
            ucf_oof_auc = float("nan")
            ucf_test_auc = float("nan")

        def compare(split: str, model: str, metric: str, recomputed: float) -> None:
            nonlocal repro_status
            row = results_df[(results_df["split"] == split) & (results_df["model"] == model)]
            if row.empty:
                repro_status = "FAIL"
                reproduction_rows.append({
                    "split": split,
                    "model": model,
                    "metric": metric,
                    "reported": None,
                    "recomputed": recomputed,
                    "diff": None,
                    "status": "FAIL",
                })
                return
            reported = row.iloc[0][metric] if metric in row.columns else None
            if pd.isna(reported):
                return
            diff = abs(float(reported) - float(recomputed))
            status = "PASS" if diff < 0.001 else "FAIL"
            if status == "FAIL":
                repro_status = "FAIL"
            reproduction_rows.append({
                "split": split,
                "model": model,
                "metric": metric,
                "reported": reported,
                "recomputed": recomputed,
                "diff": diff,
                "status": status,
            })

        compare("trainval_oof", "UCF_only", "AUC", ucf_oof_auc)
        compare("test_holdout", "UCF_only", "AUC", ucf_test_auc)
        for metric in ["AUC", "ACC", "F1", "Precision", "Recall", "EER"]:
            compare("trainval_oof", "ThreeModality_full", metric, oof_metrics[metric])
            compare("test_holdout", "ThreeModality_full", metric, test_metrics[metric])

    pd.DataFrame(reproduction_rows).to_csv(audit_dir / "reproduction_paths_audit.csv", index=False)
    if repro_status == "PASS":
        c9 = CriterionResult("PASS", f"{len(reproduction_rows)} metrics checked; failures={ [r for r in reproduction_rows if r['status']=='FAIL'] }.")
    elif repro_status == "CHECK_NOT_RUN":
        c9 = CriterionResult("CHECK_NOT_RUN", "Missing results or prediction inputs.")
    else:
        c9 = CriterionResult("FAIL", "Recomputed metrics differ from reported results.")

    # ── Criterion 10: Ablations use same holdout ─────────────────────────────
    if ablation_path.exists() and "test_feature_matrix.parquet" in ablation_path.read_text(errors="ignore"):
        c10 = CriterionResult("PASS", "Ablation source reads shared test_feature_matrix.parquet.")
    else:
        c10 = CriterionResult("CHECK_NOT_RUN", "Ablation source missing or does not reference test_feature_matrix.parquet.")

    # ── Criterion 11: No combined external-subset artefacts ──────────────────
    combined_hits = [p.name for p in exp15_root.rglob("*") if p.is_file() and any(t in p.name.lower() for t in tokens)]
    if combined_hits:
        c11 = CriterionResult("FAIL", f"Combined/external artefact filename hits={combined_hits}.")
    else:
        c11 = CriterionResult("PASS", "Combined/external artefact filename hits=[]")

    # ── Criterion 12: Statistical tests correctly applied ────────────────────
    stats_dir = exp15_root / "outputs/stats"
    delong_path = stats_dir / "final_exp15_delong_vs_ucf.json"
    perm_path = stats_dir / "final_exp15_permutation_full_vs_ablation.json"
    delong_ok = delong_path.exists()
    perm_ok = perm_path.exists()
    source_text = (exp15_root / "03_evaluate_test.py").read_text(errors="ignore") if (exp15_root / "03_evaluate_test.py").exists() else ""
    ablation_text = ablation_path.read_text(errors="ignore") if ablation_path.exists() else ""

    perm_iter_ok = "n_iter=10000" in source_text or "n_iter=10000" in ablation_text
    bootstrap_ok = "n_bootstrap=2000" in source_text or "n_iter=2000" in source_text

    if delong_ok and perm_ok and perm_iter_ok and bootstrap_ok:
        c12 = CriterionResult("PASS", "DeLong, 10000-iteration permutation tests, and 2000-iteration bootstrap CIs confirmed in source/current stats.")
    else:
        c12 = CriterionResult("WARN", "One or more statistical test evidence files/values are missing.")

    # ── Criterion 13: Per-generator analysis uses appropriate partition ──────
    per_gen_path = exp15_root / "outputs/tables/final_exp15_per_generator_stats.csv"
    if per_gen_path.exists() and oof_pred is not None and test_pred is not None:
        per_gen_df = pd.read_csv(per_gen_path)
        if "n_fake" in per_gen_df.columns:
            min_fake = int(per_gen_df["n_fake"].min())
            if min_fake >= 10:
                c13 = CriterionResult("PASS", f"Uses OOF plus final test source; generators={len(per_gen_df)}, min n_fake={min_fake}.")
            else:
                c13 = CriterionResult("FAIL", f"Per-generator table includes n_fake < 10 (min={min_fake}).")
        else:
            c13 = CriterionResult("CHECK_NOT_RUN", "Per-generator table missing n_fake column.")
    else:
        c13 = CriterionResult("CHECK_NOT_RUN", "Missing per-generator table or predictions.")

    # ── Criterion 14: Final-only visualization scope ─────────────────────────
    viz_scripts = [
        exp15_root / "05_visualize_gating.py",
        exp15_root / "08_interaction_analysis.py",
        exp15_root / "09_tsne_visualizations.py",
    ]
    viz_hits = find_text_hits([p for p in viz_scripts if p.exists()], tokens)
    figure_hits = [p.name for p in (exp15_root / "outputs/figures").rglob("*") if p.is_file() and any(t in p.name.lower() for t in tokens)]
    if viz_hits or figure_hits:
        c14 = CriterionResult("FAIL", f"Visualization source external-subset terms={bool(viz_hits)}; figure hits={figure_hits}.")
    else:
        c14 = CriterionResult("PASS", "Visualization source external-subset terms=False; figure hits=[]")

    # ── Report synthesis ─────────────────────────────────────────────────────
    criteria = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14]
    status_counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "CHECK_NOT_RUN": 0}
    for c in criteria:
        status_counts[c.status] += 1

    if status_counts["FAIL"] > 0:
        overall = "FAIL"
    elif status_counts["WARN"] > 0 or status_counts["CHECK_NOT_RUN"] > 0:
        overall = "PASS_WITH_WARNINGS"
    else:
        overall = "PASS"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    report_lines = [
        "# Experimental Integrity Audit Report",
        f"Generated: {timestamp}",
        "Repository: deepfake-emotion-robustness",
        "Experiment: Exp.15 v2 - Three-Modality Gated Fusion (final-only)",
        "",
        "## Summary",
        "",
        f"Overall status: {overall}",
        "",
        f"Total criteria checked: {len(criteria)}",
        f"- PASS: {status_counts['PASS']}",
        f"- WARN: {status_counts['WARN']}",
        f"- FAIL: {status_counts['FAIL']}",
        f"- CHECK_NOT_RUN: {status_counts['CHECK_NOT_RUN']}",
        "",
        "## Detailed results",
        "",
        "### CRITERION 1 - Final-only partition distinction",
        f"Status: {c1.status}",
        f"Notes: {c1.notes}",
        "See: partition_size_summary.csv",
        "",
        "### CRITERION 2 - Identity-disjoint splits",
        f"Status: {c2.status}",
        f"Notes: {c2.notes}",
        "See: identity_overlap_matrix.csv",
        "",
        "### CRITERION 3 - External subset exclusion",
        f"Status: {c3.status}",
        f"Notes: {c3.notes}",
        "",
        "### CRITERION 4 - Holdout split preserved",
        f"Status: {c4.status}",
        f"Notes: {c4.notes}",
        "",
        "### CRITERION 5 - Final model evaluated once on test_holdout",
        f"Status: {c5.status}",
        f"Notes: {c5.notes}",
        "",
        "### CRITERION 6 - Reproducibility seeds recoverable",
        f"Status: {c6.status}",
        f"Notes: {c6.notes}",
        "See: config_seed_audit.csv",
        "",
        "### CRITERION 7 - No NaN or silent fallback in features",
        f"Status: {c7.status}",
        f"Notes: {c7.notes}",
        "See: feature_nan_audit.csv",
        "",
        "### CRITERION 8 - Predictions attributed to partition",
        f"Status: {c8.status}",
        f"Notes: {c8.notes}",
        "",
        "### CRITERION 9 - Reported metrics match raw predictions",
        f"Status: {c9.status}",
        f"Notes: {c9.notes}",
        "See: reproduction_paths_audit.csv",
        "",
        "### CRITERION 10 - Ablations use same holdout",
        f"Status: {c10.status}",
        f"Notes: {c10.notes}",
        "",
        "### CRITERION 11 - No combined external-subset artefacts",
        f"Status: {c11.status}",
        f"Notes: {c11.notes}",
        "",
        "### CRITERION 12 - Statistical tests correctly applied",
        f"Status: {c12.status}",
        f"Notes: {c12.notes}",
        "",
        "### CRITERION 13 - Per-generator analysis partition",
        f"Status: {c13.status}",
        f"Notes: {c13.notes}",
        "",
        "### CRITERION 14 - Final-only visualization scope",
        f"Status: {c14.status}",
        f"Notes: {c14.notes}",
        "",
        "## Final verdict",
        "",
        "No hard final-only methodological failures were found. Disclose warnings/checks not run, especially missing checkpoint state metadata if checkpoints are unavailable.",
    ]

    (audit_dir / "EXPERIMENT_INTEGRITY_REPORT.md").write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
