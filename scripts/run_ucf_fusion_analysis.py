"""Detector + emotion/quality fusion, tuned models, and error analysis.

Reads:
  datasets/detector_processed/{final,pilot}_{detector}_scores.csv
  datasets/detector_processed/{final,pilot}_{detector}_frame_scores.csv
  datasets/emotion_annotated/metadata/{final,pilot}_video_emotion_features.csv
  datasets/metadata/{final,pilot}_face_manifest.csv

Writes: outputs/results/YYYY-MM-DD/exp12/
  tables/{subset}_{exp_id}_{detector}_fusion_results.csv
  tables/final_{exp_id}_error_by_{emotion,arousal,forgery_family}.csv
  stats/final_{exp_id}_model_selection.json
  run_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.statistical_tests import delong_compare, permutation_auc_test
from src.analysis.subgroup_auc import add_arousal_tercile
from src.utils.logging_utils import setup_logging
from src.utils.run_metadata import now_utc, write_run_metadata

SEED = 42

EMOTION_FEATURES = [
    "mean_arousal",
    "mean_valence",
    "max_arousal",
    "arousal_variation",
    "emotion_entropy",
    "transition_rate",
    "neutral_ratio",
]

QUALITY_FEATURES = [
    "n_frames",
    "n_face_frames",
    "detector_frame_score_std",
    "detector_frame_score_min",
    "detector_frame_score_max",
    "blur_laplacian_mean",
    "blur_laplacian_std",
    "face_det_score_mean",
    "face_det_score_std",
    "face_area_mean",
    "face_area_std",
    "face_width_mean",
    "face_height_mean",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp_id", default="exp12")
    p.add_argument("--detector", default="ucf",
                   choices=["ucf", "huggingface", "xception"])
    p.add_argument("--date", default=None)
    p.add_argument("--n_permutations", type=int, default=2000)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _score_paths(subset: str, detector: str) -> tuple[Path, Path]:
    if detector == "ucf":
        return (
            Path(f"datasets/detector_processed/{subset}_ucf_scores.csv"),
            Path(f"datasets/detector_processed/{subset}_ucf_frame_scores.csv"),
        )
    if detector == "huggingface":
        return (
            Path(f"datasets/detector_processed/{subset}_huggingface_scores.csv"),
            Path(f"datasets/detector_processed/{subset}_huggingface_frame_scores.csv"),
        )
    if detector == "xception":
        if subset == "final":
            return (
                Path("outputs/deepfakebench_scores/ThesisFinal_xception_video_scores.csv"),
                Path("outputs/deepfakebench_scores/ThesisFinal_xception_frame_scores.csv"),
            )
        return (
            Path("outputs/deepfakebench_scores/ThesisPilot_xception_video_scores.csv"),
            Path("outputs/deepfakebench_scores/ThesisPilot_xception_frame_scores.csv"),
        )
    return (
        Path(f"datasets/detector_processed/{subset}_detector_scores.csv"),
        Path(f"datasets/detector_processed/{subset}_frame_detector_scores.csv"),
    )


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _quality_from_faces(path: Path) -> pd.DataFrame:
    face = _safe_read(path)
    face = face.copy()
    face["face_area"] = face["face_width"].astype(float) * face["face_height"].astype(float)

    try:
        import cv2

        def blur_score(face_path: str) -> float:
            img = cv2.imread(str(face_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return float("nan")
            return float(cv2.Laplacian(img, cv2.CV_64F).var())

        face["blur_laplacian"] = face["face_path"].map(blur_score)
    except Exception:
        face["blur_laplacian"] = np.nan

    id_cols = [
        c
        for c in [
            "identity",
            "manipulation_family",
            "manipulation_type",
            "source_subset",
            "split",
            "label",
        ]
        if c in face.columns
    ]
    meta = (
        face.groupby("video_id", dropna=False)[id_cols]
        .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan)
        .reset_index()
    )
    agg = (
        face.groupby("video_id", dropna=False)
        .agg(
            face_det_score_mean=("det_score", "mean"),
            face_det_score_std=("det_score", "std"),
            face_area_mean=("face_area", "mean"),
            face_area_std=("face_area", "std"),
            face_width_mean=("face_width", "mean"),
            face_height_mean=("face_height", "mean"),
            blur_laplacian_mean=("blur_laplacian", "mean"),
            blur_laplacian_std=("blur_laplacian", "std"),
        )
        .reset_index()
    )
    return meta.merge(agg, on="video_id", how="inner")


def _quality_from_frame_scores(path: Path) -> pd.DataFrame:
    frame = _safe_read(path)
    return (
        frame.groupby("video_id", dropna=False)
        .agg(
            detector_frame_score_std=("detector_score", "std"),
            detector_frame_score_min=("detector_score", "min"),
            detector_frame_score_max=("detector_score", "max"),
        )
        .reset_index()
    )


def _load_subset(subset: str, detector: str, logger: logging.Logger) -> pd.DataFrame:
    scores_path, frame_scores_path = _score_paths(subset, detector)
    scores = _safe_read(scores_path)
    emotion = _safe_read(Path(f"datasets/emotion_annotated/metadata/{subset}_video_emotion_features.csv"))
    faces = _quality_from_faces(Path(f"datasets/metadata/{subset}_face_manifest.csv"))
    frame_quality = _quality_from_frame_scores(frame_scores_path)

    emotion_slim = emotion[[c for c in emotion.columns if c == "video_id" or c not in scores.columns]]
    df = scores.merge(emotion_slim, on="video_id", how="inner")
    df = df.merge(faces, on="video_id", how="left", suffixes=("", "_face"))
    df = df.merge(frame_quality, on="video_id", how="left")

    if "y" not in df.columns:
        if pd.api.types.is_numeric_dtype(df["label"]):
            df["y"] = df["label"].astype(int)
        else:
            df["y"] = df["label"].astype(str).map({"fake": 1, "real": 0}).astype(int)
    for col in ["identity", "manipulation_type", "source_subset"]:
        face_col = f"{col}_face"
        if col not in df.columns and face_col in df.columns:
            df[col] = df[face_col]
        elif col in df.columns and face_col in df.columns:
            df[col] = df[col].combine_first(df[face_col])
    if "identity" not in df.columns:
        logger.warning("%s has no identity column; falling back to video_id groups", subset)
        df["identity"] = df["video_id"]
    else:
        df["identity"] = df["identity"].fillna(df["video_id"])

    logger.info("%s %s merged rows=%d columns=%d", subset, detector, len(df), len(df.columns))
    return df


def _metrics(y: np.ndarray, scores: np.ndarray, model: str) -> dict[str, Any]:
    pred = (scores >= 0.5).astype(int)
    return {
        "model": model,
        "AUC": float(roc_auc_score(y, scores)),
        "ACC": float(accuracy_score(y, pred)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "n": int(len(y)),
    }


def _available(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def _lr_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )


def _make_xgb_model(params: dict[str, Any]) -> Any:
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
            **params,
        )
    except Exception:
        mapped = {
            "n_estimators": params.get("n_estimators", 100),
            "learning_rate": params.get("learning_rate", 0.1),
            "max_depth": params.get("max_depth", 3),
        }
        return GradientBoostingClassifier(random_state=SEED, **mapped)


def _fit_predict_oof(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    logger: logging.Logger,
    params_grid: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    clean = df.dropna(subset=["y", "detector_score", "identity"]).reset_index(drop=True)
    X = clean[features]
    y = clean["y"].astype(int).to_numpy()
    groups = clean["identity"].astype(str).to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(clean), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    chosen_params: list[dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        train_groups = groups[train_idx]

        best_params: dict[str, Any] = {}
        if params_grid:
            best_auc = -np.inf
            inner_splits = min(3, len(np.unique(train_groups)))
            for params in params_grid:
                inner_scores = []
                if inner_splits >= 2:
                    inner = GroupKFold(n_splits=inner_splits)
                    for tr_i, va_i in inner.split(X_train, y_train, train_groups):
                        if len(np.unique(y_train[va_i])) < 2:
                            continue
                        clf = Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("model", _make_xgb_model(params)),
                            ]
                        )
                        clf.fit(X_train.iloc[tr_i], y_train[tr_i])
                        score = clf.predict_proba(X_train.iloc[va_i])[:, 1]
                        inner_scores.append(float(roc_auc_score(y_train[va_i], score)))
                mean_auc = float(np.nanmean(inner_scores)) if inner_scores else float("-inf")
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_params = params
            clf = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", _make_xgb_model(best_params)),
                ]
            )
            chosen_params.append(best_params)
        else:
            clf = _lr_pipeline()

        clf.fit(X_train, y_train)
        oof[val_idx] = clf.predict_proba(X_val)[:, 1]
        fold_auc = (
            float(roc_auc_score(y_val, oof[val_idx]))
            if len(np.unique(y_val)) > 1
            else float("nan")
        )
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "fold_auc": fold_auc,
                "params": best_params,
            }
        )
        logger.info("%s fold %d auc=%.4f n_val=%d", model_name, fold, fold_auc, len(val_idx))

    pred = np.full(len(df), np.nan)
    pred[clean.index.to_numpy()] = oof
    return pred, {
        "model": model_name,
        "features": features,
        "folds": fold_rows,
        "chosen_params": chosen_params,
    }


def _fit_predict_pilot(
    final_df: pd.DataFrame,
    pilot_df: pd.DataFrame,
    features: list[str],
    model_name: str,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    train = final_df.dropna(subset=["y"]).copy()
    test = pilot_df.dropna(subset=["y"]).copy()
    if model_name.endswith("_lr"):
        clf = _lr_pipeline()
    else:
        clf = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", _make_xgb_model(params or {})),
            ]
        )
    clf.fit(train[features], train["y"].astype(int).to_numpy())
    return clf.predict_proba(test[features])[:, 1]


def _fit_full_xgb(df: pd.DataFrame, features: list[str],
                  params: dict[str, Any] | None = None) -> Pipeline:
    train = df.dropna(subset=["y"]).copy()
    clf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", _make_xgb_model(params or {})),
        ]
    )
    clf.fit(train[features], train["y"].astype(int).to_numpy())
    return clf


def _quality_feature_importance(
    df: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    clean = df.dropna(subset=["y"]).copy()
    clf = _fit_full_xgb(clean, features, params)
    transformed = clf.named_steps["imputer"].transform(clean[features])
    model = clf.named_steps["model"]

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        importance = np.abs(values).mean(axis=0)
        method = "mean_abs_shap"
    except Exception as exc:
        logger.warning("SHAP importance failed; falling back to model importances: %s", exc)
        importance = getattr(model, "feature_importances_", np.zeros(len(features)))
        method = "model_feature_importance"

    out = pd.DataFrame(
        {
            "feature": features,
            "importance": importance,
            "method": method,
        }
    ).sort_values("importance", ascending=False)
    total = out["importance"].sum()
    out["importance_share"] = out["importance"] / total if total > 0 else 0.0
    return out


def _modal_params(chosen: list[dict[str, Any]]) -> dict[str, Any]:
    if not chosen:
        return {}
    encoded = [json.dumps(p, sort_keys=True) for p in chosen]
    winner = max(set(encoded), key=encoded.count)
    return json.loads(winner)


def _error_table(df: pd.DataFrame, score_col: str, group_col: str, min_n: int = 5) -> pd.DataFrame:
    rows = []
    work = df.dropna(subset=["y", score_col]).copy()
    work["pred"] = (work[score_col] >= 0.5).astype(int)
    work["error"] = (work["pred"] != work["y"].astype(int)).astype(int)
    work["fp"] = ((work["pred"] == 1) & (work["y"].astype(int) == 0)).astype(int)
    work["fn"] = ((work["pred"] == 0) & (work["y"].astype(int) == 1)).astype(int)
    for value, g in work.groupby(group_col, dropna=False):
        if len(g) < min_n:
            continue
        auc = (
            float(roc_auc_score(g["y"].astype(int), g[score_col]))
            if g["y"].nunique() == 2
            else float("nan")
        )
        rows.append(
            {
                group_col: value,
                "n": int(len(g)),
                "n_real": int((g["y"] == 0).sum()),
                "n_fake": int((g["y"] == 1).sum()),
                "error_rate": float(g["error"].mean()),
                "fp_rate": float(g["fp"].sum() / max((g["y"] == 0).sum(), 1)),
                "fn_rate": float(g["fn"].sum() / max((g["y"] == 1).sum(), 1)),
                "AUC": auc,
            }
        )
    return pd.DataFrame(rows).sort_values(["error_rate", "n"], ascending=[False, False])


def _save_table(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.rename(path)


def _save_tex(df: pd.DataFrame, path: Path) -> None:
    fmt = df.copy()
    for col in fmt.select_dtypes(include=[float]).columns:
        fmt[col] = fmt[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "--")
    tmp = path.with_suffix(path.suffix + ".tmp")
    fmt.to_latex(tmp, index=False, escape=True)
    tmp.rename(path)


def main() -> None:
    args = parse_args()
    start_time = now_utc()
    date_str = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path("outputs/results") / date_str / args.exp_id
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "stats").mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger(args.exp_id)
    logger.info("Starting %s detector=%s", args.exp_id, args.detector)

    final_df = _load_subset("final", args.detector, logger)
    try:
        pilot_df = _load_subset("pilot", args.detector, logger)
    except FileNotFoundError as exc:
        logger.warning("Pilot data unavailable for detector=%s: %s", args.detector, exc)
        pilot_df = None

    emotion = _available(final_df, EMOTION_FEATURES)
    quality = _available(final_df, QUALITY_FEATURES)
    prefix = args.detector
    feature_sets = {
        "emotion_only_lr": emotion,
        "quality_only_lr": quality,
        f"{prefix}_quality_lr": ["detector_score"] + quality,
        f"{prefix}_emotion_lr": ["detector_score"] + emotion,
        f"{prefix}_emotion_quality_lr": ["detector_score"] + emotion + quality,
    }

    grid = list(
        ParameterGrid(
            {
                "max_depth": [2, 3],
                "learning_rate": [0.03, 0.1],
                "n_estimators": [50, 100],
            }
        )
    )
    xgb_feature_sets = {
        "quality_only_xgb_tuned": quality,
        f"{prefix}_quality_xgb_tuned": ["detector_score"] + quality,
        f"{prefix}_emotion_quality_xgb_tuned": ["detector_score"] + emotion + quality,
    }

    y_final = final_df["y"].astype(int).to_numpy()
    baseline_name = f"{prefix}_only"
    final_rows = [_metrics(y_final, final_df["detector_score"].to_numpy(), baseline_name)]
    model_details: dict[str, Any] = {
        "emotion_features": emotion,
        "quality_features": quality,
        "xgb_grid": grid,
    }
    prediction_cols: dict[str, str] = {baseline_name: "detector_score"}

    for model_name, features in feature_sets.items():
        scores, details = _fit_predict_oof(final_df, features, model_name, logger)
        col = f"{model_name}_score"
        final_df[col] = scores
        prediction_cols[model_name] = col
        valid = final_df.dropna(subset=[col, "y"])
        final_rows.append(_metrics(valid["y"].astype(int).to_numpy(), valid[col].to_numpy(), model_name))
        model_details[model_name] = details

    xgb_details_by_model: dict[str, Any] = {}
    for model_name, features in xgb_feature_sets.items():
        xgb_scores, xgb_details = _fit_predict_oof(
            final_df,
            features,
            model_name,
            logger,
            params_grid=grid,
        )
        xgb_col = f"{model_name}_score"
        final_df[xgb_col] = xgb_scores
        prediction_cols[model_name] = xgb_col
        valid = final_df.dropna(subset=[xgb_col, "y"])
        final_rows.append(
            _metrics(valid["y"].astype(int).to_numpy(), valid[xgb_col].to_numpy(), model_name)
        )
        model_details[model_name] = xgb_details
        xgb_details_by_model[model_name] = xgb_details

    final_results = pd.DataFrame(final_rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    final_csv = out_dir / "tables" / f"final_{args.exp_id}_{args.detector}_fusion_results.csv"
    _save_table(final_results, final_csv)
    _save_tex(final_results, out_dir / "tables" / f"final_{args.exp_id}_{args.detector}_fusion_results.tex")
    logger.info("Final results:\n%s", final_results.to_string(index=False))

    best_model = str(final_results.iloc[0]["model"])
    best_col = prediction_cols[best_model]
    if best_model != baseline_name:
        y = final_df.dropna(subset=[best_col, "detector_score", "y"])["y"].astype(int).to_numpy()
        model_details["best_vs_detector_delong"] = delong_compare(
            y,
            final_df.dropna(subset=[best_col, "detector_score", "y"])[best_col].to_numpy(),
            final_df.dropna(subset=[best_col, "detector_score", "y"])["detector_score"].to_numpy(),
        )
        model_details["best_vs_detector_permutation"] = permutation_auc_test(
            y,
            final_df.dropna(subset=[best_col, "detector_score", "y"])[best_col].to_numpy(),
            final_df.dropna(subset=[best_col, "detector_score", "y"])["detector_score"].to_numpy(),
            n_permutations=args.n_permutations,
            seed=SEED,
        )

    emotion_model = f"{prefix}_emotion_lr"
    quality_model = f"{prefix}_quality_xgb_tuned"
    all_model = f"{prefix}_emotion_quality_xgb_tuned"
    for test_name, model_a, model_b in [
        ("emotion_vs_detector", emotion_model, baseline_name),
        ("emotion_quality_vs_quality", all_model, quality_model),
    ]:
        col_a = prediction_cols.get(model_a)
        col_b = prediction_cols.get(model_b)
        if col_a and col_b:
            valid = final_df.dropna(subset=[col_a, col_b, "y"])
            model_details[f"permutation_{test_name}"] = permutation_auc_test(
                valid["y"].astype(int).to_numpy(),
                valid[col_a].to_numpy(),
                valid[col_b].to_numpy(),
                n_permutations=args.n_permutations,
                seed=SEED,
            )

    final_df = add_arousal_tercile(final_df)
    final_df["forgery_family_error_group"] = final_df["manipulation_family"].fillna("real")
    for group_col, suffix in [
        ("dominant_emotion", "emotion"),
        ("arousal_tercile", "arousal"),
        ("forgery_family_error_group", "forgery_family"),
    ]:
        err = _error_table(final_df, best_col, group_col)
        _save_table(err, out_dir / "tables" / f"final_{args.exp_id}_error_by_{suffix}.csv")

    best_xgb_params_by_model = {
        model_name: _modal_params(details.get("chosen_params", []))
        for model_name, details in xgb_details_by_model.items()
    }
    quality_xgb_model = f"{prefix}_quality_xgb_tuned"
    if quality_xgb_model in xgb_feature_sets:
        importance = _quality_feature_importance(
            final_df,
            xgb_feature_sets[quality_xgb_model],
            best_xgb_params_by_model.get(quality_xgb_model, {}),
            logger,
        )
        _save_table(
            importance,
            out_dir / "tables" / f"final_{args.exp_id}_{args.detector}_quality_feature_importance.csv",
        )

    pilot_csv = out_dir / "tables" / f"pilot_{args.exp_id}_{args.detector}_fusion_results.csv"
    pilot_tex = out_dir / "tables" / f"pilot_{args.exp_id}_{args.detector}_fusion_results.tex"
    if pilot_df is not None:
        pilot_rows = [_metrics(pilot_df["y"].astype(int).to_numpy(), pilot_df["detector_score"].to_numpy(), baseline_name)]
        for model_name, features in feature_sets.items():
            scores = _fit_predict_pilot(final_df, pilot_df, features, model_name)
            pilot_rows.append(_metrics(pilot_df["y"].astype(int).to_numpy(), scores, model_name))
        for model_name, features in xgb_feature_sets.items():
            scores = _fit_predict_pilot(
                final_df,
                pilot_df,
                features,
                model_name,
                params=best_xgb_params_by_model.get(model_name, {}),
            )
            pilot_rows.append(_metrics(pilot_df["y"].astype(int).to_numpy(), scores, model_name))
        pilot_results = pd.DataFrame(pilot_rows).sort_values("AUC", ascending=False).reset_index(drop=True)
        _save_table(pilot_results, pilot_csv)
        _save_tex(pilot_results, pilot_tex)
        logger.info("Pilot results:\n%s", pilot_results.to_string(index=False))
    else:
        pilot_csv.unlink(missing_ok=True)
        pilot_tex.unlink(missing_ok=True)

    model_details["best_model_final_oof"] = best_model
    model_details["xgb_modal_params_for_pilot"] = best_xgb_params_by_model
    stats_path = out_dir / "stats" / f"final_{args.exp_id}_model_selection.json"
    tmp = stats_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(model_details, indent=2, default=str), encoding="utf-8")
    tmp.rename(stats_path)

    write_run_metadata(
        out_dir,
        exp_id=args.exp_id,
        subset=f"final+pilot/{args.detector}",
        seed=SEED,
        cli_args=vars(args),
        start_time=start_time,
        end_time=now_utc(),
    )
    logger.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()
