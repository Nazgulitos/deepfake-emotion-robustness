"""
Stage 01 — Build three-modality feature matrix from predefined train/val/test split.

Computes per-video features for three semantically distinct modalities:
  M_q: quality (face detection confidence, size, frame count)
  M_s: emotion static (aggregated mean emotion scores)
  M_t: emotion temporal (dynamics: std over frames, variation, entropy)

The temporal std features for 40 EMONET-FACE categories are computed from
frame-level predictions (final_frame_emotion_predictions.csv).

Reads:
  datasets/metadata/final_face_manifest.csv
  datasets/emotion_annotated/metadata/final_video_emotion_features.csv
  datasets/emotion_annotated/metadata/final_frame_emotion_predictions.csv

Writes:
  outputs/predictions/trainval_feature_matrix.parquet
  outputs/predictions/test_feature_matrix.parquet
  outputs/tables/final_exp15_modality_correlation.csv
  outputs/figures/final_exp15_modality_correlation_heatmap.png

Run from project root:
  python scripts/exp15_three_modality/01_prepare_features.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from utils import get_project_root, load_config, require_file, set_seeds, setup_logger

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()

SCORE_COLS = [
    "score_affection", "score_amusement", "score_anger", "score_astonishment_surprise",
    "score_awe", "score_bitterness", "score_concentration", "score_confusion",
    "score_contemplation", "score_contempt", "score_contentment", "score_disappointment",
    "score_disgust", "score_distress", "score_doubt", "score_elation", "score_embarrassment",
    "score_emotional_numbness", "score_fatigue_exhaustion", "score_fear", "score_helplessness",
    "score_hope_enthusiasm_optimism", "score_impatience_and_irritability", "score_infatuation",
    "score_interest", "score_intoxication_altered_states_of_consciousness",
    "score_jealousy_&_envy", "score_longing", "score_malevolence_malice", "score_pain",
    "score_pleasure_ecstasy", "score_pride", "score_relief", "score_sadness",
    "score_sexual_lust", "score_shame", "score_sourness", "score_teasing",
    "score_thankfulness_gratitude", "score_triumph",
]


def build_quality_features(face_manifest: pd.DataFrame) -> pd.DataFrame:
    grp = face_manifest.groupby("video_id")
    q = pd.DataFrame()
    q["face_det_score_mean"] = grp["det_score"].mean()
    q["face_det_score_std"] = grp["det_score"].std().fillna(0.0)
    face_area = face_manifest["face_width"] * face_manifest["face_height"]
    face_manifest = face_manifest.copy()
    face_manifest["face_area"] = face_area
    grp2 = face_manifest.groupby("video_id")
    q["face_size_mean"] = grp2["face_area"].mean()
    q["face_size_std"] = grp2["face_area"].std().fillna(0.0)
    q["frame_count"] = grp["frame_id"].count()
    return q.reset_index()


def build_temporal_std_features(frame_emotion: pd.DataFrame) -> pd.DataFrame:
    """Compute per-video std of each score column (temporal variability)."""
    std_cols = {f"std_{c}": (c, "std") for c in SCORE_COLS}
    agg_dict = {c: ["std"] for c in SCORE_COLS}
    grp = frame_emotion.groupby("video_id")[SCORE_COLS].std().fillna(0.0)
    grp.columns = [f"std_{c}" for c in SCORE_COLS]
    return grp.reset_index()


def build_feature_matrix(
    face_manifest: pd.DataFrame,
    video_emotion: pd.DataFrame,
    frame_emotion: pd.DataFrame,
    qual_cols: list,
    emo_static_cols: list,
    emo_temporal_base_cols: list,
    logger,
) -> pd.DataFrame:
    logger.info("Building quality features from face manifest...")
    qual = build_quality_features(face_manifest)

    logger.info("Building temporal std features from frame-level predictions...")
    temp_std = build_temporal_std_features(frame_emotion)

    # Per-video metadata from face manifest
    meta_cols = ["video_id", "label", "split", "manipulation_family", "identity"]
    face_meta = (
        face_manifest[meta_cols]
        .drop_duplicates("video_id")
        .reset_index(drop=True)
    )

    # Merge all together
    df = (
        face_meta
        .merge(qual, on="video_id", how="inner")
        .merge(
            video_emotion[["video_id", "dominant_emotion"] + emo_static_cols + emo_temporal_base_cols],
            on="video_id", how="inner",
        )
        .merge(temp_std, on="video_id", how="inner")
    )

    df = df.rename(columns={"manipulation_family": "forgery_family"})
    df["label_int"] = (df["label"] == "fake").astype(int)
    logger.info(f"After merge: {len(df)} videos")
    return df


def validate_features(df: pd.DataFrame, name: str, all_feature_cols: list, logger) -> None:
    nan_counts = df[all_feature_cols].isnull().sum()
    if nan_counts.any():
        bad = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(f"[{name}] NaN in feature columns: {bad}")
    logger.info(f"[{name}] shape={df.shape}  label: {df['label'].value_counts().to_dict()}")
    logger.info(f"[{name}] forgery: {df['forgery_family'].value_counts().to_dict()}")
    logger.info(f"[{name}] split: {df['split'].value_counts().to_dict()}")


def save_modality_correlation(df: pd.DataFrame, qual_cols: list, emo_static_cols: list,
                               emo_temporal_cols: list, table_dir: Path, fig_dir: Path,
                               logger) -> None:
    """Compute and save mean-feature correlation matrix across modalities."""
    import warnings
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available — skipping correlation heatmap")
        return

    # Use means of each modality block as representative scalar
    df_corr = pd.DataFrame({
        "quality": df[qual_cols].mean(axis=1),
        "emo_static": df[emo_static_cols].mean(axis=1),
        "emo_temporal": df[emo_temporal_cols].mean(axis=1),
    })
    corr = df_corr.corr()
    corr.to_csv(table_dir / "final_exp15_modality_correlation.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1,
                    ax=ax, square=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "final_exp15_modality_correlation_heatmap.png", dpi=300)
    plt.close(fig)

    logger.info(f"Modality correlation matrix:\n{corr.to_string()}")
    max_off = corr.where(~np.eye(3, dtype=bool)).abs().max().max()
    if max_off > 0.5:
        logger.warning(
            f"Max off-diagonal modality correlation = {max_off:.3f} > 0.5 — "
            "modalities may overlap significantly."
        )
    else:
        logger.info(f"Max off-diagonal modality correlation = {max_off:.3f} (acceptable)")


def main():
    set_seeds(42)
    cfg = load_config(str(CONFIG_PATH))
    out_dir = ROOT / cfg["paths"]["output_root"]
    pred_dir = out_dir / "predictions"
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    log_dir = out_dir / "logs"
    for d in [pred_dir, table_dir, fig_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15_tm.prepare", str(log_dir / "run.log"))
    logger.info("=== Stage 01: Prepare Three-Modality Features ===")

    qual_cols = cfg["quality_features"]
    emo_static_cols = cfg["emotion_static_features"]
    emo_temporal_base_cols = [
        c for c in cfg["emotion_temporal_features"] if not c.startswith("std_score_")
    ]
    emo_temporal_std_cols = [
        c for c in cfg["emotion_temporal_features"] if c.startswith("std_score_")
    ]
    emo_temporal_cols = emo_temporal_base_cols + emo_temporal_std_cols

    logger.info(f"Quality features: {len(qual_cols)}")
    logger.info(f"Emotion static features: {len(emo_static_cols)}")
    logger.info(f"Emotion temporal features: {len(emo_temporal_cols)}")

    # ── Load inputs ────────────────────────────────────────────────────────────
    face = pd.read_csv(require_file(ROOT / cfg["paths"]["face_manifest"], "face manifest"))
    video_emo = pd.read_csv(require_file(ROOT / cfg["paths"]["video_emotion"], "video emotion"))
    frame_emo = pd.read_csv(require_file(ROOT / cfg["paths"]["frame_emotion"], "frame emotion"))

    logger.info(f"Face manifest: {len(face)} rows, {face['video_id'].nunique()} videos")
    logger.info(f"Video emotion: {len(video_emo)} videos")
    logger.info(f"Frame emotion: {len(frame_emo)} rows, {frame_emo['video_id'].nunique()} videos")

    # Verify required columns exist in video_emotion
    missing_static = [c for c in emo_static_cols if c not in video_emo.columns]
    missing_temporal_base = [c for c in emo_temporal_base_cols if c not in video_emo.columns]
    if missing_static:
        raise ValueError(f"Missing emotion_static cols in video_emotion: {missing_static}")
    if missing_temporal_base:
        raise ValueError(f"Missing temporal base cols in video_emotion: {missing_temporal_base}")

    # Verify frame emotion has score columns for std computation
    missing_frame = [c for c in SCORE_COLS if c not in frame_emo.columns]
    if missing_frame:
        raise ValueError(f"Missing score cols in frame_emotion: {missing_frame}")

    # ── Build feature matrix ───────────────────────────────────────────────────
    all_feature_cols = qual_cols + emo_static_cols + emo_temporal_cols
    df = build_feature_matrix(
        face, video_emo, frame_emo,
        qual_cols, emo_static_cols, emo_temporal_base_cols,
        logger,
    )

    # Verify std cols are present after merge
    missing_std = [c for c in emo_temporal_std_cols if c not in df.columns]
    if missing_std:
        raise ValueError(f"Temporal std cols missing after merge: {missing_std}")

    # ── Split ──────────────────────────────────────────────────────────────────
    trainval = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    validate_features(trainval, "trainval", all_feature_cols, logger)
    validate_features(test, "test", all_feature_cols, logger)

    # Identity disjointness
    tv_ids = set(trainval["identity"].dropna())
    te_ids = set(test["identity"].dropna())
    overlap = tv_ids & te_ids
    if overlap:
        logger.warning(f"Identity overlap trainval/test: {overlap}")
    else:
        logger.info("Identity disjoint check PASSED")

    trainval.to_parquet(pred_dir / "trainval_feature_matrix.parquet", index=False)
    test.to_parquet(pred_dir / "test_feature_matrix.parquet", index=False)
    logger.info(f"Saved trainval ({len(trainval)}) and test ({len(test)}) feature matrices")

    # ── Modality correlation ───────────────────────────────────────────────────
    save_modality_correlation(df, qual_cols, emo_static_cols, emo_temporal_cols,
                               table_dir, fig_dir, logger)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== Feature Matrix Summary ===")
    print(f"Quality features    : {len(qual_cols)}")
    print(f"Emotion static      : {len(emo_static_cols)}")
    print(f"Emotion temporal    : {len(emo_temporal_cols)}")
    print(f"Total feature cols  : {len(all_feature_cols)}")
    print(f"TrainVal videos     : {len(trainval)}")
    print(f"Test holdout        : {len(test)}")
    print(f"Identity-disjoint   : {len(overlap) == 0}")


if __name__ == "__main__":
    main()
