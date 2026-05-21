"""
Stage 01 — Build feature matrices from the predefined train/val/test split.

When use_predefined_split=true (default):
  - trainval_feature_matrix.parquet  → split in (train, val)  — used for GroupKFold CV
  - test_feature_matrix.parquet      → split == test           — held out, never seen during training

Reads:
  datasets/metadata/final_face_manifest.csv
  datasets/emotion_annotated/metadata/final_video_emotion_features.csv
  datasets/detector_processed/final_ucf_scores.csv
  datasets/detector_processed/final_ucf_frame_scores.csv

Writes:
  outputs/predictions/trainval_feature_matrix.parquet
  outputs/predictions/test_feature_matrix.parquet

Run from project root:
  python scripts/exp15_modality_gated/01_prepare_features.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd

from utils import get_project_root, load_config, require_file, set_seeds, setup_logger

CONFIG_PATH = HERE / "config.yaml"
ROOT = get_project_root()


def build_quality_features(face_manifest: pd.DataFrame, ucf_frame_scores: pd.DataFrame) -> pd.DataFrame:
    face_grp = face_manifest.groupby("video_id")
    qual = pd.DataFrame()
    qual["face_det_score_mean"] = face_grp["det_score"].mean()
    qual["face_size_mean"] = face_grp.apply(lambda g: (g["face_width"] * g["face_height"]).mean())
    qual["frame_count"] = face_grp["frame_id"].count()

    ucf_var = ucf_frame_scores.groupby("video_id")["detector_score"].std().rename("ucf_score_variance")
    qual = qual.join(ucf_var, how="left")
    qual["ucf_score_variance"] = qual["ucf_score_variance"].fillna(0.0)
    return qual.reset_index()


def build_feature_matrix(
    face_manifest_path: Path,
    emotion_path: Path,
    ucf_scores_path: Path,
    ucf_frame_scores_path: Path,
    emo_feature_cols: list,
    quality_feature_cols: list,
    logger,
) -> pd.DataFrame:

    logger.info(f"Reading face manifest: {face_manifest_path}")
    face = pd.read_csv(face_manifest_path)

    logger.info(f"Reading emotion features: {emotion_path}")
    emo = pd.read_csv(emotion_path)

    logger.info(f"Reading UCF video scores: {ucf_scores_path}")
    ucf = pd.read_csv(ucf_scores_path)

    logger.info(f"Reading UCF frame scores: {ucf_frame_scores_path}")
    ucf_frame = pd.read_csv(ucf_frame_scores_path)

    # Per-video metadata
    meta_cols = ["video_id", "label", "identity", "manipulation_family", "split"]
    face_video = face[meta_cols + ["det_score", "face_width", "face_height", "frame_id"]].copy()

    qual = build_quality_features(face_video, ucf_frame)

    video_meta = (
        face_video.groupby("video_id")
        .agg(
            label=("label", "first"),
            identity=("identity", "first"),
            manipulation_family=("manipulation_family", "first"),
            split=("split", "first"),
        )
        .reset_index()
    )

    # Verify emotion columns
    missing = [c for c in emo_feature_cols if c not in emo.columns]
    if missing:
        raise ValueError(f"Missing emotion columns: {missing}")
    emo_sub = emo[["video_id", "dominant_emotion"] + emo_feature_cols].copy()

    ucf_sub = ucf[["video_id", "detector_score"]].copy()

    df = (
        video_meta
        .merge(ucf_sub, on="video_id", how="inner")
        .merge(emo_sub, on="video_id", how="inner")
        .merge(qual[["video_id"] + quality_feature_cols], on="video_id", how="inner")
    )
    logger.info(f"After inner join: {len(df)} videos")

    df = df.rename(columns={"manipulation_family": "forgery_family"})
    df["label_int"] = (df["label"] == "fake").astype(int)
    return df


def validate_matrix(df: pd.DataFrame, name: str, feature_cols: list, logger) -> None:
    nan_counts = df[feature_cols].isnull().sum()
    if nan_counts.any():
        bad = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(f"[{name}] NaN in feature columns: {bad}")
    logger.info(f"[{name}] shape={df.shape}  label dist: {df['label'].value_counts().to_dict()}")
    logger.info(f"[{name}] forgery families: {df['forgery_family'].value_counts().to_dict()}")
    if "split" in df.columns:
        logger.info(f"[{name}] split dist: {df['split'].value_counts().to_dict()}")


def main():
    set_seeds(42)
    cfg = load_config(str(CONFIG_PATH))

    out_dir = ROOT / cfg["paths"]["output_root"]
    pred_dir = out_dir / "predictions"
    log_dir = out_dir / "logs"
    pred_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("exp15.prepare", str(log_dir / "run.log"))
    logger.info("=== Stage 01: Prepare Features ===")

    emo_cols = cfg["emotion_feature_cols"]
    qual_cols = cfg["quality_feature_cols"]

    df = build_feature_matrix(
        face_manifest_path=require_file(ROOT / cfg["paths"]["face_manifest"], "face manifest"),
        emotion_path=require_file(ROOT / cfg["paths"]["emotion_features"], "emotion features"),
        ucf_scores_path=require_file(ROOT / cfg["paths"]["ucf_scores"], "UCF scores"),
        ucf_frame_scores_path=require_file(ROOT / cfg["paths"]["ucf_frame_scores"], "UCF frame scores"),
        emo_feature_cols=emo_cols,
        quality_feature_cols=qual_cols,
        logger=logger,
    )

    feature_cols = ["detector_score"] + emo_cols + qual_cols

    if cfg.get("use_predefined_split", True):
        trainval = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
        test = df[df["split"] == "test"].reset_index(drop=True)

        validate_matrix(trainval, "trainval", feature_cols, logger)
        validate_matrix(test, "test", feature_cols, logger)

        trainval.to_parquet(pred_dir / "trainval_feature_matrix.parquet", index=False)
        test.to_parquet(pred_dir / "test_feature_matrix.parquet", index=False)
        logger.info(f"Saved trainval ({len(trainval)}) and test ({len(test)}) feature matrices")

        # Verify identity disjointness
        tv_ids = set(trainval["identity"].dropna())
        te_ids = set(test["identity"].dropna())
        overlap = tv_ids & te_ids
        if overlap:
            logger.warning(f"Identity overlap between trainval and test: {overlap}")
        else:
            logger.info("Identity disjoint check PASSED — no overlap between trainval and test")
    else:
        validate_matrix(df, "full", feature_cols, logger)
        df.to_parquet(pred_dir / "trainval_feature_matrix.parquet", index=False)
        logger.info(f"Saved full feature matrix ({len(df)} videos) as trainval")

    print("\n=== Feature Matrix Summary ===")
    if cfg.get("use_predefined_split", True):
        print(f"TrainVal: {len(trainval)} videos  |  Test holdout: {len(test)} videos")
        print(f"Test identity-disjoint from trainval: {len(tv_ids & te_ids) == 0}")
    else:
        print(f"Full (no holdout): {len(df)} videos")


if __name__ == "__main__":
    main()
