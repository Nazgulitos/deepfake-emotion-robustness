"""Aggregate frame-level emotions to video-level descriptors."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.emotion.aggregation import aggregate_emotion_features
from src.utils.config import get_config_value, load_yaml_config
from src.utils.io import read_table, write_table
from src.utils.logging_utils import setup_logging
from src.utils.naming import build_run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    log_path = Path(get_config_value(config, ["paths", "logs_dir"], "outputs/logs")) / "aggregate_emotion_features.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger("aggregate_emotion_features")

    seed = int(get_config_value(config, ["project", "seed"], 42))
    run_id = build_run_id("aggregate_emotion_features", seed)
    logger.info("Starting stage run_id=%s", run_id)

    frame_pred_path = Path(
        get_config_value(config, ["files", "emotion_frame_predictions"], "metadata/emotion_frame_predictions.csv")
    )
    features_path = Path(get_config_value(config, ["files", "video_emotion_features"], "metadata/video_emotion_features.csv"))
    neutral_label = str(get_config_value(config, ["emotion", "neutral_label"], "neutral"))

    frame_pred_df = read_table(frame_pred_path)
    features_df = aggregate_emotion_features(frame_pred_df, neutral_label=neutral_label)
    write_table(features_df, features_path)
    logger.info("Saved video-level emotion features rows=%d to %s", len(features_df), features_path)


if __name__ == "__main__":
    main()
