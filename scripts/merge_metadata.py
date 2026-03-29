"""Merge subset, emotion features, and detector outputs into one table."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.merge import merge_on_video_id
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

    log_path = Path(get_config_value(config, ["paths", "logs_dir"], "outputs/logs")) / "merge_metadata.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger("merge_metadata")

    seed = int(get_config_value(config, ["project", "seed"], 42))
    run_id = build_run_id("merge_metadata", seed)
    logger.info("Starting stage run_id=%s", run_id)

    subset_path = Path(get_config_value(config, ["files", "subset_manifest"], "metadata/subset_manifest.csv"))
    emotion_path = Path(get_config_value(config, ["files", "video_emotion_features"], "metadata/video_emotion_features.csv"))
    detector_path = Path(get_config_value(config, ["files", "detector_scores"], "metadata/detector_scores.csv"))
    output_path = Path(get_config_value(config, ["files", "final_merged_table"], "metadata/final_merged_table.csv"))

    subset_df = read_table(subset_path)
    emotion_df = read_table(emotion_path)
    detector_df = read_table(detector_path)

    merged_df = merge_on_video_id([subset_df, emotion_df, detector_df])
    write_table(merged_df, output_path)
    logger.info("Saved merged metadata rows=%d to %s", len(merged_df), output_path)


if __name__ == "__main__":
    main()
