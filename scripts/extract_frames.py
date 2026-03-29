"""Extract frames from curated subset videos."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.frame_extractor import extract_frames_from_manifest
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

    log_path = Path(get_config_value(config, ["paths", "logs_dir"], "outputs/logs")) / "extract_frames.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger("extract_frames")

    seed = int(get_config_value(config, ["project", "seed"], 42))
    run_id = build_run_id("extract_frames", seed)
    logger.info("Starting stage run_id=%s", run_id)

    subset_path = Path(get_config_value(config, ["files", "subset_manifest"], "metadata/subset_manifest.csv"))
    frame_path = Path(get_config_value(config, ["files", "frame_manifest"], "metadata/frame_manifest.csv"))
    fps = float(get_config_value(config, ["preprocessing", "frame_fps"], 2.0))
    frame_output_dir = Path(get_config_value(config, ["preprocessing", "frame_output_dir"], "outputs/frames"))

    subset_df = read_table(subset_path)
    frame_df = extract_frames_from_manifest(subset_df, frame_output_dir=frame_output_dir, fps=fps)
    write_table(frame_df, frame_path)
    logger.info("Saved frame manifest rows=%d to %s", len(frame_df), frame_path)


if __name__ == "__main__":
    main()
