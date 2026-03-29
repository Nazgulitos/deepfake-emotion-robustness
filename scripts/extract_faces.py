"""Detect and crop faces from extracted frames."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.face_extractor import extract_faces_from_frames
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

    log_path = Path(get_config_value(config, ["paths", "logs_dir"], "outputs/logs")) / "extract_faces.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger("extract_faces")

    seed = int(get_config_value(config, ["project", "seed"], 42))
    run_id = build_run_id("extract_faces", seed)
    logger.info("Starting stage run_id=%s", run_id)

    frame_path = Path(get_config_value(config, ["files", "frame_manifest"], "metadata/frame_manifest.csv"))
    face_path = Path(get_config_value(config, ["files", "face_manifest"], "metadata/face_manifest.csv"))
    face_output_dir = Path(get_config_value(config, ["preprocessing", "face_output_dir"], "outputs/faces"))
    detector_name = str(get_config_value(config, ["preprocessing", "face_detector"], "mediapipe"))

    frame_df = read_table(frame_path)
    face_df = extract_faces_from_frames(frame_df, face_output_dir=face_output_dir, detector_name=detector_name)
    write_table(face_df, face_path)
    logger.info("Saved face manifest rows=%d to %s", len(face_df), face_path)


if __name__ == "__main__":
    main()
