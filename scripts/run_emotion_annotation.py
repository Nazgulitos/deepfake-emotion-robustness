"""Run frame-level emotion inference on face crops."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.emotion.annotator import annotate_faces
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

    log_path = Path(get_config_value(config, ["paths", "logs_dir"], "outputs/logs")) / "run_emotion_annotation.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger("run_emotion_annotation")

    seed = int(get_config_value(config, ["project", "seed"], 42))
    run_id = build_run_id("run_emotion_annotation", seed)
    logger.info("Starting stage run_id=%s", run_id)

    face_path = Path(get_config_value(config, ["files", "face_manifest"], "metadata/face_manifest.csv"))
    emotion_path = Path(
        get_config_value(config, ["files", "emotion_frame_predictions"], "metadata/emotion_frame_predictions.csv")
    )
    model_name = str(get_config_value(config, ["emotion", "model_name"], "placeholder_fer_model"))

    face_df = read_table(face_path)
    emotion_df = annotate_faces(face_df, model_name=model_name)
    write_table(emotion_df, emotion_path)
    logger.info("Saved emotion frame predictions rows=%d to %s", len(emotion_df), emotion_path)


if __name__ == "__main__":
    main()
