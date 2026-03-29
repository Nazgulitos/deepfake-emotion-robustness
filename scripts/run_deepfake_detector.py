"""Run baseline deepfake detector and save scores."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.baseline_detector import run_baseline_detector
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

    log_path = Path(get_config_value(config, ["paths", "logs_dir"], "outputs/logs")) / "run_deepfake_detector.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger("run_deepfake_detector")

    seed = int(get_config_value(config, ["project", "seed"], 42))
    run_id = build_run_id("run_deepfake_detector", seed)
    logger.info("Starting stage run_id=%s", run_id)

    subset_path = Path(get_config_value(config, ["files", "subset_manifest"], "metadata/subset_manifest.csv"))
    detector_path = Path(get_config_value(config, ["files", "detector_scores"], "metadata/detector_scores.csv"))
    model_name = str(get_config_value(config, ["detection", "model_name"], "placeholder_detector"))
    threshold = float(get_config_value(config, ["detection", "threshold"], 0.5))

    subset_df = read_table(subset_path)
    detector_df = run_baseline_detector(subset_df, model_name=model_name, threshold=threshold)
    write_table(detector_df, detector_path)
    logger.info("Saved detector scores rows=%d to %s", len(detector_df), detector_path)


if __name__ == "__main__":
    main()
