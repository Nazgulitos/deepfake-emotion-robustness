"""Run a simple late-fusion baseline using detector + emotion features."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.fusion import run_late_fusion
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

    log_path = Path(get_config_value(config, ["paths", "logs_dir"], "outputs/logs")) / "run_late_fusion.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger("run_late_fusion")

    seed = int(get_config_value(config, ["project", "seed"], 42))
    run_id = build_run_id("run_late_fusion", seed)
    logger.info("Starting stage run_id=%s", run_id)

    merged_path = Path(get_config_value(config, ["files", "final_merged_table"], "metadata/final_merged_table.csv"))
    output_path = Path(get_config_value(config, ["files", "late_fusion_scores"], "outputs/tables/late_fusion_scores.csv"))

    feature_columns = list(get_config_value(config, ["fusion", "feature_columns"], ["detector_score"]))
    target_column = str(get_config_value(config, ["fusion", "target_column"], "is_fake"))

    table = read_table(merged_path)
    scored_table, _ = run_late_fusion(table, feature_columns=feature_columns, target_column=target_column)
    write_table(scored_table, output_path)
    logger.info("Saved late-fusion scores rows=%d to %s", len(scored_table), output_path)


if __name__ == "__main__":
    main()
