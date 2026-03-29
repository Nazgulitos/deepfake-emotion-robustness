"""Evaluate deepfake detector performance overall and by emotion groups."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_binary_metrics, stratify_by_column
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

    log_path = Path(get_config_value(config, ["paths", "logs_dir"], "outputs/logs")) / "evaluate_by_emotion.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger("evaluate_by_emotion")

    seed = int(get_config_value(config, ["project", "seed"], 42))
    run_id = build_run_id("evaluate_by_emotion", seed)
    logger.info("Starting stage run_id=%s", run_id)

    merged_path = Path(get_config_value(config, ["files", "final_merged_table"], "metadata/final_merged_table.csv"))
    output_path = Path(get_config_value(config, ["files", "evaluation_summary"], "outputs/tables/evaluation_summary.csv"))

    label_col = str(get_config_value(config, ["evaluation", "label_column"], "is_fake"))
    score_col = str(get_config_value(config, ["evaluation", "score_column"], "detector_score"))
    emotion_col = str(get_config_value(config, ["evaluation", "emotion_column"], "dominant_emotion"))

    table = read_table(merged_path)

    overall = compute_binary_metrics(table[label_col], table[score_col])
    overall_row = pd.DataFrame([{"group_column": "overall", "group_value": "all", "n": len(table), **overall}])

    by_emotion = stratify_by_column(table, label_column=label_col, score_column=score_col, group_column=emotion_col)
    result = pd.concat([overall_row, by_emotion], ignore_index=True)

    write_table(result, output_path)
    logger.info("Saved evaluation summary rows=%d to %s", len(result), output_path)


if __name__ == "__main__":
    main()
