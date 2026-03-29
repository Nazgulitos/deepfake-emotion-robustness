"""Build curated subset manifest from configured dataset sources."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.subset_builder import build_subset_manifest
from src.utils.config import get_config_value, load_yaml_config
from src.utils.io import write_table
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

    log_path = Path(get_config_value(config, ["paths", "logs_dir"], "outputs/logs")) / "build_subset.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger("build_subset")

    seed = int(get_config_value(config, ["project", "seed"], 42))
    run_id = build_run_id("build_subset", seed)
    logger.info("Starting stage run_id=%s", run_id)

    sources = list(get_config_value(config, ["datasets", "sources"], []))
    max_videos = int(get_config_value(config, ["subset", "max_videos_per_source"], 200))

    subset_df = build_subset_manifest(sources=sources, max_videos_per_source=max_videos, seed=seed)

    output_path = Path(get_config_value(config, ["files", "subset_manifest"], "metadata/subset_manifest.csv"))
    write_table(subset_df, output_path)
    logger.info("Saved subset manifest rows=%d to %s", len(subset_df), output_path)


if __name__ == "__main__":
    main()
