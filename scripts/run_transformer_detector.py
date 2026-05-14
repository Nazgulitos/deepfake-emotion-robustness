"""PR8 / Exp.08 — UCF (DeepfakeBench) transformer detector inference.

Uses the UCF model from DeepfakeBench v1.0.1.
UCF uses an Xception backbone with uncertainty-aware contrastive learning.

Prerequisites:
  1. Clone DeepfakeBench into ./DeepfakeBench/
  2. Download UCF weights to ./DeepfakeBench/training/weights/ucf_best.pth
     (from the DeepfakeBench v1.0.1 GitHub release assets)
  3. Face crops already extracted at paths in face_manifest — no re-extraction needed.

Usage:
    python scripts/run_transformer_detector.py --subset final

This script wraps the DeepfakeBench test runner and then converts its output
into the standard {subset}_transformer_scores.csv schema.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.io import write_table
from src.utils.logging_utils import setup_logging
from src.utils.run_metadata import now_utc

DEEPFAKEBENCH_DIR = ROOT / "DeepfakeBench"
UCF_CONFIG = DEEPFAKEBENCH_DIR / "training/config/detector/ucf.yaml"
UCF_WEIGHTS = DEEPFAKEBENCH_DIR / "training/weights/ucf_best.pth"
# DeepfakeBench writes results to this path by default
DFB_RESULTS_DIR = DEEPFAKEBENCH_DIR / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", required=True, choices=["final", "pilot"])
    p.add_argument("--face_manifest", type=Path, default=None)
    p.add_argument("--out_dir", type=Path,
                   default=Path("datasets/detector_processed"))
    p.add_argument("--dfb_dir", type=Path, default=DEEPFAKEBENCH_DIR,
                   help="Path to DeepfakeBench repo root.")
    p.add_argument("--weights", type=Path, default=UCF_WEIGHTS)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _check_prerequisites(args: argparse.Namespace, logger: logging.Logger) -> bool:
    ok = True
    if not args.dfb_dir.exists():
        logger.error("DeepfakeBench directory not found: %s\n"
                     "  → git clone https://github.com/SCLBD/DeepfakeBench.git", args.dfb_dir)
        ok = False
    if not args.weights.exists():
        logger.error("UCF weights not found: %s\n"
                     "  → Download ucf_best.pth from DeepfakeBench v1.0.1 release:\n"
                     "    https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1", args.weights)
        ok = False
    config = args.dfb_dir / "training/config/detector/ucf.yaml"
    if not config.exists():
        logger.error("UCF config not found: %s", config)
        ok = False
    return ok


def _run_deepfakebench(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Call DeepfakeBench test.py and return its exit code."""
    cmd = [
        sys.executable,
        str(args.dfb_dir / "training/test.py"),
        "--detector_path", str(args.dfb_dir / "training/config/detector/ucf.yaml"),
        "--test_dataset", "Celeb-DF-v2",
        "--weights_path", str(args.weights),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(args.dfb_dir), capture_output=False)
    return result.returncode


def _find_dfb_output(dfb_dir: Path, logger: logging.Logger) -> Path | None:
    """Locate the most recent DeepfakeBench result CSV/JSON under results/."""
    results_dir = dfb_dir / "results"
    candidates = sorted(results_dir.rglob("*.csv"), key=lambda p: p.stat().st_mtime,
                         reverse=True) if results_dir.exists() else []
    if not candidates:
        candidates = sorted(results_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime,
                             reverse=True) if results_dir.exists() else []
    if candidates:
        logger.info("Found DeepfakeBench output: %s", candidates[0])
        return candidates[0]
    logger.warning("No DeepfakeBench result files found in %s", results_dir)
    return None


def _parse_dfb_output(result_path: Path, manifest: pd.DataFrame,
                      logger: logging.Logger) -> pd.DataFrame:
    """Convert DeepfakeBench output to our standard schema.

    DeepfakeBench output format varies by version. This handles:
      - CSV with columns: video_name / video_path, score / prob, label
      - JSON with per-video predictions

    Falls back to reading raw file and mapping video_id by filename stem.
    """
    suffix = result_path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(result_path)
    elif suffix == ".json":
        with open(result_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            raw = pd.DataFrame(data)
        else:
            raw = pd.DataFrame([data])
    else:
        logger.error("Unknown result format: %s", result_path)
        return pd.DataFrame()

    logger.info("Raw DFB output columns: %s", list(raw.columns))

    # Normalise score column
    for score_col in ["score", "prob", "fake_prob", "deepfake_prob"]:
        if score_col in raw.columns:
            raw = raw.rename(columns={score_col: "detector_score"})
            break

    # Normalise video ID column
    for vid_col in ["video_name", "video_path", "video_id", "filename"]:
        if vid_col in raw.columns:
            raw["_vid_stem"] = raw[vid_col].apply(
                lambda x: Path(str(x)).stem if pd.notna(x) else "")
            break

    # Map to our video_id via filename stem
    manifest["_vid_stem"] = manifest["video_id"].apply(lambda x: Path(str(x)).stem)
    merged = manifest[["video_id", "label", "split", "manipulation_family",
                        "manipulation_type", "_vid_stem"]].drop_duplicates("video_id")
    merged = merged.merge(
        raw[["_vid_stem", "detector_score"]].drop_duplicates("_vid_stem"),
        on="_vid_stem", how="left",
    )
    merged = merged.drop(columns=["_vid_stem"])
    logger.info("Merged %d / %d videos with DFB scores",
                merged["detector_score"].notna().sum(), len(merged))
    return merged


def main() -> None:
    args = parse_args()
    start_time = now_utc()

    log_path = Path("outputs/logs") / f"run_transformer_{args.subset}.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger("run_transformer_detector")

    if not _check_prerequisites(args, logger):
        logger.error("Prerequisites not met — aborting.")
        sys.exit(1)

    manifest_path = args.face_manifest or Path(
        f"datasets/metadata/{args.subset}_face_manifest.csv")
    if not manifest_path.exists():
        logger.error("Face manifest not found: %s", manifest_path)
        sys.exit(1)
    manifest = pd.read_csv(manifest_path)

    # Run DeepfakeBench
    rc = _run_deepfakebench(args, logger)
    if rc != 0:
        logger.error("DeepfakeBench exited with code %d", rc)
        sys.exit(rc)

    # Parse and convert output
    result_path = _find_dfb_output(args.dfb_dir, logger)
    if result_path is None:
        logger.error("Cannot find DeepfakeBench output. Check %s/results/", args.dfb_dir)
        sys.exit(1)

    video_df = _parse_dfb_output(result_path, manifest, logger)
    if video_df.empty:
        logger.error("No scores parsed — check DeepfakeBench output format.")
        sys.exit(1)

    video_df["video_score_mode"] = "ucf_output"
    video_df["detector_pred"] = (video_df["detector_score"] >= 0.5).astype(int)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.subset}_transformer_scores.csv"
    write_table(video_df, out_path)
    logger.info("Saved → %s  (%d videos)", out_path, len(video_df))

    end_time = now_utc()
    logger.info("Done in %.1f s", (end_time - start_time).total_seconds())


if __name__ == "__main__":
    main()
