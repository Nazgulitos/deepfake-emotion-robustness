"""Exp.08 — UCF detector (DeepfakeBench) inference.

UCF uses an XceptionNet backbone with uncertainty-aware contrastive learning.
We run DeepfakeBench's test.py inside its own venv (to avoid numpy ABI
conflicts) and then parse its output CSV into our standard schema.

Prerequisites on spectrum:
  1. DeepfakeBench cloned at ./DeepfakeBench/
  2. Weights at ./DeepfakeBench/training/weights/ucf_best.pth
  3. DeepfakeBench venv at ./DeepfakeBench/.venv/
  4. Face crops already extracted (paths in face manifest)

Writes:
  datasets/detector_processed/{subset}_ucf_scores.csv
  outputs/logs/run_ucf_{subset}.log
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.logging_utils import setup_logging
from src.utils.run_metadata import now_utc

DEEPFAKEBENCH_DIR = ROOT / "DeepfakeBench"
UCF_WEIGHTS = DEEPFAKEBENCH_DIR / "training/weights/ucf_best.pth"
# Python inside DeepfakeBench's own venv — has matching numpy/torch
DFB_PYTHON = DEEPFAKEBENCH_DIR / ".venv/bin/python"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", required=True, choices=["final", "pilot"])
    p.add_argument("--face_manifest", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, default=Path("datasets/detector_processed"))
    p.add_argument("--weights", type=Path, default=UCF_WEIGHTS)
    p.add_argument("--dfb_dir", type=Path, default=DEEPFAKEBENCH_DIR)
    p.add_argument("--dfb_python", type=Path, default=DFB_PYTHON,
                   help="Python interpreter inside DeepfakeBench venv.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _check_prerequisites(args: argparse.Namespace, logger: logging.Logger) -> bool:
    ok = True
    if not args.dfb_dir.exists():
        logger.error("DeepfakeBench not found: %s", args.dfb_dir)
        ok = False
    if not args.weights.exists():
        logger.error("UCF weights not found: %s", args.weights)
        ok = False
    if not args.dfb_python.exists():
        logger.error(
            "DeepfakeBench Python not found: %s\n"
            "  → Run inside DeepfakeBench dir:  python -m venv .venv && "
            ".venv/bin/pip install -r requirements.txt",
            args.dfb_python,
        )
        ok = False
    return ok


def _write_inference_script(manifest: pd.DataFrame, weights: Path,
                             dfb_dir: Path, device: str,
                             out_csv: Path) -> Path:
    """Write a self-contained Python script that DeepfakeBench's venv will run."""
    # Build a simple list of (face_path, video_id, label) from the manifest
    records = []
    for _, row in manifest.iterrows():
        records.append({
            "face_path": str(row.get("face_path", "")),
            "video_id": str(row.get("video_id", "")),
            "label": str(row.get("label", "")),
            "split": str(row.get("split", "")),
            "manipulation_family": str(row.get("manipulation_family", "")),
        })

    script = f"""
import sys, json, csv
from pathlib import Path

sys.path.insert(0, {str(dfb_dir / 'training')!r})

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# ---- load UCF model via DeepfakeBench registry ----
import yaml
from detectors import DETECTOR

config_candidates = [
    {str(dfb_dir / 'training/config/detector/ucf.yaml')!r},
]
config_path = None
for c in config_candidates:
    if Path(c).exists():
        config_path = c
        break

if config_path is None:
    # list available configs for debugging
    cfg_dir = Path({str(dfb_dir / 'training/config/detector')!r})
    available = list(cfg_dir.glob('*.yaml')) if cfg_dir.exists() else []
    raise FileNotFoundError(f"ucf.yaml not found. Available: {{available}}")

with open(config_path) as f:
    config = yaml.safe_load(f)

config['weights_path'] = {str(weights)!r}
device = {device!r}

model_name = config['model_name']
model = DETECTOR[model_name](config)

state = torch.load({str(weights)!r}, map_location=device)
if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']
# strip DataParallel prefix
state = {{k.replace('module.', ''): v for k, v in state.items()}}
model.load_state_dict(state, strict=False)
model.to(device).eval()
print(f"Model {{model_name}} loaded on {{device}}", flush=True)

# ---- transform ----
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

records = {json.dumps(records)}

results = []
batch_size = 32
batch_imgs, batch_meta = [], []

def flush(batch_imgs, batch_meta):
    if not batch_imgs:
        return []
    tensors = torch.stack(batch_imgs).to(device)
    with torch.no_grad():
        out = model({{'image': tensors}}, inference=True)
        if isinstance(out, dict):
            logits = out.get('cls', out.get('logits', list(out.values())[0]))
        else:
            logits = out
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return [(float(probs[i, 1]), m) for i, m in enumerate(batch_meta)]

for i, rec in enumerate(records):
    try:
        img = Image.open(rec['face_path']).convert('RGB')
        batch_imgs.append(transform(img))
        batch_meta.append(rec)
    except Exception as e:
        print(f"SKIP {{rec['face_path']}}: {{e}}", flush=True)
        continue
    if len(batch_imgs) >= batch_size:
        results.extend(flush(batch_imgs, batch_meta))
        batch_imgs, batch_meta = [], []
        if (i+1) % 500 == 0:
            print(f"  {{i+1}} / {{len(records)}} processed", flush=True)

results.extend(flush(batch_imgs, batch_meta))

with open({str(out_csv)!r}, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['video_id','label','split',
                                           'manipulation_family','detector_score'])
    writer.writeheader()
    for score, meta in results:
        writer.writerow({{
            'video_id': meta['video_id'],
            'label': meta['label'],
            'split': meta['split'],
            'manipulation_family': meta['manipulation_family'],
            'detector_score': score,
        }})

print(f"Wrote {{len(results)}} frame scores to {str(out_csv)!r}", flush=True)
"""
    tmp = Path(tempfile.mktemp(suffix="_ucf_infer.py"))
    tmp.write_text(script)
    return tmp


def main() -> None:
    args = parse_args()
    start_time = now_utc()

    log_path = Path("outputs/logs") / f"run_ucf_{args.subset}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger("run_ucf_detector")

    if not _check_prerequisites(args, logger):
        sys.exit(1)

    manifest_path = args.face_manifest or Path(
        f"datasets/metadata/{args.subset}_face_manifest.csv")
    if not manifest_path.exists():
        logger.error("Face manifest not found: %s", manifest_path)
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    logger.info("Loaded manifest: %d rows subset=%s", len(manifest), args.subset)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_csv = args.out_dir / f"{args.subset}_ucf_frame_scores.csv"

    # Write and run the inference script inside DeepfakeBench's venv
    infer_script = _write_inference_script(
        manifest, args.weights, args.dfb_dir, args.device, frame_csv)
    logger.info("Running inference via %s", args.dfb_python)

    result = subprocess.run(
        [str(args.dfb_python), str(infer_script)],
        capture_output=False,
        cwd=str(args.dfb_dir),
    )
    infer_script.unlink(missing_ok=True)

    if result.returncode != 0:
        logger.error("Inference script exited with code %d", result.returncode)
        sys.exit(result.returncode)

    if not frame_csv.exists():
        logger.error("Expected output not found: %s", frame_csv)
        sys.exit(1)

    frame_df = pd.read_csv(frame_csv)
    logger.info("Frame scores: %d rows", len(frame_df))

    # Aggregate to video level (mean score per video)
    id_cols = [c for c in ["video_id", "label", "split", "manipulation_family"]
               if c in frame_df.columns]
    video_df = (frame_df.groupby(id_cols, dropna=False)
                .agg(detector_score=("detector_score", "mean"),
                     n_frames=("detector_score", "count"))
                .reset_index())
    video_df["detector_pred"] = (video_df["detector_score"] >= 0.5).astype(int)
    video_df["y"] = (video_df["label"].astype(str)
                     .map({"fake": 1, "real": 0}).fillna(0).astype(int))
    video_df["video_score_mode"] = "ucf_mean"

    video_path = args.out_dir / f"{args.subset}_ucf_scores.csv"
    video_df.to_csv(video_path, index=False)
    logger.info("Saved video scores → %s  (%d videos)", video_path, len(video_df))

    # Quick AUC
    try:
        from sklearn.metrics import roc_auc_score
        valid = video_df.dropna(subset=["detector_score", "y"])
        if valid["y"].nunique() == 2:
            auc = roc_auc_score(valid["y"], valid["detector_score"])
            logger.info("Video-level AUC: %.4f  (n=%d)", auc, len(valid))
    except Exception as e:
        logger.warning("Could not compute AUC: %s", e)

    end_time = now_utc()
    logger.info("Done in %.1f s", (end_time - start_time).total_seconds())


if __name__ == "__main__":
    main()
