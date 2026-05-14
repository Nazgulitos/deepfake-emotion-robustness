"""PR3 / Exp.01 — HuggingFace deepfake detector inference.

Model: dima806/deepfake_vs_real_image_detection
       (ViT-based image classifier, label 0=real, 1=fake)

Reads:  datasets/metadata/{subset}_face_manifest.csv
Writes: datasets/detector_processed/{subset}_huggingface_scores.csv
        datasets/detector_processed/{subset}_huggingface_frame_scores.csv

Usage:
    python scripts/run_huggingface_detector.py --subset final
    python scripts/run_huggingface_detector.py --subset pilot
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.io import write_table
from src.utils.logging_utils import setup_logging
from src.utils.run_metadata import now_utc

HF_MODEL_ID = "dima806/deepfake_vs_real_image_detection"
BATCH_SIZE = 32
FAKE_LABEL = "fake"   # label in the model's id2label that means deepfake


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", required=True, choices=["final", "pilot"])
    p.add_argument("--face_manifest", type=Path, default=None,
                   help="Override path to face manifest CSV.")
    p.add_argument("--out_dir", type=Path,
                   default=Path("datasets/detector_processed"))
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--device", default=None,
                   help="'cuda', 'cpu', or None for auto-detect.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _get_device(requested: str | None) -> str:
    import torch
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(device: str):
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification
    extractor = AutoFeatureExtractor.from_pretrained(HF_MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
    model.to(device).eval()
    return extractor, model


def _infer_batch(images: list, extractor, model, device: str,
                 fake_idx: int) -> list[float]:
    """Return per-image fake probabilities."""
    import torch
    inputs = extractor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs[:, fake_idx].tolist()


def _load_image(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


def main() -> None:
    args = parse_args()
    start_time = now_utc()

    log_path = Path("outputs/logs") / f"run_huggingface_{args.subset}.log"
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger("run_huggingface_detector")

    # Resolve manifest path
    manifest_path = args.face_manifest or Path(
        f"datasets/metadata/{args.subset}_face_manifest.csv")
    if not manifest_path.exists():
        logger.error("Face manifest not found: %s", manifest_path)
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    logger.info("Loaded manifest: %d rows subset=%s", len(manifest), args.subset)

    # Import heavy deps here so the script fails fast if they're missing
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification
    except ImportError:
        logger.error("Missing deps. Run: pip install transformers pillow torch torchvision")
        sys.exit(1)

    device = _get_device(args.device)
    logger.info("Loading model %s on device=%s", HF_MODEL_ID, device)
    extractor, model = _load_model(device)

    # Determine fake label index from model config
    id2label = model.config.id2label
    label2id = {v.lower(): k for k, v in id2label.items()}
    fake_idx = label2id.get(FAKE_LABEL, label2id.get("deepfake", 1))
    logger.info("Model labels: %s  fake_idx=%d", id2label, fake_idx)

    # Frame-level inference
    frame_scores: list[dict] = []
    rows = manifest.to_dict(orient="records")
    batch_images, batch_meta = [], []

    def _flush(batch_images, batch_meta):
        if not batch_images:
            return []
        scores = _infer_batch(batch_images, extractor, model, device, fake_idx)
        return [{"score": s, **m} for s, m in zip(scores, batch_meta)]

    for i, row in enumerate(rows):
        face_path = row.get("face_path", "")
        try:
            img = _load_image(face_path)
            batch_images.append(img)
            batch_meta.append(row)
        except Exception as exc:
            logger.warning("Could not load image %s: %s", face_path, exc)
            continue

        if len(batch_images) >= args.batch_size:
            frame_scores.extend(_flush(batch_images, batch_meta))
            batch_images, batch_meta = [], []
            if (i + 1) % 500 == 0:
                logger.info("  %d / %d faces processed", i + 1, len(rows))

    frame_scores.extend(_flush(batch_images, batch_meta))

    frame_df = pd.DataFrame(frame_scores)
    frame_df = frame_df.rename(columns={"score": "detector_score"})
    frame_df["detector_pred"] = (frame_df["detector_score"] >= 0.5).astype(int)
    logger.info("Frame-level inference done: %d rows", len(frame_df))

    # Aggregate to video level (mean pooling)
    group_cols = ["video_id", "label", "split", "manipulation_family",
                  "manipulation_type"]
    group_cols = [c for c in group_cols if c in frame_df.columns]
    video_df = (
        frame_df.groupby(group_cols, dropna=False)
        .agg(
            n_face_frames=("detector_score", "count"),
            detector_score=("detector_score", "mean"),
        )
        .reset_index()
    )
    video_df["video_score_mode"] = "mean"
    video_df["detector_pred"] = (video_df["detector_score"] >= 0.5).astype(int)

    # Save
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_out = args.out_dir / f"{args.subset}_huggingface_frame_scores.csv"
    video_out = args.out_dir / f"{args.subset}_huggingface_scores.csv"

    write_table(frame_df, frame_out)
    write_table(video_df, video_out)
    logger.info("Saved frame scores → %s", frame_out)
    logger.info("Saved video scores → %s  (%d videos)", video_out, len(video_df))

    end_time = now_utc()
    elapsed = (end_time - start_time).total_seconds()
    logger.info("Done in %.1f s", elapsed)


if __name__ == "__main__":
    main()
