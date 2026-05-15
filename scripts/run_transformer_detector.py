"""Exp.08 — XceptionNet (DeepfakeBench) detector inference.

Runs inference using the DeepfakeBench XceptionNet model
(xception_best.pth) directly on face crops from the face manifest,
without going through DeepfakeBench's test.py dataset pipeline.

Prerequisites on spectrum:
  1. DeepfakeBench cloned at ./DeepfakeBench/
  2. Weights at ./DeepfakeBench/training/weights/xception_best.pth
  3. Face crops already extracted (paths in face manifest)

Writes:
  datasets/detector_processed/{subset}_xception_dfb_scores.csv
  datasets/detector_processed/{subset}_xception_dfb_frame_scores.csv
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

from src.utils.logging_utils import setup_logging
from src.utils.run_metadata import now_utc

DEEPFAKEBENCH_DIR = ROOT / "DeepfakeBench"
XCEPTION_WEIGHTS = DEEPFAKEBENCH_DIR / "training/weights/xception_best.pth"
BATCH_SIZE = 32
IMG_SIZE = 299   # XceptionNet input size


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", required=True, choices=["final", "pilot"])
    p.add_argument("--face_manifest", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, default=Path("datasets/detector_processed"))
    p.add_argument("--weights", type=Path, default=XCEPTION_WEIGHTS)
    p.add_argument("--dfb_dir", type=Path, default=DEEPFAKEBENCH_DIR)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _get_device(requested: str) -> str:
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        logging.getLogger("run_xception_dfb").warning(
            "CUDA not available — falling back to CPU")
        return "cpu"
    return requested


def _load_model(weights_path: Path, dfb_dir: Path, device: str):
    """Load XceptionNet from DeepfakeBench using its own model registry."""
    import torch

    # Add DeepfakeBench training dir to path so its imports work
    training_dir = str(dfb_dir / "training")
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)

    try:
        # DeepfakeBench model loader (works with v1.0.x)
        from detectors import DETECTOR
        import yaml

        # Try several common config filenames
        for cfg_name in ["xception.yaml", "xception_df.yaml", "xception_c23.yaml"]:
            config_path = dfb_dir / "training/config/detector" / cfg_name
            if config_path.exists():
                break
        else:
            raise FileNotFoundError(
                f"No xception config found in {dfb_dir}/training/config/detector/. "
                f"Available: {list((dfb_dir / 'training/config/detector').glob('*.yaml'))}"
            )

        with open(config_path) as f:
            config = yaml.safe_load(f)

        config["pretrained"] = str(weights_path)
        config["device"] = device

        model = DETECTOR[config["model_name"]](config)
        state = torch.load(weights_path, map_location=device)
        # DFB checkpoints may be wrapped under 'state_dict' key
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        logging.getLogger("run_xception_dfb").info(
            "Loaded XceptionNet via DeepfakeBench registry from %s", weights_path)
        return model, "dfb"

    except Exception as e:
        logging.getLogger("run_xception_dfb").warning(
            "DeepfakeBench loader failed (%s) — trying timm/direct load", e)

    # Fallback: load XceptionNet directly via timm
    try:
        import timm
        import torch

        model = timm.create_model("xception", pretrained=False, num_classes=2)
        state = torch.load(weights_path, map_location=device)
        state_dict = state.get("state_dict", state)
        # Strip any 'module.' prefix from DataParallel wrapping
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        logging.getLogger("run_xception_dfb").info(
            "Loaded XceptionNet via timm from %s", weights_path)
        return model, "timm"

    except Exception as e:
        raise RuntimeError(
            f"Could not load XceptionNet weights from {weights_path}.\n"
            f"Tried DeepfakeBench registry and timm. Last error: {e}\n"
            f"Make sure timm is installed: pip install timm"
        ) from e


def _build_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _infer_batch(images: list, model, device: str, loader_type: str) -> list[float]:
    """Return per-image fake probabilities."""
    import torch
    transform = _build_transform()

    tensors = torch.stack([transform(img) for img in images]).to(device)
    with torch.no_grad():
        if loader_type == "dfb":
            # DFB models expect a dict input
            out = model({"image": tensors}, inference=True)
            # Output may be dict with 'cls' key or a tensor
            if isinstance(out, dict):
                logits = out.get("cls", out.get("logits", list(out.values())[0]))
            else:
                logits = out
        else:
            logits = model(tensors)

        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Index 1 = fake class (standard DFB convention: 0=real, 1=fake)
    return probs[:, 1].tolist()


def _load_image(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


def main() -> None:
    args = parse_args()
    start_time = now_utc()

    log_path = Path("outputs/logs") / f"run_xception_dfb_{args.subset}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(level=args.log_level, log_file=log_path)
    logger = logging.getLogger("run_xception_dfb")

    # Prerequisites
    if not args.weights.exists():
        logger.error("Weights not found: %s", args.weights)
        logger.error("Expected: DeepfakeBench/training/weights/xception_best.pth")
        sys.exit(1)
    if not args.dfb_dir.exists():
        logger.error("DeepfakeBench directory not found: %s", args.dfb_dir)
        sys.exit(1)

    manifest_path = args.face_manifest or Path(
        f"datasets/metadata/{args.subset}_face_manifest.csv")
    if not manifest_path.exists():
        logger.error("Face manifest not found: %s", manifest_path)
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    logger.info("Loaded manifest: %d rows subset=%s", len(manifest), args.subset)

    device = _get_device(args.device)
    logger.info("Loading XceptionNet weights from %s on %s", args.weights, device)
    model, loader_type = _load_model(args.weights, args.dfb_dir, device)
    logger.info("Model ready (loader=%s)", loader_type)

    # Frame-level inference
    frame_scores: list[dict] = []
    rows = manifest.to_dict(orient="records")
    batch_images, batch_meta = [], []

    def _flush(batch_images, batch_meta):
        if not batch_images:
            return []
        scores = _infer_batch(batch_images, model, device, loader_type)
        return [{"detector_score": s, **m} for s, m in zip(scores, batch_meta)]

    for i, row in enumerate(rows):
        face_path = row.get("face_path", "")
        try:
            img = _load_image(face_path)
            batch_images.append(img)
            batch_meta.append(row)
        except Exception as exc:
            logger.warning("Could not load %s: %s", face_path, exc)
            continue

        if len(batch_images) >= args.batch_size:
            frame_scores.extend(_flush(batch_images, batch_meta))
            batch_images, batch_meta = [], []
            if (i + 1) % 500 == 0:
                logger.info("  %d / %d faces processed", i + 1, len(rows))

    frame_scores.extend(_flush(batch_images, batch_meta))
    logger.info("Frame-level inference done: %d scored", len(frame_scores))

    if not frame_scores:
        logger.error("No frames scored — check face_path column in manifest.")
        sys.exit(1)

    frame_df = pd.DataFrame(frame_scores)

    # Aggregate to video level (mean score per video)
    id_cols = [c for c in ["video_id", "label", "split",
                            "manipulation_family", "manipulation_type"]
               if c in frame_df.columns]
    video_df = (frame_df.groupby(id_cols, dropna=False)
                .agg(detector_score=("detector_score", "mean"),
                     n_frames=("detector_score", "count"))
                .reset_index())
    video_df["detector_pred"] = (video_df["detector_score"] >= 0.5).astype(int)
    video_df["y"] = video_df["label"].astype(str).map(
        {"fake": 1, "real": 0}).fillna(0).astype(int)
    video_df["video_score_mode"] = "xception_dfb_mean"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame_path = args.out_dir / f"{args.subset}_xception_dfb_frame_scores.csv"
    frame_df.to_csv(frame_path, index=False)
    logger.info("Saved frame scores → %s", frame_path)

    video_path = args.out_dir / f"{args.subset}_xception_dfb_scores.csv"
    video_df.to_csv(video_path, index=False)
    logger.info("Saved video scores → %s  (%d videos)", video_path, len(video_df))

    # Quick AUC report
    from sklearn.metrics import roc_auc_score
    valid = video_df.dropna(subset=["detector_score", "y"])
    if valid["y"].nunique() == 2:
        auc = roc_auc_score(valid["y"], valid["detector_score"])
        logger.info("Video-level AUC: %.4f  (n=%d)", auc, len(valid))
    else:
        logger.warning("Cannot compute AUC — only one class present in scored videos.")

    end_time = now_utc()
    logger.info("Done in %.1f s", (end_time - start_time).total_seconds())


if __name__ == "__main__":
    main()
