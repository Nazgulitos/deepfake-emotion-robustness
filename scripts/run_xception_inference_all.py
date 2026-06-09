"""
Run XceptionNet inference on all 800 videos via DeepfakeBench.
Writes: outputs/deepfakebench_scores/ThesisFinalAll_xception_video_scores.csv
        outputs/deepfakebench_scores/ThesisFinalAll_xception_frame_scores.csv
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path("~/deepfake-emotion-robustness").expanduser().resolve()
BENCH_ROOT   = PROJECT_ROOT / "DeepfakeBench"
sys.path.insert(0, str(BENCH_ROOT / "training"))

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR

DETECTOR_CONFIG  = BENCH_ROOT / "training/config/detector/xception.yaml"
TEST_CONFIG      = BENCH_ROOT / "training/config/test_config.yaml"
WEIGHTS_PATH     = BENCH_ROOT / "training/weights/xception_best.pth"
DATASET_JSON_DIR = PROJECT_ROOT / "outputs/deepfakebench_scores/dataset_json"
CUSTOM_DATASET   = "ThesisFinalAll"   # string, not list
OUT_DIR          = PROJECT_ROOT / "outputs/deepfakebench_scores"
SUBSET_NAME      = "final"

def main():
    """Run DeepfakeBench Xception inference for the thesis-wide evaluation set."""
    print("Loading config...")
    with open(DETECTOR_CONFIG) as f:
        cfg = yaml.safe_load(f)
    with open(TEST_CONFIG) as f:
        test_cfg = yaml.safe_load(f)
    cfg.update(test_cfg)

    rgb_root = (PROJECT_ROOT / "datasets/processed" / SUBSET_NAME / "datasets/frames").resolve()
    cfg["test_dataset"]        = CUSTOM_DATASET       # string
    cfg["dataset_json_folder"] = str(DATASET_JSON_DIR)
    cfg["rgb_dir"]             = rgb_root.as_posix()
    cfg["weights_path"] = str(WEIGHTS_PATH)
    cfg["pretrained"]    = str(BENCH_ROOT / "training/pretrained/xception-b5690688.pth")
    cfg["lmdb"]                = False
    cfg["workers"]             = 4
    cfg["cuda"]                = bool(torch.cuda.is_available())
    cfg["label_dict"]          = {"THESIS_real": 0, "THESIS_fake": 1}

    print(f"CUDA: {cfg['cuda']} | rgb_dir: {cfg['rgb_dir']}")

    device = torch.device("cuda" if cfg["cuda"] else "cpu")

    test_set = DeepfakeAbstractBaseDataset(config=cfg, mode="test")
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=cfg["test_batchSize"],
        shuffle=False,
        num_workers=int(cfg["workers"]),
        collate_fn=test_set.collate_fn,
        drop_last=False,
    )
    print(f"Dataset: {len(test_set)} frames")

    model_class = DETECTOR[cfg["model_name"]]
    model = model_class(cfg).to(device)
    ckpt = torch.load(cfg["weights_path"], map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print("Model loaded.")

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="XceptionNet inference"):
            labels = torch.where(batch["label"] != 0, 1, 0)
            batch["image"]    = batch["image"].to(device)
            batch["label"]    = labels.to(device)
            if batch.get("mask") is not None:
                batch["mask"] = batch["mask"].to(device)
            if batch.get("landmark") is not None:
                batch["landmark"] = batch["landmark"].to(device)
            pred = model(batch, inference=True)
            probs = pred["prob"].detach().cpu().numpy().reshape(-1)
            labs  = labels.detach().cpu().numpy().reshape(-1)
            all_probs.extend(probs.tolist())
            all_labels.extend(labs.tolist())

    print(f"Frames processed: {len(all_probs)}")

    frame_paths = test_set.data_dict["image"][:len(all_probs)]
    frame_df = pd.DataFrame({
        "frame_path":     pd.Series(frame_paths, dtype=str),
        "detector_score": np.array(all_probs, dtype=float),
        "label":          np.array(all_labels, dtype=int),
    })
    frame_df["video_id"] = frame_df["frame_path"].apply(lambda p: Path(p).parent.name)

    video_df = (
        frame_df.groupby("video_id", as_index=False)
        .agg(detector_score=("detector_score","mean"),
             n_frames=("frame_path","count"),
             label=("label","max"))
    )

    frame_out = OUT_DIR / f"{CUSTOM_DATASET}_xception_frame_scores.csv"
    video_out = OUT_DIR / f"{CUSTOM_DATASET}_xception_video_scores.csv"
    frame_df.to_csv(frame_out, index=False)
    video_df.to_csv(video_out, index=False)

    print(f"Saved: {video_out}")
    print(f"Videos: {len(video_df)} | labels: {video_df['label'].value_counts().to_dict()}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
