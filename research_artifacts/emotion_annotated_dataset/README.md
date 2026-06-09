# Emotion-Annotated Dataset Artifact

This folder is the self-contained handoff for recreating the emotion-annotated dataset used in the thesis.

## What it produces

- curated face manifests
- frame-level emotion predictions
- video-level emotion features
- detector score tables
- merged analysis tables for downstream experiments

## Required inputs

- benchmark metadata and media stored outside the repo
- local paths configured in `configs/paths.local.yaml`
- pipeline options configured in `configs/pipeline.local.yaml`

## Reproduction steps

Run from the repository root:

```bash
uv sync
cp configs/paths.example.yaml configs/paths.local.yaml
cp configs/pipeline.example.yaml configs/pipeline.local.yaml
```

Then build the dataset in order:

```bash
uv run python scripts/build_subset.py --config configs/base.yaml
uv run python scripts/extract_frames.py --config configs/base.yaml
uv run python scripts/extract_faces.py --config configs/base.yaml
uv run python scripts/run_emotion_annotation.py --config configs/base.yaml
uv run python scripts/aggregate_emotion_features.py --config configs/base.yaml
```

If you also want the detector and analysis artifacts used in the thesis:

```bash
uv run python scripts/run_deepfake_detector.py --config configs/base.yaml
uv run python scripts/merge_metadata.py --config configs/base.yaml
uv run python scripts/evaluate_by_emotion.py --config configs/base.yaml
uv run python scripts/run_late_fusion.py --config configs/base.yaml
```

## Notebook reference

The same workflow is explained in the preprocessing notebooks under `notebooks/data-preprocessing/`:

- `face-preprocessing__pilot.ipynb`
- `face-preprocessing__final.ipynb`
- `thesis-stage3-emotion-annotation__pilot.ipynb`
- `thesis-stage3-emotion-annotation__final.ipynb`

## Main outputs

- `datasets/metadata/*_face_manifest.csv`
- `datasets/emotion_annotated/metadata/*_frame_emotion_predictions.csv`
- `datasets/emotion_annotated/metadata/*_video_emotion_features.csv`
- `datasets/detector_processed/*_scores.csv`
- `outputs/results/*/run_metadata.json`

## Notes

- The `pilot` split is for quick checks.
- The `final` split is the thesis dataset.
- Keep the raw media and generated artifacts in sync when rerunning the pipeline.
