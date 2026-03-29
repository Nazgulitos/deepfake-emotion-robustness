# Deepfake Emotion Robustness (Bachelor Thesis)

## Project Title
Development of an Emotion-Annotated Deepfake Video Dataset and Analysis of Deepfake Detection Robustness to Facial Emotional Dynamics

## Overview
This repository contains a modular and reproducible Python pipeline to:
1. Build a curated subset from public deepfake benchmarks.
2. Extract frames and face crops.
3. Annotate facial emotions with a pretrained FER model.
4. Aggregate frame-level emotions into video-level descriptors.
5. Run a baseline deepfake detector.
6. Merge metadata into one analysis table.
7. Evaluate detection robustness across emotional conditions.
8. Optionally run a simple late-fusion baseline.

The scope is intentionally practical for a bachelor thesis. No large model training is required.

## Research Questions
- RQ1: Does the type and intensity of facial emotion affect deepfake detection quality?
- RQ2: Which emotional conditions are most difficult for deepfake detectors?
- RQ3: Can explicit emotional features improve deepfake detection robustness?

## Hypotheses
- H1: Detection quality decreases for videos with strong and rapidly changing emotions.
- H2: Detection errors correlate with facial emotional dynamics.
- H3: Adding emotional descriptors to detector outputs improves robustness.

## Repository Structure
```text
.
|-- README.md
|-- .gitignore
|-- requirements.txt
|-- Makefile
|-- configs/
|   |-- base.yaml
|   |-- paths.example.yaml
|   `-- pipeline.example.yaml
|-- docs/
|   |-- experiment_plan.md
|   `-- thesis_notes.md
|-- scripts/
|   |-- build_subset.py
|   |-- extract_frames.py
|   |-- extract_faces.py
|   |-- run_emotion_annotation.py
|   |-- aggregate_emotion_features.py
|   |-- run_deepfake_detector.py
|   |-- merge_metadata.py
|   |-- evaluate_by_emotion.py
|   `-- run_late_fusion.py
|-- src/
|   |-- __init__.py
|   |-- data/
|   |   |-- __init__.py
|   |   `-- subset_builder.py
|   |-- preprocessing/
|   |   |-- __init__.py
|   |   |-- frame_extractor.py
|   |   `-- face_extractor.py
|   |-- emotion/
|   |   |-- __init__.py
|   |   |-- annotator.py
|   |   `-- aggregation.py
|   |-- detection/
|   |   |-- __init__.py
|   |   |-- baseline_detector.py
|   |   `-- fusion.py
|   |-- features/
|   |   |-- __init__.py
|   |   `-- merge.py
|   |-- evaluation/
|   |   |-- __init__.py
|   |   `-- metrics.py
|   `-- utils/
|       |-- __init__.py
|       |-- config.py
|       |-- io.py
|       |-- logging_utils.py
|       `-- naming.py
|-- notebooks/
|   `-- .gitkeep
|-- metadata/
|   `-- .gitkeep
`-- outputs/
    |-- figures/
    |   `-- .gitkeep
    |-- logs/
    |   `-- .gitkeep
    `-- tables/
        `-- .gitkeep
```

## Setup
1. Create and activate a Python environment (recommended Python 3.10+).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy and edit configuration examples as needed:

```bash
cp configs/paths.example.yaml configs/paths.local.yaml
cp configs/pipeline.example.yaml configs/pipeline.local.yaml
```

## Quick Run (Stage by Stage)
All scripts use `configs/base.yaml` by default.

```bash
PYTHONPATH=. python scripts/build_subset.py --config configs/base.yaml
PYTHONPATH=. python scripts/extract_frames.py --config configs/base.yaml
PYTHONPATH=. python scripts/extract_faces.py --config configs/base.yaml
PYTHONPATH=. python scripts/run_emotion_annotation.py --config configs/base.yaml
PYTHONPATH=. python scripts/aggregate_emotion_features.py --config configs/base.yaml
PYTHONPATH=. python scripts/run_deepfake_detector.py --config configs/base.yaml
PYTHONPATH=. python scripts/merge_metadata.py --config configs/base.yaml
PYTHONPATH=. python scripts/evaluate_by_emotion.py --config configs/base.yaml
PYTHONPATH=. python scripts/run_late_fusion.py --config configs/base.yaml
```

Or use Make targets:

```bash
make all
```

## Expected Core Outputs
- metadata/subset_manifest.csv
- metadata/frame_manifest.csv
- metadata/face_manifest.csv
- metadata/emotion_frame_predictions.csv
- metadata/video_emotion_features.csv
- metadata/detector_scores.csv
- metadata/final_merged_table.csv
- outputs/figures/
- outputs/tables/

## Notes
- This scaffold intentionally includes TODO blocks where benchmark-specific implementation details are needed.
- Paths are config-driven and do not assume private dataset access.
- Intermediate tables can be saved as CSV or Parquet based on file extension.
