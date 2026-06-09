# Deepfake Emotion Robustness

Development of an Emotion-Annotated Deepfake Video Dataset and Analysis of Deepfake Detection Robustness to Facial Emotional Dynamics.

## Project Overview
This repository contains the code and research artifacts for a bachelor thesis on how facial emotion affects deepfake detection. The pipeline is config-driven and reproducible: it prepares a benchmark subset, extracts frames and faces, runs emotion annotation, aggregates emotion features, evaluates baseline detectors, and compares results across emotional conditions.

## Motivation
Deepfake detectors are usually evaluated on clean benchmark splits, but real videos often contain strong facial expressions, rapid emotion changes, and inconsistent face quality. This project studies whether those conditions change detector reliability and whether explicit emotion features help explain or improve robustness.

## Features
- Curated subset construction from benchmark metadata.
- Frame extraction and face cropping.
- Emotion annotation with a pretrained FER model.
- Video-level aggregation of emotion and temporal descriptors.
- Baseline deepfake inference and late-fusion experiments.
- Robustness analysis by emotion, forgery family, and split.
- Statistical testing, SHAP/UMAP analysis, and thesis artifact generation.

## Architecture Overview
The repository is organized as a staged data pipeline:

1. `scripts/build_subset.py` prepares a curated manifest and split metadata.
2. `scripts/extract_frames.py` and `scripts/extract_faces.py` create visual inputs.
3. `scripts/run_emotion_annotation.py` annotates each face crop with emotion scores.
4. `scripts/aggregate_emotion_features.py` converts frame-level predictions into video-level features.
5. `scripts/run_deepfake_detector.py` runs the detector baseline and writes detector scores.
6. `scripts/merge_metadata.py` assembles the final analysis table.
7. `scripts/evaluate_by_emotion.py` and `scripts/run_late_fusion.py` produce the final evaluation outputs.

The reusable implementation lives under `src/`, while experiment-specific logic and ablation utilities live in `scripts/exp15_*`.

## Repository Structure
```text
.
|-- configs/                 # Base config plus local override examples
|-- datasets/                # Prepared dataset artifacts and metadata
|-- docs/                    # Data schema, project spec, and research notes
|-- outputs/                 # Generated tables, figures, logs, and reports
|-- scripts/                 # Pipeline entrypoints and analysis scripts
|-- src/                     # Reusable preprocessing, emotion, detection, and eval code
|-- tests/                   # Unit tests for aggregation, metrics, splits, and statistics
|-- main.pdf                 # Compiled thesis report included with the release
|-- Makefile                 # Convenience targets for the full pipeline
|-- pyproject.toml           # Project metadata and dependency declarations
|-- uv.lock                  # Locked dependency graph for uv
|-- verify_ablation.py       # Ablation verification helper
`-- README.md
```

## Installation
1. Create a Python 3.10+ environment.
2. Install the project dependencies with `uv`:

```bash
uv sync
```

3. If you plan to run notebooks or interactive analysis, install the notebook-related extras as well:

```bash
uv sync --group dev
```

## Quick Start
1. Copy the example configuration files and point them at your local dataset locations:

```bash
cp configs/paths.example.yaml configs/paths.local.yaml
cp configs/pipeline.example.yaml configs/pipeline.local.yaml
```

2. Run the full pipeline with the default configuration:

```bash
make all
```

3. Inspect the generated tables and figures under `outputs/`.

## Usage Examples
Run individual stages directly when you only need part of the workflow:

```bash
uv run python scripts/build_subset.py --config configs/base.yaml
uv run python scripts/extract_frames.py --config configs/base.yaml
uv run python scripts/extract_faces.py --config configs/base.yaml
uv run python scripts/run_emotion_annotation.py --config configs/base.yaml
uv run python scripts/aggregate_emotion_features.py --config configs/base.yaml
uv run python scripts/run_deepfake_detector.py --config configs/base.yaml
uv run python scripts/merge_metadata.py --config configs/base.yaml
uv run python scripts/evaluate_by_emotion.py --config configs/base.yaml
uv run python scripts/run_late_fusion.py --config configs/base.yaml
```

For the experiment-specific three-modality ablation pipeline:

```bash
uv run python scripts/exp15_three_modality/01_prepare_features.py
uv run python scripts/exp15_three_modality/03_evaluate_test.py
```

## Configuration
The main configuration lives in `configs/base.yaml`. Local path overrides can be stored in `configs/paths.local.yaml` and `configs/pipeline.local.yaml`; the repository keeps the `.example.yaml` versions as templates. The pipeline is intentionally file-based, so the main knobs are input paths, output locations, split definitions, and experiment parameters rather than environment variables.

## Reproducibility
- Random seeds are fixed in the scripted stages that support them.
- Config files make the data locations and split definitions explicit.
- Intermediate artifacts are written to disk so each stage can be rerun independently.
- The repo includes the compiled thesis report as `main.pdf` for reference.
- Use `make test` to check the lightweight regression suite when making changes.

## Results
The repository produces analysis tables, figures, and score files under `outputs/`. Key outputs include subset manifests, frame and face manifests, emotion predictions, aggregated emotion features, detector scores, final merged tables, robustness summaries, and thesis-ready figures and tables.

## Future Work
- Add automated CI for tests and linting.
- Publish a fully parameterized dataset download guide for external users.
- Expand benchmark coverage beyond the current thesis experiments.
- Add notebook execution checks for the most important exploratory analyses.

## Citation
If you use this repository in academic work, please cite the thesis and the repository. Replace the placeholder metadata below with the final thesis citation details before publication.

```bibtex
@unknown{deepfakeEmotionRobustness2026,
    title  = {Development of an Emotion-Annotated Deepfake Video Dataset and Analysis of Deepfake Detection Robustness to Facial Emotional Dynamics},
    author = {Nazgul Salikhova, Ilya Makarov},
    school = {Innopolis University},
    year   = {2026},
    note   = {Repository and thesis report for the Deepfake Emotion Robustness project}
}
```

## License
A placeholder `LICENSE` file is included so the repository is ready for release packaging. Replace it with the final open-source license before the public GitHub release if you want to grant explicit reuse rights.
