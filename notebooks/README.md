## Notebooks

This folder contains the exploratory and analysis notebooks used alongside the scripted pipeline. The notebooks are organized by workflow stage and usually have paired `__pilot` and `__final` versions.

## Layout

- `data-preprocessing/` - dataset inspection, face preprocessing, and emotion annotation notebooks.
- `scores-collection/` - detector inference and score aggregation notebooks.
- `experiments/` - ablation, fusion, and downstream analysis notebooks.
- `visualizations/` - thesis figures and dataset visualization notebooks.
- `outputs/` - exported notebook artifacts such as figures.

## Notebook Groups

### Data preprocessing
- `celeb_dfpp_dataset_overview.ipynb`
- `face-preprocessing__pilot.ipynb`
- `face-preprocessing__final.ipynb`
- `thesis-stage3-emotion-annotation__pilot.ipynb`
- `thesis-stage3-emotion-annotation__final.ipynb`

### Scores collection
- `stage4a_baseline_detector_inference__pilot.ipynb`
- `stage4a_baseline_detector_inference__final.ipynb`

### Experiments
- `stage4b_baseline_experiments__pilot.ipynb`
- `stage4b_baseline_experiments__final.ipynb`
- `stage4b_xception_fusion_experiments.ipynb`
- `stage4b_xception_fusion_xgboost_experiments.ipynb`

### Visualizations
- `thesis_fig2_fig3_dataset_visualizations.ipynb`

## Naming Convention

- `pilot` notebooks use the smaller pilot split.
- `final` notebooks use the main thesis split.
- Notebook filenames describe the pipeline stage and the experiment they support.

## Working With the Notebooks

Recommended setup from the repository root:

```bash
uv sync --group dev
uv run jupyter lab
```

Run notebooks from the same environment used by the scripts so that the project imports and local configuration files resolve consistently.

## Reproducibility Notes

- Notebook outputs should be cleared before committing unless a saved result is intentionally part of the repository.
- Figures generated from notebooks are stored under `outputs/`.
- The notebooks complement the scripted pipeline; the scripts remain the authoritative way to reproduce the core experiment stages.
