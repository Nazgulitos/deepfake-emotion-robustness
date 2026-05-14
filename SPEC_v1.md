# SPEC v2: Emotion-Aware Deepfake Detection — Reproducible Experimental Pipeline

**Thesis title:** Development of an Emotion-Annotated Deepfake Video Dataset and Study of the Impact of Facial Emotional Dynamics on Deepfake Detection Robustness

**Repository:** `deepfake-emotion-robustness/`
**Deadline:** 2 weeks

---

## 1. Strategy: Build on Existing Structure

The repository already has a clean separation of concerns: `configs/` for settings, `scripts/` as CLI entry points, `src/` for reusable code, `datasets/` for intermediate artifacts, and `notebooks/` for exploration. **This SPEC does not require restructuring.** It defines new files to add and gaps to fill.

---

## 2. What Already Exists vs. What Is Missing

### 2.1 Already in place ✅

| Component | Existing location | Status |
|---|---|---|
| Frame extraction | `scripts/extract_frames.py` + `src/preprocessing/frame_extractor.py` | Working |
| Face detection | `scripts/extract_faces.py` + `src/preprocessing/face_extractor.py` | Working |
| Emotion annotation | `scripts/run_emotion_annotation.py` + `src/emotion/annotator.py` | Working |
| Descriptor aggregation | `scripts/aggregate_emotion_features.py` + `src/emotion/aggregation.py` | Working |
| Baseline detector | `scripts/run_deepfake_detector.py` + `src/detection/baseline_detector.py` | Working (XceptionNet) |
| Late fusion | `scripts/run_late_fusion.py` + `src/detection/fusion.py` | Working |
| Subgroup eval | `scripts/evaluate_by_emotion.py` | Working |
| Metrics | `src/evaluation/metrics.py` | Working |
| Subset construction | `scripts/build_subset.py` + `src/data/subset_builder.py` | Working |
| Final + pilot manifests | `datasets/metadata/` | Both subsets exist |
| Frame-level detector scores | `outputs/deepfakebench_scores/` and `datasets/detector_processed/` | XceptionNet done |
| Existing experiment results | `datasets/metadata/final_xception_*.csv` | Exp. 2 and 3 results stored |
| Visualization notebook | `notebooks/visualizations/thesis_fig2_fig3_dataset_visualizations.ipynb` | Working |

### 2.2 Missing — to add ❌

| Component | Reason |
|---|---|
| **HuggingFace baseline scores stored** | Exp. 1 used `dima806/...` but scores not in `datasets/detector_processed/` |
| **Transformer baseline (Exp. 8)** | Need to add a third detector run |
| **Per-emotion-class subgroup analysis (Exp. 5)** | Script exists for arousal but not per-class |
| **Forgery × emotion cross-tab (Exp. 6)** | Not implemented |
| **Statistical tests module (Exp. 7)** | DeLong + Spearman + permutation tests |
| **SHAP analysis (Exp. 9)** | Not implemented |
| **UMAP projection (Exp. 10)** | Not implemented |
| **Pilot holdout validation (Exp. 11)** | Pilot data exists but never used as out-of-tuning test |
| **Dated results folder** | `outputs/tables/` and `outputs/figures/` are flat; no run versioning |
| **Reproducibility scaffolding** | No `run_metadata.json` per run, no seed enforcement check |
| **Unit tests** | No `tests/` directory |
| **Notebook naming** | The `copy.ipynb` files actually contain the final-dataset versions; current names don't make this explicit |

---

## 3. Cleanup Tasks (Day 0, ~1 hour)

Before any new work, perform these housekeeping tasks:

1. **Rename `copy.ipynb` files to reflect that they are the final-dataset versions.** The originals process the pilot subset and the `copy` variants process the final 800-video subset. Use the convention `{original_name}__final.ipynb` and append `__pilot.ipynb` to the originals:

   | Current name | Renamed to |
   |---|---|
   | `notebooks/data-preprocessing/face-preprocessing.ipynb` | `notebooks/data-preprocessing/face-preprocessing__pilot.ipynb` |
   | `notebooks/data-preprocessing/face-preprocessing copy.ipynb` | `notebooks/data-preprocessing/face-preprocessing__final.ipynb` |
   | `notebooks/data-preprocessing/thesis-stage3-emotion-annotation.ipynb` | `notebooks/data-preprocessing/thesis-stage3-emotion-annotation__pilot.ipynb` |
   | `notebooks/data-preprocessing/thesis-stage3-emotion-annotation copy.ipynb` | `notebooks/data-preprocessing/thesis-stage3-emotion-annotation__final.ipynb` |
   | `notebooks/scores-collection/stage4a_baseline_detector_inference.ipynb` | `notebooks/scores-collection/stage4a_baseline_detector_inference__pilot.ipynb` |
   | `notebooks/scores-collection/stage4a_baseline_detector_inference copy.ipynb` | `notebooks/scores-collection/stage4a_baseline_detector_inference__final.ipynb` |
   | `notebooks/experiments/stage4b_baseline_experiments.ipynb` | `notebooks/experiments/stage4b_baseline_experiments__pilot.ipynb` |
   | `notebooks/experiments/stage4b_baseline_experiments copy.ipynb` | `notebooks/experiments/stage4b_baseline_experiments__final.ipynb` |

   Rationale: the current naming hides that two separate runs exist. Filenames with spaces also break shell autocomplete and many CI tools. Using `__pilot` and `__final` suffixes makes the subset explicit and matches the `final_*` / `pilot_*` CSV naming convention already used throughout `datasets/`.

2. **At the top of each renamed notebook, add a markdown cell stating the subset processed.** Example for `face-preprocessing__final.ipynb`:
   ```markdown
   # Face Preprocessing — Final Dataset (N = 800)

   This notebook processes the **final** curated subset.
   The pilot version is in `face-preprocessing__pilot.ipynb`.
   Inputs: `datasets/metadata/final_frame_manifest.csv`
   Outputs: `datasets/metadata/final_face_manifest.csv`
   ```

3. **Add `.gitignore` entries** for `.venv/`, `__pycache__/`, `*.pyc`, large `outputs/` artifacts.

4. **Verify `requirements.txt` matches `uv.lock`** — pin everything currently in use.

5. **Create new directories** that don't yet exist:
   ```
   src/analysis/        # for new analysis modules
   src/utils/splits.py  # GroupKFold helpers (if not already present)
   tests/               # unit tests
   outputs/results/     # dated experiment results
   ```

---

## 4. Naming Convention Decisions (Apply to All New Files)

Your existing files use the prefix `final_` and `pilot_` to distinguish subsets. **Keep this convention.** New analysis outputs must follow the same pattern.

| Type | Pattern | Example |
|---|---|---|
| Per-subset experiment table | `{subset}_{exp_id}_{descriptor}.csv` | `final_exp05_per_emotion_auc.csv` |
| Per-subset figure | `{subset}_{exp_id}_{descriptor}.png` | `final_exp06_forgery_emotion_heatmap.png` |
| Statistical test result | `{subset}_{exp_id}_{test_name}.json` | `final_exp07_h1_delong.json` |
| Run metadata | `{subset}_{exp_id}_metadata.json` | `final_exp05_metadata.json` |

---

## 5. New Files to Add

### 5.1 New scripts (one per new experiment)

```
scripts/
├── run_huggingface_detector.py       # Exp. 1 — store HF scores in same format as XceptionNet
├── run_transformer_detector.py       # Exp. 8 — add UCF or CLIP from DeepfakeBench
├── analyze_per_emotion_auc.py        # Exp. 5
├── analyze_forgery_emotion_crosstab.py  # Exp. 6
├── run_statistical_tests.py          # Exp. 7 (DeLong + Spearman + permutation)
├── analyze_shap.py                   # Exp. 9
├── analyze_umap.py                   # Exp. 10
├── validate_on_pilot.py              # Exp. 11
└── build_thesis_artifacts.py         # final aggregation step
```

### 5.2 New `src/` modules

```
src/
├── analysis/
│   ├── __init__.py
│   ├── subgroup_auc.py               # reusable subgroup analysis (used by Exp. 4b, 5, 6)
│   ├── statistical_tests.py          # delong, bootstrap, spearman
│   ├── shap_helpers.py
│   └── umap_helpers.py
└── utils/
    ├── splits.py                     # GroupKFold + identity disjoint split generator
    └── run_metadata.py               # writes run_metadata.json with git hash + seeds
```

### 5.3 New config

```
configs/
└── experiments.yaml                  # experiment registry (see §7)
```

### 5.4 New tests

```
tests/
├── test_metrics.py                   # AUC matches sklearn; bootstrap CI sane
├── test_splits.py                    # train/val/test identities disjoint
├── test_aggregation.py               # descriptor math on hand-crafted input
└── test_statistical_tests.py         # DeLong p-value sane on identical vs different scores
```

---

## 6. Data Flow (Existing + New)

```
Stage 0: Subset construction (existing: build_subset.py)
   └─> datasets/metadata/{final,pilot}_*_manifest.csv

Stage 1: Frame extraction (existing: extract_frames.py)
   └─> outputs/frames/ (not stored in git — local only)

Stage 2: Face detection (existing: extract_faces.py)
   └─> outputs/crops/ (local only)
   └─> datasets/metadata/{final,pilot}_face_manifest.csv

Stage 3: Emotion annotation (existing: run_emotion_annotation.py)
   └─> datasets/emotion_annotated/metadata/{final,pilot}_frame_emotion_predictions.csv

Stage 4: Descriptor aggregation (existing: aggregate_emotion_features.py)
   └─> datasets/emotion_annotated/metadata/{final,pilot}_video_emotion_features.csv

Stage 5a: XceptionNet inference (existing: run_deepfake_detector.py)
   └─> outputs/deepfakebench_scores/ThesisFinal_xception_*.csv
   └─> datasets/detector_processed/{final,pilot}_detector_scores.csv

Stage 5b: HuggingFace inference (NEW: run_huggingface_detector.py)
   └─> datasets/detector_processed/{final,pilot}_huggingface_scores.csv

Stage 5c: Transformer inference (NEW: run_transformer_detector.py)
   └─> datasets/detector_processed/{final,pilot}_transformer_scores.csv

Stage 6: Existing Experiments — already stored
   ├─> datasets/metadata/final_xception_ablation_results.csv (Exp. 2)
   ├─> datasets/metadata/final_xception_fusion_results.csv (Exp. 2)
   ├─> datasets/metadata/final_xception_fusion_xgboost_results.csv (Exp. 3)
   └─> datasets/metadata/final_xception_xgboost_ablation_results.csv (Exp. 3)

Stage 7: New Experiments (all NEW)
   └─> outputs/results/YYYY-MM-DD/{exp_id}/
        ├── tables/*.csv + *.tex
        ├── figures/*.png + *.pdf
        ├── stats/*.json
        └── run_metadata.json
```

---

## 7. Experiment Registry (`configs/experiments.yaml`)

```yaml
subsets:
  final:
    manifest: datasets/metadata/final_face_manifest.csv
    emotion_features: datasets/emotion_annotated/metadata/final_video_emotion_features.csv
    n_videos: 800
  pilot:
    manifest: datasets/metadata/pilot_face_manifest.csv
    emotion_features: datasets/emotion_annotated/metadata/pilot_video_emotion_features.csv
    n_videos: 200

detectors:
  xception:
    scores: datasets/detector_processed/final_detector_scores.csv
    status: done
  huggingface:
    scores: datasets/detector_processed/final_huggingface_scores.csv
    status: needs_recompute    # was computed but not stored in standard format
  transformer:
    scores: datasets/detector_processed/final_transformer_scores.csv
    status: todo

experiments:
  exp01:
    status: done
    detector: huggingface
    fusion: logistic_regression
    output_table: datasets/metadata/final_huggingface_fusion_results.csv
    rq: [3]
    hypothesis: [3]

  exp02:
    status: done
    detector: xception
    fusion: logistic_regression
    output_table: datasets/metadata/final_xception_ablation_results.csv
    rq: [1, 3]
    hypothesis: [3]

  exp03:
    status: done
    detector: xception
    fusion: xgboost
    output_table: datasets/metadata/final_xception_xgboost_ablation_results.csv
    rq: [3]
    hypothesis: [3]

  exp04b:
    status: existing_data_new_analysis
    detector: xception
    type: subgroup_analysis
    stratify_by: arousal_tercile
    existing_input: datasets/metadata/final_xception_auc_by_arousal.csv
    output_table: outputs/results/{date}/exp04b/final_exp04b_arousal_subgroup.csv
    rq: [1]
    hypothesis: [1]

  exp05:
    status: existing_data_new_analysis
    detector: xception
    type: subgroup_analysis
    stratify_by: dominant_emotion
    existing_input: datasets/metadata/final_xception_auc_by_emotion.csv
    output_table: outputs/results/{date}/exp05/final_exp05_per_emotion_auc.csv
    rq: [1, 2]
    hypothesis: [1]

  exp06:
    status: todo
    detector: xception
    type: cross_tabulation
    stratify_by: [forgery_family, dominant_emotion]
    output_table: outputs/results/{date}/exp06/final_exp06_forgery_emotion.csv
    output_figure: outputs/results/{date}/exp06/final_exp06_heatmap.png
    rq: [2]
    hypothesis: [1]

  exp07:
    status: todo
    type: statistical
    tests:
      - h1_delong_arousal_terciles
      - h1_delong_emotion_classes
      - h2_spearman_error_vs_descriptors
      - h3_delong_fusion_vs_baseline
    rq: [1, 3]
    hypothesis: [1, 2, 3]

  exp08:
    status: todo
    detector: transformer
    fusion: logistic_regression
    rq: [3]
    hypothesis: [3]

  exp09:
    status: todo
    type: interpretability
    upstream: exp03
    rq: [1, 3]

  exp10:
    status: todo
    type: visualization
    rq: [1]
    hypothesis: [3]

  exp11:
    status: todo
    type: generalization
    upstream: exp02
    test_subset: pilot
    rq: [3]
    hypothesis: [3]
```

---

## 8. Data Schema Alignment

Existing CSV files use specific column names. **New code must read existing files using their current schema** and **new files must follow the same schema** for downstream consistency.

### 8.1 Verify existing column names (Day 0 task)

Open each file and record actual column names:

```python
# scripts/inspect_schema.py — run once
import pandas as pd
from pathlib import Path

for csv in Path("datasets").rglob("*.csv"):
    df = pd.read_csv(csv, nrows=1)
    print(f"{csv.name}: {list(df.columns)}")
```

Document column names in `docs/data_schema.md`. All new scripts read/write using these exact names.

### 8.2 Required columns across files

These columns must appear in their respective files for downstream code to work:

- **Manifests:** `video_id`, `label`, `identity`, `forgery_family`, `generator`, `split`
- **Frame-level emotion:** `video_id`, `frame_index`, `emonet_class`, `emonet_valence`, `emonet_arousal`
- **Video-level features:** `video_id`, `dominant_emotion`, `mean_arousal`, `mean_valence`, `arousal_variation`, `transition_rate`, `emotion_entropy`, `neutral_ratio`, `max_arousal`
- **Detector scores:** `video_id`, `video_score` (per-video mean), `frame_index` (for frame-level files)

If any column is missing or named differently, add a one-time migration script in `scripts/migrate_schema.py` rather than touching the source files.

---

## 9. Reproducibility Requirements

### 9.1 Every experiment script must:

1. Accept `--config configs/experiments.yaml --exp_id expNN --subset {final,pilot}` arguments
2. Read `configs/base.yaml` for global seed (default 42)
3. Set seeds for numpy, torch, sklearn, xgboost, umap at the start
4. Write `outputs/results/YYYY-MM-DD/{exp_id}/run_metadata.json` with:
   - git commit hash (`subprocess.check_output(["git","rev-parse","HEAD"])`)
   - Python version
   - Package versions (or path to `requirements.txt`)
   - Random seed used
   - CLI arguments
   - Start and end timestamps
5. Log to both stdout and `outputs/results/YYYY-MM-DD/{exp_id}/run.log`
6. Write outputs atomically (write to `.tmp` file, rename on success)

### 9.2 Makefile additions

Add to existing `Makefile`:

```makefile
.PHONY: huggingface transformer exp04b exp05 exp06 exp07 exp08 exp09 exp10 exp11 all-new

DATE := $(shell date +%Y-%m-%d)

huggingface:
	python scripts/run_huggingface_detector.py --subset final
	python scripts/run_huggingface_detector.py --subset pilot

transformer:
	python scripts/run_transformer_detector.py --subset final

exp04b:
	python scripts/evaluate_by_emotion.py --exp_id exp04b --subset final
exp05:
	python scripts/analyze_per_emotion_auc.py --exp_id exp05 --subset final
exp06:
	python scripts/analyze_forgery_emotion_crosstab.py --exp_id exp06 --subset final
exp07:
	python scripts/run_statistical_tests.py --exp_id exp07 --subset final
exp08:
	python scripts/run_late_fusion.py --exp_id exp08 --detector transformer --subset final
exp09:
	python scripts/analyze_shap.py --exp_id exp09 --subset final
exp10:
	python scripts/analyze_umap.py --exp_id exp10 --subset final
exp11:
	python scripts/validate_on_pilot.py --exp_id exp11

all-new: huggingface transformer exp04b exp05 exp06 exp07 exp08 exp09 exp10 exp11

thesis-artifacts:
	python scripts/build_thesis_artifacts.py --date $(DATE)

test:
	pytest tests/ -v
```

---

## 10. Output Specification per Experiment

| Exp | Tables | Figures | Stats |
|---|---|---|---|
| 01 | `final_huggingface_fusion_results.csv` (already exists in your `metadata/`?) | `final_exp01_roc.png` | — |
| 02 | `final_xception_ablation_results.csv` ✅ | `final_exp02_roc.png` | — |
| 03 | `final_xception_xgboost_ablation_results.csv` ✅ | `final_exp03_roc.png` | — |
| 04b | `final_exp04b_arousal_subgroup.csv` | `final_exp04b_auc_by_arousal.png` | `final_exp04b_delong.json` |
| 05 | `final_exp05_per_emotion_auc.csv` | `final_exp05_emotion_auc_bars.png` | `final_exp05_delong.json` |
| 06 | `final_exp06_forgery_emotion.csv` | `final_exp06_heatmap.png` | — |
| 07 | `final_exp07_h2_spearman.csv` | — | `final_exp07_h1.json`, `final_exp07_h3.json` |
| 08 | `final_exp08_transformer_results.csv` | `final_exp08_roc_overlay.png` | — |
| 09 | `final_exp09_shap_importance.csv` | `final_exp09_shap_summary.png`, `final_exp09_dependence.png` | — |
| 10 | — | `final_exp10_umap_by_label.png`, `final_exp10_umap_by_emotion.png` | — |
| 11 | `pilot_exp11_holdout_results.csv` | `pilot_exp11_roc.png` | — |

---

## 11. Final Aggregation Step

`scripts/build_thesis_artifacts.py` reads all output tables and figures from the most recent dated folder and produces:

- `outputs/thesis_artifacts/{date}/all_tables.tex` — IEEE-formatted LaTeX tables
- `outputs/thesis_artifacts/{date}/all_figures.zip` — figures named `Fig_NN_description.png`
- `outputs/thesis_artifacts/{date}/results_summary.md` — copy-pasteable Chapter 4 paragraphs per experiment

---

## 12. Acceptance Criteria

The project is complete when:

1. `make all-new` runs end-to-end without manual intervention
2. Every experiment in §7 produces files listed in §10
3. `pytest tests/` passes
4. Three existing experiments (Exp. 01–03) are verified to reproduce their stored CSV values within 1% AUC tolerance when re-run from raw data
5. `results_summary.md` exists and contains a paragraph per experiment
6. A fresh clone + `make all-new` produces identical numerical results

---

## 13. Out of Scope

- Restructuring existing directories
- Renaming existing scripts in `src/`
- Audio modality, real-time inference, web UI
- LLM/VLM-based detection
- Cross-dataset evaluation beyond Celeb-DF++

---

## 14. PR Sequence for the Agent

| PR | Scope | Verification |
|---|---|---|
| 1 | Cleanup (delete duplicates, add `.gitignore`), `tests/` skeleton, `configs/experiments.yaml`, `docs/data_schema.md` | `pytest` passes on empty tests |
| 2 | `src/utils/splits.py`, `src/utils/run_metadata.py`, `src/analysis/statistical_tests.py` | Unit tests pass |
| 3 | `scripts/run_huggingface_detector.py` produces `final_huggingface_scores.csv` | Schema matches existing XceptionNet scores |
| 4 | Verify existing Exp. 01–03 reproduce from stored data via new pipeline (no recompute, just rebuild tables) | Numerical match within 1% |
| 5 | Exp. 04b + Exp. 05 (per-emotion-class subgroup) | Tables + figures produced |
| 6 | Exp. 06 (forgery × emotion cross-tab) | Heatmap produced |
| 7 | Exp. 07 (statistical tests — DeLong, Spearman) | All hypothesis tests output JSON |
| 8 | Exp. 08 (transformer detector + fusion) | Third baseline AUC stored |
| 9 | Exp. 09 (SHAP) + Exp. 10 (UMAP) | Interpretability figures produced |
| 10 | Exp. 11 (pilot holdout) + `build_thesis_artifacts.py` | Final aggregated tables and figures |

Each PR includes: updated tests, README delta describing how to run new pieces, and a one-paragraph summary of what was verified.

---

## 15. Critical Notes for the Agent

1. **Do not rerun frame extraction or face detection** unless data is missing. Treat `datasets/metadata/*_face_manifest.csv` as ground truth for which frames exist.
2. **Do not re-extract emotion annotations** unless they are missing. EmoNet inference is expensive.
3. **Existing Exp. 01–03 outputs are authoritative.** New tables must match these within 1% AUC after re-running. If they don't match, debug before adding new experiments.
4. **The pilot subset is held out** from all model selection and hyperparameter tuning. Only Exp. 11 may touch it for evaluation.
5. **Use `final_` and `pilot_` prefix consistently** — never break this convention.
