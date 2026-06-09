# SPEC - Experimental Integrity Audit

**Role of this SPEC:** This is an audit script, not a training script. The agent runs read-only checks against existing Exp.15 artefacts and writes a single report with PASS / FAIL / WARN / CHECK_NOT_RUN for each criterion.

**Scope:** Exp.15 is now a final-only experiment. The only valid partitions are:

- `trainval`: rows from `datasets/metadata/final_face_manifest.csv` where `split in {"train", "val"}`.
- `test_holdout`: rows from `datasets/metadata/final_face_manifest.csv` where `split == "test"`.

No external or overlapping subset is part of the Exp.15 protocol.

## 1. Output Location

```text
outputs/audit/
├── EXPERIMENT_INTEGRITY_REPORT.md
├── identity_overlap_matrix.csv
├── partition_size_summary.csv
├── feature_nan_audit.csv
├── config_seed_audit.csv
└── reproduction_paths_audit.csv
```

## 2. Inputs

All paths are relative to project root:

| Source | What is read |
|---|---|
| `datasets/metadata/final_face_manifest.csv` | identity, video_id, label, manipulation family/type, predefined split |
| `datasets/emotion_annotated/metadata/final_video_emotion_features.csv` | final video-level emotion features |
| `datasets/detector_processed/final_ucf_scores.csv` | UCF baseline scores |
| `scripts/exp15_three_modality/outputs/predictions/trainval_feature_matrix.parquet` | trainval feature matrix |
| `scripts/exp15_three_modality/outputs/predictions/test_feature_matrix.parquet` | fixed test feature matrix |
| `scripts/exp15_three_modality/outputs/predictions/trainval_oof_predictions.csv` | trainval OOF predictions |
| `scripts/exp15_three_modality/outputs/predictions/test_exp15_predictions.csv` | fixed test holdout predictions |
| `scripts/exp15_three_modality/outputs/tables/*.csv` | result tables |
| `scripts/exp15_three_modality/outputs/checkpoints/fold_*/state.json` | training state metadata, if present |
| `scripts/exp15_three_modality/config.yaml` | experimental config |
| `scripts/exp15_three_modality/outputs/logs/*.log` | training/evaluation logs |

## 3. Audit Criteria

### CRITERION 1 - Final-only partition distinction

Verify that Exp.15 prediction and feature artefacts contain only final-manifest videos, with `trainval` and `test_holdout` stored separately. `trainval` and `test_holdout` video_id sets must be disjoint.

### CRITERION 2 - Identity-disjoint splits

Verify all per-fold train/validation identity groups are disjoint within each fold. Verify `trainval` and `test_holdout` identities are disjoint.

### CRITERION 3 - External subset exclusion

Verify active Exp.15 scripts, config, and current Exp.15 outputs do not read, write, or report any external subset artefacts.

### CRITERION 4 - Holdout split preserved

Verify the actual `test_feature_matrix.parquet` video IDs exactly match the predefined final manifest rows where `split == "test"` and are not present in `trainval_feature_matrix.parquet`.

### CRITERION 5 - Final model evaluated once on test_holdout

Search logs and source for test-holdout evaluation events. WARN if multiple test evaluations exist due to ablations; FAIL only if source uses test metrics for model selection after evaluation.

### CRITERION 6 - Reproducibility seeds are fixed and recoverable

Verify `seed: 42` in config, seeded Python/numpy/torch/cuda code, and checkpoint `config_hash` / RNG state when checkpoint metadata exists.

### CRITERION 7 - No NaN or silent fallback in features

Verify configured feature columns in trainval/test matrices contain no NaN or inf. WARN on placeholder values such as `-1`.

### CRITERION 8 - Predictions attributed to partition

Each prediction row must include `video_id`, `label` or `label_int`, `prediction`, `fold`, and `split_type`. Allowed `split_type` values are `trainval_oof` and `test_holdout`.

### CRITERION 9 - Reported metrics match raw predictions

Recompute AUC, accuracy, F1, precision, recall, and EER from raw predictions and compare to `final_exp15_results.csv` with tolerance `< 0.001`.

### CRITERION 10 - Ablations use same holdout

Verify all ablations use the same `test_feature_matrix.parquet` holdout.

### CRITERION 11 - No combined external-subset artefacts

Verify there are no current Exp.15 outputs combining final test data with any external subset.

### CRITERION 12 - Statistical tests correctly applied

Verify AUC comparisons use DeLong where reported, permutation tests use at least 10,000 iterations, bootstrap CIs use at least 2,000 iterations, and multiple-comparison correction is disclosed where applicable.

### CRITERION 13 - Per-generator analysis uses appropriate partition

Verify per-generator analysis uses trainval OOF plus final test predictions, and `n_fake >= 10` for reported generator-level AUC rows.

### CRITERION 14 - Final-only visualization scope

Verify visualization scripts and outputs use final test data only for test-set visualizations and do not load external-subset feature matrices or scores.

## 4. Report Format

The audit report must include:

- generated timestamp,
- repository and experiment name,
- overall status,
- count of PASS / WARN / FAIL / CHECK_NOT_RUN,
- one section per criterion,
- links to supporting CSVs where relevant,
- final verdict.

## 5. Acceptance Criteria

1. All 14 criteria are evaluated.
2. Report is written to `outputs/audit/EXPERIMENT_INTEGRITY_REPORT.md`.
3. Supporting CSVs are written for detailed checks.
4. The audit does not modify existing files outside `outputs/audit/`.
5. If a check cannot be performed, report `CHECK_NOT_RUN` with explanation.
