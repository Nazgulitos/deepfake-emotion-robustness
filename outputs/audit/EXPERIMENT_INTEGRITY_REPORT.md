# Experimental Integrity Audit Report
Generated: 2026-05-22 07:34 UTC
Repository: deepfake-emotion-robustness
Experiment: Exp.15 v2 - Three-Modality Gated Fusion (final-only)

## Summary

Overall status: PASS_WITH_WARNINGS

Total criteria checked: 14
- PASS: 12
- WARN: 1
- FAIL: 0
- CHECK_NOT_RUN: 1

## Detailed results

### CRITERION 1 - Final-only partition distinction
Status: PASS
Notes: Non-final artefacts=[]; trainval/test video overlap=0.
See: partition_size_summary.csv

### CRITERION 2 - Identity-disjoint splits
Status: PASS
Notes: Max critical identity overlap=0.
See: identity_overlap_matrix.csv

### CRITERION 3 - External subset exclusion
Status: PASS
Notes: Active source/spec hits=[]; output filename hits=[]

### CRITERION 4 - Holdout split preserved
Status: PASS
Notes: Expected test n=155, actual test n=155, symmetric_difference=0, trainval/test overlap=0.

### CRITERION 5 - Final model evaluated once on test_holdout
Status: WARN
Notes: Log test-evaluation mentions=0; Stage 07 ablations also evaluate the same fixed holdout and should be disclosed.

### CRITERION 6 - Reproducibility seeds recoverable
Status: CHECK_NOT_RUN
Notes: Seeds are fixed in config/source, but checkpoint state.json files are absent, so stored config_hash/RNG state cannot be verified.
See: config_seed_audit.csv

### CRITERION 7 - No NaN or silent fallback in features
Status: PASS
Notes: Checked 94 configured features across trainval/test; fail=0, warn=0.
See: feature_nan_audit.csv

### CRITERION 8 - Predictions attributed to partition
Status: PASS
Notes: OOF and test predictions contain required metadata and exact one-row-per-video coverage.

### CRITERION 9 - Reported metrics match raw predictions
Status: PASS
Notes: 14 metrics checked; failures=[].
See: reproduction_paths_audit.csv

### CRITERION 10 - Ablations use same holdout
Status: PASS
Notes: Ablation source reads shared test_feature_matrix.parquet.

### CRITERION 11 - No combined external-subset artefacts
Status: PASS
Notes: Combined/external artefact filename hits=[]

### CRITERION 12 - Statistical tests correctly applied
Status: PASS
Notes: DeLong, 10000-iteration permutation tests, and 2000-iteration bootstrap CIs confirmed in source/current stats.

### CRITERION 13 - Per-generator analysis partition
Status: PASS
Notes: Uses OOF plus final test source; generators=22, min n_fake=16.

### CRITERION 14 - Final-only visualization scope
Status: PASS
Notes: Visualization source external-subset terms=False; figure hits=[]

## Final verdict

No hard final-only methodological failures were found. Disclose warnings/checks not run, especially missing checkpoint state metadata if checkpoints are unavailable.