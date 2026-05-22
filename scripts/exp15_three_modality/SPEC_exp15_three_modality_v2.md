# SPEC: Three-Modality Gated Fusion Network - Experiment 15 (v2)

This experiment is final-only. The valid data protocol is:

- `final` split values `train` and `val` are combined into `trainval` for 5-fold GroupKFold cross-validation by identity.
- `final` split value `test` is the fixed identity-disjoint holdout.
- No external or overlapping subset is used for training, validation, model selection, threshold selection, evaluation, statistics, or visualization.

## Pipeline

Run from the project root:

```bash
python scripts/exp15_three_modality/01_prepare_features.py
python scripts/exp15_three_modality/02_train_three_modality.py
python scripts/exp15_three_modality/03_evaluate_test.py
python scripts/exp15_three_modality/04_extract_gating_weights.py
python scripts/exp15_three_modality/05_visualize_gating.py
python scripts/exp15_three_modality/07_ablation_modality_removal.py
python scripts/exp15_three_modality/08_interaction_analysis.py
python scripts/exp15_three_modality/09_tsne_visualizations.py
python scripts/exp15_three_modality/10_per_generator_analysis.py
```

## Key Inputs

- `datasets/metadata/final_face_manifest.csv`
- `datasets/emotion_annotated/metadata/final_video_emotion_features.csv`
- `datasets/emotion_annotated/metadata/final_frame_emotion_predictions.csv`
- `datasets/detector_processed/final_ucf_scores.csv`

## Key Outputs

- `outputs/predictions/trainval_feature_matrix.parquet`
- `outputs/predictions/test_feature_matrix.parquet`
- `outputs/predictions/trainval_oof_predictions.csv`
- `outputs/predictions/test_exp15_predictions.csv`
- `outputs/tables/final_exp15_results.csv`
- `outputs/tables/final_exp15_ablation_summary.csv`
- `outputs/tables/final_exp15_interaction_pairs.csv`
- `outputs/tables/final_exp15_per_generator_stats.csv`
- `outputs/figures/final_exp15_roc_overlay.png`
- `outputs/figures/final_exp15_training_curves.png`
- `outputs/figures/final_exp15_tsne_per_modality.png`
- `outputs/figures/final_exp15_tsne_ucf_vs_gated.png`
- `outputs/figures/final_exp15_tsne_gating_coloured.png`

## Prediction Metadata Contract

Every prediction row must include:

- `video_id`
- `label` or `label_int`
- `prediction`
- `fold`
- `split_type`

Allowed `split_type` values:

- `trainval_oof`
- `test_holdout`

`trainval_oof_predictions.csv` uses numeric fold IDs. `test_exp15_predictions.csv` uses `fold=ensemble`.

## Acceptance Criteria

1. The trainval/test holdout split is exactly the predefined final split.
2. Trainval and test identities are disjoint.
3. Feature matrices contain no NaN or inf in configured feature columns.
4. OOF predictions cover each trainval video exactly once.
5. Test predictions cover each test video exactly once.
6. Reported metrics reproduce from raw predictions within tolerance.
7. Ablations use the same test holdout.
8. Per-generator analysis uses OOF plus final test predictions only.
