## Datasets Directory

This directory stores the generated dataset artifacts used by the thesis pipeline. It does not contain the original benchmark media files; those live outside the repository and are referenced through configuration files.

## Contents

### `metadata/`
Split manifests and merged analysis tables.

- `final_face_manifest.csv`
- `final_frame_manifest.csv`
- `final_videos_without_faces.csv`
- `pilot_face_manifest.csv`
- `pilot_frame_manifest.csv`
- `pilot_videos_without_faces.csv`
- `final_merged_hf_emotion.csv`
- `final_merged_ucf_emotion.csv`
- `final_merged_xception_emotion.csv`
- `final_merged_xception_all_emotion.csv`
- `final_xception_auc_by_emotion.csv`
- `final_xception_auc_by_arousal.csv`
- `final_xception_ablation_results.csv`
- `final_xception_fusion_results.csv`
- `final_xception_fusion_xgboost_results.csv`
- `final_xception_xgboost_ablation_results.csv`

### `emotion_annotated/metadata/`
Emotion annotation outputs derived from face crops.

- `final_frame_emotion_predictions.csv`
- `final_video_emotion_features.csv`
- `pilot_frame_emotion_predictions.csv`
- `pilot_video_emotion_features.csv`

### `detector_processed/`
Detector score files used for robustness analysis and downstream merges.

- `final_detector_scores.csv`
- `final_frame_detector_scores.csv`
- `final_huggingface_scores.csv`
- `final_huggingface_frame_scores.csv`
- `final_ucf_scores.csv`
- `final_ucf_frame_scores.csv`
- `pilot_detector_scores.csv`
- `pilot_frame_detector_scores.csv`
- `pilot_huggingface_scores.csv`
- `pilot_huggingface_frame_scores.csv`
- `pilot_ucf_scores.csv`
- `pilot_ucf_frame_scores.csv`

## Naming Convention

The `final_` prefix refers to the main experiment split, while `pilot_` refers to the smaller pilot split. Files are named by the pipeline stage that generated them so they can be traced back to the corresponding script in `scripts/`.

## How These Files Are Produced

- Face and frame manifests are produced by the preprocessing stages.
- Emotion predictions and aggregated emotion features are produced by the emotion annotation stages.
- Detector score files are produced by the detector inference stages.
- Merged analysis tables and AUC summaries are produced by the evaluation and analysis stages.
- A step-by-step reproduction guide lives in [research_artifacts/emotion_annotated_dataset/README.md](../research_artifacts/emotion_annotated_dataset/README.md).

## Reproducibility Notes

- These files are generated artifacts and can be recreated from the configured input paths.
- Keep the corresponding raw data and configuration files in sync with the artifact versions in this directory.
- When rerunning the pipeline, overwrite the existing files only after you have confirmed the new outputs are intended.

- Release artifact location: see [dataset_release/emotion_annotated/README.md](../dataset_release/emotion_annotated/README.md) for a packaged dataset handoff and hosting guidance.
