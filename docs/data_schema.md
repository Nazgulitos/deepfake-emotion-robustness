# Data Schema Reference

All column names are as observed in the actual CSV files (verified 2026-05-13).
New scripts must read/write using these exact names.

---

## Face Manifests (`datasets/metadata/{final,pilot}_face_manifest.csv`)

| Column | Type | Description |
|---|---|---|
| `video_id` | str | Unique video identifier |
| `frame_id` | str | Unique frame identifier |
| `timestamp_sec` | float | Frame timestamp in seconds |
| `label` | int | 0 = real, 1 = fake |
| `split` | str | `train`, `val`, or `test` |
| `manipulation_family` | str | Forgery family (e.g. `FaceSwap`, `FaceEnactment`) |
| `manipulation_type` | str | Specific generator name |
| `identity` | str | Subject identity |
| `source_subset` | str | `final` or `pilot` |
| `video_path` | str | Path to source video |
| `frame_path` | str | Path to extracted frame |
| `face_id` | str | Unique face crop identifier |
| `face_path` | str | Path to face crop image |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | float | Face bounding box pixels |
| `det_score` | float | Face detector confidence |
| `face_width`, `face_height` | int | Face crop dimensions |

---

## Frame Manifests (`datasets/metadata/{final,pilot}_frame_manifest.csv`)

| Column | Type | Description |
|---|---|---|
| `video_id` | str | Unique video identifier |
| `frame_id` | str | Unique frame identifier |
| `frame_index` | int | Frame sequence index |
| `sample_index` | int | Sample index within video |
| `timestamp_sec` | float | Frame timestamp in seconds |
| `frame_path` | str | Path to extracted frame |
| `label` | int | 0 = real, 1 = fake |
| `split` | str | `train`, `val`, or `test` |
| `manipulation_family` | str | Forgery family |
| `manipulation_type` | str | Specific generator name |
| `identity` | str | Subject identity |
| `source_subset` | str | `final` or `pilot` |
| `video_path` | str | Path to source video |

---

## Detector Scores (`datasets/detector_processed/{final,pilot}_detector_scores.csv`)

Video-level scores (one row per video):

| Column | Type | Description |
|---|---|---|
| `video_id` | str | Unique video identifier |
| `label` | int | 0 = real, 1 = fake |
| `split` | str | `train`, `val`, or `test` |
| `manipulation_family` | str | Forgery family |
| `manipulation_type` | str | Specific generator name |
| `n_face_frames` | int | Number of frames with detected faces |
| `video_score_mode` | str | Aggregation mode (e.g. `mean`) |
| `detector_score` | float | Video-level detection score [0,1] |
| `detector_pred` | int | Predicted label at threshold 0.5 |

Frame-level scores (`{final,pilot}_frame_detector_scores.csv`):

| Column | Type | Description |
|---|---|---|
| `video_id` | str | Unique video identifier |
| `frame_id` | str | Unique frame identifier |
| `face_id` | str | Unique face crop identifier |
| `face_path` | str | Path to face crop image |
| `label` | int | 0 = real, 1 = fake |
| `split` | str | `train`, `val`, or `test` |
| `timestamp_sec` | float | Frame timestamp in seconds |
| `manipulation_family` | str | Forgery family |
| `manipulation_type` | str | Specific generator name |
| `detector_score` | float | Frame-level detection score [0,1] |
| `detector_pred` | int | Predicted label at threshold 0.5 |

---

## Frame-Level Emotion Predictions (`datasets/emotion_annotated/metadata/{final,pilot}_frame_emotion_predictions.csv`)

| Column | Type | Description |
|---|---|---|
| `video_id` | str | Unique video identifier |
| `frame_id` | str | Unique frame identifier |
| `face_id` | str | Unique face crop identifier |
| `face_path` | str | Path to face crop image |
| `timestamp_sec` | float | Frame timestamp in seconds |
| `label` | int | 0 = real, 1 = fake |
| `split` | str | `train`, `val`, or `test` |
| `manipulation_family` | str | Forgery family |
| `manipulation_type` | str | Specific generator name |
| `pred_emotion` | str | Dominant predicted emotion class |
| `pred_emotion_score` | float | Confidence for dominant emotion |
| `valence` | float | Circumplex valence [-1, 1] |
| `arousal` | float | Circumplex arousal [-1, 1] |
| `score_<emotion>` | float | Per-class score for each of 40 emotion categories |

Note: `pred_emotion` maps to the SPEC column `emonet_class`; `valence`/`arousal` map to `emonet_valence`/`emonet_arousal`.

---

## Video-Level Emotion Features (`datasets/emotion_annotated/metadata/{final,pilot}_video_emotion_features.csv`)

| Column | Type | Description |
|---|---|---|
| `video_id` | str | Unique video identifier |
| `label` | int | 0 = real, 1 = fake |
| `split` | str | `train`, `val`, or `test` |
| `manipulation_family` | str | Forgery family |
| `manipulation_type` | str | Specific generator name |
| `n_face_frames` | int | Number of frames with detected faces |
| `dominant_emotion` | str | Most frequent predicted emotion class |
| `mean_valence` | float | Mean frame-level valence |
| `std_valence` | float | Std dev of frame-level valence |
| `mean_arousal` | float | Mean frame-level arousal |
| `std_arousal` | float | Std dev of frame-level arousal |
| `max_arousal` | float | Maximum frame-level arousal |
| `emotion_entropy` | float | Shannon entropy over emotion distribution |
| `transition_rate` | float | Fraction of frames with emotion change |
| `arousal_variation` | float | Mean absolute difference of consecutive arousal values |
| `neutral_ratio` | float | Fraction of frames predicted as emotional_numbness |
| `mean_score_<emotion>` | float | Mean per-class score across frames (40 columns) |

---

## Experiment Result Tables

### Ablation / fusion results (Exp. 02, 03)
`datasets/metadata/final_xception_ablation_results.csv` and variants:

| Column | Description |
|---|---|
| `ablation` or `model` | Configuration label |
| `AUC` | Area under ROC curve |
| `ACC` | Accuracy |
| `F1` | F1 score |
| `Precision` | Precision |
| `Recall` | Recall |

### Subgroup AUC (Exp. 04b, 05)
`datasets/metadata/final_xception_auc_by_arousal.csv`:

| Column | Description |
|---|---|
| `arousal_bin` | Tercile label (low / medium / high) |
| `n` | Count of videos |
| `AUC` | Area under ROC curve |

`datasets/metadata/final_xception_auc_by_emotion.csv`:

| Column | Description |
|---|---|
| `dominant_emotion` | Emotion class name |
| `n` | Count of videos |
| `AUC` | Area under ROC curve |

---

## Schema Alignment Notes

- The SPEC refers to `forgery_family` but actual CSVs use `manipulation_family`. New scripts should use `manipulation_family` as the canonical name.
- The SPEC refers to `video_score` but actual detector CSVs use `detector_score`. Use `detector_score`.
- The SPEC column `emonet_class` corresponds to `pred_emotion` in actual files.
- The `identity` column exists in manifests but may be absent from merged tables; join on `video_id` to recover it.
