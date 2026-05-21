# Exp.15 — Modality-Gated Fusion Network

A three-branch neural network that learns per-video softmax gating weights over
detector, emotion, and quality modalities to classify deepfake videos. The gating
weights are fully interpretable: they show which modality the model relied on for
each individual video at inference time.

This experiment is the final architecture contribution of the thesis. It addresses
the supervisor requirement for a custom model with an interpretable inter-modality
interaction mechanism.

---

## Inputs

All read-only from existing project files (no recomputation):

| File | Purpose |
|---|---|
| `datasets/metadata/final_face_manifest.csv` | Identity, forgery family, bbox stats |
| `datasets/emotion_annotated/metadata/final_video_emotion_features.csv` | 49-d emotion features |
| `datasets/detector_processed/final_ucf_scores.csv` | UCF video-level score |
| `datasets/detector_processed/final_ucf_frame_scores.csv` | Per-frame UCF scores (for variance quality feature) |
| Pilot equivalents of all above | Generalisation evaluation |

---

## Running end-to-end

Run all commands from the **project root** (`deepfake-emotion-robustness/`):

```bash
python scripts/exp15_modality_gated/01_prepare_features.py
python scripts/exp15_modality_gated/02_train_modality_gated.py
python scripts/exp15_modality_gated/03_evaluate_test.py
python scripts/exp15_modality_gated/04_extract_gating_weights.py
python scripts/exp15_modality_gated/05_visualize_gating.py
python scripts/exp15_modality_gated/06_pilot_holdout.py
```

Expected total runtime on A100: ~15–30 minutes for all 5 folds.

---

## Outputs

All outputs are written to `scripts/exp15_modality_gated/outputs/`.

| Path | Description |
|---|---|
| `predictions/final_feature_matrix.parquet` | Merged feature matrix (final) |
| `predictions/pilot_feature_matrix.parquet` | Merged feature matrix (pilot) |
| `predictions/final_exp15_oof_predictions.csv` | OOF predictions with gate weights per video |
| `predictions/pilot_exp15_predictions.csv` | Pilot holdout predictions |
| `checkpoints/fold_{k}/best.pt` | Best model per fold |
| `checkpoints/fold_{k}/last.pt` | Latest checkpoint (for resume) |
| `checkpoints/fold_{k}/state.json` | Training state (epoch, AUC, patience) |
| `checkpoints/fold_{k}/DONE` | Sentinel: fold completed |
| `tables/final_exp15_results.csv` | AUC, F1, EER comparison table |
| `tables/final_exp15_gating_per_forgery.csv` | Mean gate weights per forgery family |
| `tables/final_exp15_gating_per_emotion.csv` | Mean gate weights per dominant emotion |
| `tables/final_exp15_per_video_gating.csv` | Top-10 examples per dominant modality |
| `figures/final_exp15_gating_per_forgery.png` | Stacked bar: forgery families |
| `figures/final_exp15_gating_per_emotion.png` | Stacked bar: emotion classes |
| `figures/final_exp15_roc_overlay.png` | ROC comparison: UCF, Exp.12, Exp.15 |
| `figures/final_exp15_modality_dominance_examples.png` | Scatter of extreme gating examples |
| `figures/final_exp15_training_curves.png` | 2×2 training convergence figure |
| `stats/final_exp15_delong_vs_ucf_only.json` | DeLong test vs UCF baseline |
| `stats/final_exp15_delong_vs_ucf_quality.json` | DeLong test vs Exp.12 (if available) |
| `stats/final_exp15_permutation_tests.json` | Permutation tests (10 000 iterations) |
| `logs/run.log` | Full execution log with timestamps |
| `logs/training_curves.csv` | Per-epoch metrics for all folds |
| `tensorboard/fold_{k}/` | TensorBoard event files |

All `.csv` tables also have a `.tex` sibling for direct thesis inclusion.

---

## Resume / fault tolerance

Training (`02_train_modality_gated.py`) supports full resume:

- If a fold has a `DONE` sentinel it is skipped entirely.
- If `last.pt` exists and config hash matches, training resumes from the saved epoch with full RNG state restoration.
- If `last.pt` is corrupted, falls back to `best.pt` (loses only the post-best epochs).
- If config changes after a completed fold, that fold is kept; incomplete folds restart from scratch.

### Full reset

```bash
rm -rf scripts/exp15_modality_gated/outputs/checkpoints/
rm -rf scripts/exp15_modality_gated/outputs/predictions/
rm    scripts/exp15_modality_gated/outputs/logs/training_curves.csv
python scripts/exp15_modality_gated/02_train_modality_gated.py
```

### Reset single fold

```bash
rm -rf scripts/exp15_modality_gated/outputs/checkpoints/fold_2/
python scripts/exp15_modality_gated/02_train_modality_gated.py
```

---

## TensorBoard

```bash
tensorboard --logdir scripts/exp15_modality_gated/outputs/tensorboard/ --port 6006
```

---

## Sanity-check ranges

| Metric | Expected range | If outside |
|---|---|---|
| Final OOF AUC | 0.85 – 0.93 | >0.95 → overfitting; <0.80 → featurisation broken |
| Pilot AUC | 0.90 – 0.97 | Check pilot/train identity overlap |
| Per-fold training time | < 10 min | Stop, something is wrong |

---

## Architecture

`model.py: ModalityGatedFusion` — ~17K parameters.

Three embedding branches (detector scalar → 1→16, emotion 49→64→16, quality 4→16)
feed a gating head (48→32→3) that produces per-video softmax weights. Final logit
is the weighted sum of three branch logits.

---

## Troubleshooting

- **`FileNotFoundError`** on any input — the required source file is missing. Check
  that all upstream experiments have been run.
- **`Missing column in feature matrix`** — column names in the CSVs changed. Update
  `config.yaml → emotion_feature_cols` or `quality_feature_cols` to match.
- **CUDA OOM** — reduce `batch_size` in `config.yaml` (default 32; try 16).
- **AUC = NaN** — a fold's validation set has only one class. Increase `n_folds`
  or check `identity` column integrity in the face manifest.
