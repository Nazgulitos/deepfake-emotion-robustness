# Exp.15 v2 — Three-Modality Gated Fusion

Neural network with learnable per-video gating over three semantically distinct modalities:

| Modality | What it captures |
|---|---|
| **Quality (M_q)** | Face detection confidence, size, frame count — static technical artefacts |
| **Emotion Static (M_s)** | Mean valence/arousal, 40-category mean scores — aggregated semantic content |
| **Emotion Temporal (M_t)** | Arousal variation, transition rate, entropy, 40-category std scores — dynamics over time |

## Architecture

Each modality → MLP branch → scalar branch logit.  
A gating head (concat of 3 embeddings → softmax) produces per-video weights over the three branches.  
Final logit = weighted sum of branch logits.

~25K parameters. Trained with GroupKFold (5 folds by identity) + early stopping.

## Run end-to-end

```bash
# From project root: deepfake-emotion-robustness/
python scripts/exp15_three_modality/01_prepare_features.py
python scripts/exp15_three_modality/02_train_three_modality.py
python scripts/exp15_three_modality/03_evaluate_test.py
python scripts/exp15_three_modality/04_extract_gating_weights.py
python scripts/exp15_three_modality/05_visualize_gating.py
python scripts/exp15_three_modality/07_ablation_modality_removal.py
python scripts/exp15_three_modality/08_interaction_analysis.py
```

Resume is automatic — each fold has `DONE` sentinel; re-running skips completed folds.

## Key outputs

| File | Description |
|---|---|
| `outputs/predictions/trainval_oof_predictions.csv` | Out-of-fold predictions for all 635 trainval videos |
| `outputs/predictions/test_exp15_predictions.csv` | Test holdout predictions + per-video gate weights |
| `outputs/tables/final_exp15_results.csv` | OOF + test AUC, comparison vs UCF baseline |
| `outputs/tables/final_exp15_ablation_summary.csv` | Ablation: per-modality removal impact |
| `outputs/tables/final_exp15_interaction_pairs.csv` | Pairwise modality interaction analysis |
| `outputs/figures/final_exp15_gating_per_forgery.png` | Gate weights per forgery family |
| `outputs/figures/final_exp15_roc_overlay.png` | ROC curves vs UCF baseline |
| `outputs/stats/final_exp15_delong_vs_ucf.json` | DeLong's test result |
