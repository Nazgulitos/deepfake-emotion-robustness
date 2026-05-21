# SPEC: Three-Modality Gated Fusion Network — Experiment 15 (v2)

**Цель:** обучить и оценить собственную нейросетевую архитектуру для multimodal deepfake detection, которая моделирует взаимодействие между **тремя семантически разными модальностями** — quality-сигналом, эмоциональной динамикой (статической) и временными признаками — через интерпретируемый gating-механизм, показывающий на каждом видео какая модальность доминирует.

**Контекст:** этот SPEC замещает предыдущую версию (Exp.15 с detector branch). Предыдущая архитектура показала что gate weight detector ≈ 0.004 — то есть detector signal избыточен когда есть emotion + quality. Новая архитектура полностью убирает detector branch и заменяет его на третью семантическую модальность — **temporal dynamics features** — которая прямо отвечает на запрос научного руководителя:

> "штуку которая моделирует взаимодействие между quality-сигналом, эмоциональной динамикой и **другими временными фичами** с интерпретируемым механизмом, который показывает, какая модальность на каком видео работает"

**Окружение:** A100 GPU, PyTorch, scikit-learn, pandas, numpy. Если что-то отсутствует — `pip install --break-system-packages`. Все детерминированно (seed=42).

---

## 1. Архитектурный концепт

Три модальности **семантически разные** и **минимально перекрывающиеся**:

| Модальность | Что описывает | Природа сигнала | Примеры features |
|---|---|---|---|
| **M_q: Quality** | Как снято/сгенерировано видео | Static appearance / technical artefacts | face_det_score_mean, face_size_mean, blur_mean, frame_count |
| **M_s: Emotion (Static)** | Что выражает лицо в среднем | Semantic content (aggregated) | mean_arousal, max_arousal, mean_valence, dominant_emotion one-hot, 40-cat EMONET-FACE BIG means |
| **M_t: Emotion (Temporal)** | Как меняется эмоция во времени | Dynamics (per-video variability) | arousal_variation, transition_rate, emotion_entropy, neutral_ratio, 40-cat std/min/max/gradient |

**Зачем это правильно:**
- M_q и M_s ортогональны: первая про артефакты пайплайна, вторая про семантику лица
- M_s и M_t разные репрезентации одного и того же сигнала: одна агрегатная, другая динамическая. Это позволяет gating решать "на этом видео важнее **что** выражено или **как** оно меняется"
- M_q и M_t тоже различны: quality artefacts не меняются по времени, temporal features именно про изменение

Гейтинг учится **за видео** выбирать какая из трёх модальностей релевантнее.

---

## 2. Структура папки

Новый эксперимент идёт в новой папке (старый Exp.15 не трогаем):

```
scripts/exp15_three_modality/
├── README.md
├── config.yaml
├── 01_prepare_features.py
├── 02_train_three_modality.py
├── 03_evaluate_test.py
├── 04_extract_gating_weights.py
├── 05_visualize_gating.py
├── 06_pilot_holdout.py
├── 07_ablation_modality_removal.py     # NEW: per-modality ablation
├── 08_interaction_analysis.py          # NEW: pairwise modality interaction
├── model.py
├── dataset.py
├── utils.py
└── outputs/
    ├── checkpoints/
    │   ├── fold_0/  ... fold_4/
    │   └── ablation/
    │       ├── no_quality/
    │       ├── no_emotion_static/
    │       └── no_emotion_temporal/
    ├── predictions/
    │   ├── final_exp15_oof_predictions.csv
    │   ├── final_exp15_test_predictions.csv
    │   ├── pilot_exp15_predictions.csv
    │   ├── final_feature_matrix.parquet
    │   └── pilot_feature_matrix.parquet
    ├── tables/
    │   ├── final_exp15_results.csv
    │   ├── final_exp15_ablation_summary.csv
    │   ├── final_exp15_gating_per_forgery.csv
    │   ├── final_exp15_gating_per_emotion.csv
    │   ├── final_exp15_gating_per_arousal_tercile.csv
    │   ├── final_exp15_per_video_gating.csv
    │   ├── final_exp15_modality_correlation.csv
    │   └── final_exp15_interaction_pairs.csv
    ├── figures/
    │   ├── final_exp15_gating_per_forgery.png
    │   ├── final_exp15_gating_per_emotion.png
    │   ├── final_exp15_gating_per_arousal.png
    │   ├── final_exp15_modality_dominance_examples.png
    │   ├── final_exp15_roc_overlay.png
    │   ├── final_exp15_training_curves.png
    │   ├── final_exp15_modality_correlation_heatmap.png
    │   └── final_exp15_ablation_bars.png
    ├── stats/
    │   ├── final_exp15_delong_vs_ucf.json
    │   ├── final_exp15_delong_vs_quality_only.json
    │   ├── final_exp15_permutation_full_vs_ablation.json
    │   └── final_exp15_modality_redundancy_test.json
    ├── tensorboard/
    │   └── fold_0/  ... fold_4/
    └── logs/
        ├── run.log
        └── training_curves.csv
```

---

## 3. Входные данные

| Источник | Путь | Что используется |
|---|---|---|
| Face manifest | `datasets/metadata/final_face_manifest.csv` | `video_id`, `label`, `identity`, `forgery_family`, `generator`, det_score per frame |
| Face manifest pilot | `datasets/metadata/pilot_face_manifest.csv` | то же для пилота |
| EmoNet frame-level | `datasets/emotion_annotated/metadata/final_frame_emotion_predictions.csv` | per-frame emotion, valence, arousal |
| EmoNet frame-level pilot | `datasets/emotion_annotated/metadata/pilot_frame_emotion_predictions.csv` | то же |
| Video features (aggregated) | `datasets/emotion_annotated/metadata/final_video_emotion_features.csv` | если уже агрегировано |
| Quality features (если есть) | из выходов Exp.12 | quality features уже посчитаны |
| UCF scores | `datasets/detector_processed/final_ucf_scores.csv` | **только для baseline сравнения, не для модели** |

**Важно:** detector scores больше не входят в модель. Они используются только в скрипте `03_evaluate_test.py` для **сравнения** ModalityGated против UCF в финальной таблице.

Агент должен сначала `view` каждый файл и зафиксировать реальные имена колонок.

---

## 4. Архитектура — ThreeModalityGated

```python
class ThreeModalityGated(nn.Module):
    """
    Three-branch architecture with learnable per-video gating over modalities.
    
    Modalities are semantically distinct:
      M_q: quality features (static technical signals)
      M_s: emotion (static, aggregated semantic content)  
      M_t: emotion (temporal, dynamics over time)
    
    Each branch projects its input to a shared embedding dim.
    A gating head computes per-video softmax weights over the three modalities.
    Each branch produces its own scalar logit. Final prediction is a 
    gated mixture of the three branch logits.
    """
    
    def __init__(self, quality_dim, emo_static_dim, emo_temporal_dim,
                 embed_dim=16, gate_hidden=32, dropout=0.2):
        super().__init__()
        
        # Per-modality embedders (similar capacity)
        self.q_embed = nn.Sequential(
            nn.Linear(quality_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
            nn.ReLU(),
        )
        self.s_embed = nn.Sequential(
            nn.Linear(emo_static_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )
        self.t_embed = nn.Sequential(
            nn.Linear(emo_temporal_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
            nn.ReLU(),
        )
        
        # Per-modality logit heads
        self.q_head = nn.Linear(embed_dim, 1)
        self.s_head = nn.Linear(embed_dim, 1)
        self.t_head = nn.Linear(embed_dim, 1)
        
        # Gating head over concatenated embeddings
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 3),
        )
    
    def forward(self, x_q, x_s, x_t):
        h_q = self.q_embed(x_q)
        h_s = self.s_embed(x_s)
        h_t = self.t_embed(x_t)
        
        z_q = self.q_head(h_q).squeeze(-1)
        z_s = self.s_head(h_s).squeeze(-1)
        z_t = self.t_head(h_t).squeeze(-1)
        
        gate_logits = self.gate(torch.cat([h_q, h_s, h_t], dim=-1))
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        z_stacked = torch.stack([z_q, z_s, z_t], dim=-1)
        z_final = (gate_weights * z_stacked).sum(dim=-1)
        
        return {
            "logit": z_final,
            "gate_weights": gate_weights,
            "branch_logits": z_stacked,
            "embeddings": {"q": h_q, "s": h_s, "t": h_t},
        }
```

**Размер:** ~25K параметров. Безопасно для 800 видео.

**Параметры по умолчанию в `config.yaml`:**

```yaml
seed: 42
device: cuda

# Architecture
embed_dim: 16
gate_hidden: 32
dropout: 0.2

# Training
batch_size: 32
n_epochs: 100
patience: 12
lr: 1e-3
weight_decay: 1e-4
optimizer: adamw
pos_weight_auto: true

# Cross-validation
n_folds: 5
group_col: identity
test_size: 0.2          # отдельный test holdout перед CV
val_size_within_train: 0.15

# Feature partitioning — explicit lists for reproducibility
# Quality modality
quality_features:
  - face_det_score_mean
  - face_det_score_std
  - face_size_mean
  - face_size_std
  - blur_mean
  - frame_count

# Emotion static modality (per-video aggregates of average state)
emotion_static_features:
  - mean_arousal
  - mean_valence
  - max_arousal
  - neutral_ratio
  - dominant_emotion_onehot      # 8-d one-hot of EmoNet dominant class
  - emonetface_mean_*            # 40-d means over EMONET-FACE BIG categories

# Emotion temporal modality (per-video dynamics features)
emotion_temporal_features:
  - arousal_variation
  - transition_rate
  - emotion_entropy
  - arousal_min_max_diff
  - valence_min_max_diff
  - emonetface_std_*             # 40-d temporal std over EMONET-FACE BIG
  - emonetface_gradient_*        # 40-d mean abs first-difference

# Reference baseline for comparison (NOT used as model input)
reference_baseline:
  detector: ucf
  scores_path: datasets/detector_processed/final_ucf_scores.csv

# Paths
paths:
  face_manifest: datasets/metadata/final_face_manifest.csv
  face_manifest_pilot: datasets/metadata/pilot_face_manifest.csv
  frame_emotion: datasets/emotion_annotated/metadata/final_frame_emotion_predictions.csv
  frame_emotion_pilot: datasets/emotion_annotated/metadata/pilot_frame_emotion_predictions.csv
  video_emotion: datasets/emotion_annotated/metadata/final_video_emotion_features.csv
  video_emotion_pilot: datasets/emotion_annotated/metadata/pilot_video_emotion_features.csv
  ucf_scores: datasets/detector_processed/final_ucf_scores.csv
  ucf_scores_pilot: datasets/detector_processed/pilot_ucf_scores.csv
  output_root: scripts/exp15_three_modality/outputs
```

---

## 5. Этапы выполнения

### Этап 01 — `01_prepare_features.py`

**Цель:** построить **три отдельные feature matrices** для трёх модальностей. Каждая матрица — по одной строке на video_id.

**Pipeline:**

1. Загрузить face manifest + frame_emotion CSV.
2. Загрузить video_emotion_features (если есть aggregated) или вычислить заново.
3. **Quality features (per video):**
   ```python
   quality_df = face_manifest.groupby("video_id").agg(
       face_det_score_mean=("det_score", "mean"),
       face_det_score_std=("det_score", "std"),
       face_size_mean=("face_area", "mean"),
       face_size_std=("face_area", "std"),
       blur_mean=("blur", "mean") if "blur" in cols else None,
       frame_count=("frame_index", "count"),
   )
   ```
4. **Emotion static features (per video):**
   - aggregated валентность, arousal — берётся из `video_emotion_features.csv`
   - one-hot of `dominant_emotion`
   - per-category **mean** для 40-cat EMONET-FACE BIG
5. **Emotion temporal features (per video):**
   - `arousal_variation`, `transition_rate`, `emotion_entropy` — из video_emotion_features
   - **std**, **gradient** для 40-cat EMONET-FACE BIG (если не было — посчитать из frame-level)
   - `arousal_min_max_diff = max_arousal - min_arousal`
6. **Соединить** все три матрицы по `video_id` через outer join, отфильтровать видео с NaN.
7. Сохранить в `outputs/predictions/final_feature_matrix.parquet` с колонками:
   - `video_id`, `label`, `identity`, `forgery_family`, `dominant_emotion`
   - quality columns (~6)
   - emotion_static columns (~50)
   - emotion_temporal columns (~85)
8. Повторить для pilot.

**Проверки в конце:**
- Печатает размерности каждой модальности
- Печатает correlation matrix между **первыми моментами** каждой модальности — должна показывать что модальности минимально перекрываются
- Сохраняет `final_exp15_modality_correlation.csv` и `final_exp15_modality_correlation_heatmap.png`

**Самопроверка:** ожидаемо что correlation между средними features quality и emotion_static — около |0.1-0.3| (не выше). Если выше — модальности перекрываются и архитектура работать не будет.

### Этап 02 — `02_train_three_modality.py`

Тренировка с честным split:
1. Сначала отделить **20% test holdout** по identity (identity-disjoint).
2. На оставшихся 80% — 5-fold GroupKFold по identity.
3. Внутри каждого fold — train/val split тоже by identity.
4. Все остальные параметры — как в предыдущем SPEC.

**Loss:** `BCEWithLogitsLoss(pos_weight=...)` где pos_weight авто.

**Сохранение checkpoint'ов:** `best.pt`, `last.pt`, `state.json`, `DONE` per fold — как раньше.

**Resume mechanism:** полный, как раньше (atomic writes, RNG state, config hash).

**После всех 5 fold:** обучить **final model** на всём trainval (80%) с гиперпараметрами усреднёнными по folds, оценить на test holdout (20%).

### Этап 03 — `03_evaluate_test.py`

Считает на **trainval OOF predictions** и **test holdout predictions** отдельно:

- AUC + 2000-iter bootstrap CI
- ACC, F1, Precision, Recall, EER
- DeLong vs UCF only
- DeLong vs Quality-only (читает quality_only ablation если уже посчитан)
- Permutation test full vs ablation

Сохраняет `outputs/tables/final_exp15_results.csv`:

```csv
split, model, AUC, AUC_ci_low, AUC_ci_high, ACC, F1, Precision, Recall, EER, n
trainval_oof, three_modality_full, ..., 635
test_holdout, three_modality_full, ..., 155
trainval_oof, ucf_only, ..., 635
test_holdout, ucf_only, ..., 155
```

### Этап 04 — `04_extract_gating_weights.py`

Берёт OOF + test predictions. Группирует gate weights:

- **Per forgery_family** → `final_exp15_gating_per_forgery.csv`
- **Per dominant emotion** (только n≥10) → `final_exp15_gating_per_emotion.csv`
- **Per arousal tercile** (low/mid/high) → `final_exp15_gating_per_arousal_tercile.csv`
- **Top 10 quality-dominant** + **Top 10 static-dominant** + **Top 10 temporal-dominant** → `final_exp15_per_video_gating.csv` для слайдов

### Этап 05 — `05_visualize_gating.py`

Восемь figures (PNG, dpi=300, prefix `final_exp15_`):

**Figure 1: Per-forgery modality dominance** (3 строки × 3 модальности stacked bars)

**Figure 2: Per-emotion modality dominance** (emotion classes с n≥10)

**Figure 3: Per-arousal modality dominance** (low/mid/high terciles)

**Figure 4: Modality dominance examples** (3 субплота с правильно отнормализованными bubbles — исправить баг из предыдущей версии где bubble size был 0 или 1):
- Top-10 quality dominant видео
- Top-10 emotion-static dominant
- Top-10 emotion-temporal dominant
Каждая точка: position по двум gate weights, color = label (real/fake), label = forgery_family. Bubble size = log(frame_count).

**Figure 5: ROC overlay** — три линии:
- UCF only (test holdout)
- ThreeModality (test holdout)
- ThreeModality (trainval OOF)

**Figure 6: Training curves** — 2x2 subplot (loss, AUC, best val AUC progression, gate weights evolution)

**Figure 7: Modality correlation heatmap** — сгенерирован в этапе 01

**Figure 8: Ablation bars** (после этапа 07) — AUC при удалении каждой модальности

### Этап 06 — `06_pilot_holdout.py`

Применяет финальную модель (обученную на trainval) на pilot subset без retraining. Сохраняет AUC + metrics.

### Этап 07 — `07_ablation_modality_removal.py` (**NEW**)

Это критическая часть для интерпретации.

Для каждой из трёх модальностей обучает модель **без неё** и сравнивает с full:

```
config A: ThreeModality_full              (q + s + t)
config B: ThreeModality_no_quality        (s + t only, 2-way gating)
config C: ThreeModality_no_emotion_static (q + t only)
config D: ThreeModality_no_emotion_temporal (q + s only)
```

Для каждого config — те же 5 folds + test holdout. Считает AUC + permutation test vs full.

Сохраняет `outputs/tables/final_exp15_ablation_summary.csv`:

```csv
config, trainval_oof_auc, test_auc, delta_vs_full, permutation_p
full,            0.XXX, 0.XXX,  0.000,  ---
no_quality,      0.XXX, 0.XXX, -0.XXX,  X.XXe-XX
no_emotion_static, 0.XXX, 0.XXX, -0.XXX, X.XXe-XX
no_emotion_temporal, 0.XXX, 0.XXX, -0.XXX, X.XXe-XX
```

**Сильный finding:** если каждая модальность даёт значительный вклад (Δ > 0.02 и p < 0.05) — все три семантически релевантны. Если какая-то даёт Δ ≈ 0 — она избыточна.

### Этап 08 — `08_interaction_analysis.py` (**NEW**)

Анализ **взаимодействий между парами модальностей**. Это прямо адресует фразу куратора *"моделирует взаимодействие между..."*.

Для каждого видео в test holdout считает:

```python
interaction_qs = gate_q * gate_s    # joint contribution quality+static
interaction_qt = gate_q * gate_t    # joint quality+temporal
interaction_st = gate_s * gate_t    # joint static+temporal
```

Затем смотрит на **корреляции interaction terms с детектерными ошибками**:

- Корреляция `interaction_qs` с правильностью предсказания (binary)
- Корреляция `interaction_qt` с правильностью
- Корреляция `interaction_st` с правильностью

Сохраняет `final_exp15_interaction_pairs.csv`:

```csv
pair, mean_interaction, std_interaction, spearman_with_correctness, p_value
q × s, 0.XXX, 0.XXX, 0.XXX, X.XXe-XX
q × t, 0.XXX, 0.XXX, 0.XXX, X.XXe-XX
s × t, 0.XXX, 0.XXX, 0.XXX, X.XXe-XX
```

Также находит **топ-5 видео где quality × temporal максимально** (то есть оба сигнала работают сильно) и **топ-5 где они конфликтуют** (один высокий, другой низкий). Это для интерпретации в Discussion.

---

## 6. README.md

```markdown
# Exp.15 v2 — Three-Modality Gated Fusion

## Run end-to-end

```bash
cd scripts/exp15_three_modality/
python 01_prepare_features.py
python 02_train_three_modality.py
python 03_evaluate_test.py
python 04_extract_gating_weights.py
python 05_visualize_gating.py
python 06_pilot_holdout.py
python 07_ablation_modality_removal.py
python 08_interaction_analysis.py
```

## Modalities

1. **Quality** — static technical features (face detection confidence, sharpness, frame count)
2. **Emotion Static** — aggregated emotion content (mean valence/arousal, dominant emotion)
3. **Emotion Temporal** — emotion dynamics over time (variation, transition rate, entropy, gradients)

Each modality is processed by its own MLP branch and produces a scalar logit.
A learnable gating head produces per-video softmax weights over the three branches.
```

---

## 7. Acceptance criteria

1. Все 8 scripts отрабатывают без ошибок
2. Test holdout AUC лежит в диапазоне 0.88–0.97
3. Modality correlation matrix (этап 01) показывает что три модальности минимально перекрываются
4. Ablation table показывает что **каждая** из трёх модальностей даёт значимый вклад (если какая-то даёт Δ ≈ 0 — это тоже finding, но менее идеальный)
5. Gating distribution **варьируется** по forgery_family и dominant_emotion (т.е. не все веса равны 0.33)
6. Interaction analysis возвращает осмысленные числа (mean interaction term должен лежать в [0.02, 0.25])
7. Training curves показывают correct convergence (loss падает, val AUC растёт до плато, gating weights эволюционируют от 0.33/0.33/0.33)
8. Resume test passed: при искусственном прерывании после fold 2 повторный запуск продолжает с fold 3

---

## 8. Что НЕ делать

- Не добавлять detector branch (это и есть главное отличие от v1)
- Не использовать UCF score как input в модель (только для baseline сравнения в этапе 03)
- Не модифицировать существующие файлы вне `scripts/exp15_three_modality/`
- Не пытаться использовать pilot для tuning
- Не пропускать ablation (этап 07) — это критическая часть интерпретации

---

## 9. Защитный narrative

После завершения эксперимента у тебя должны быть три статьи для защиты:

**Finding 1:** Three semantically distinct modalities — quality, emotion-static, emotion-temporal — each contribute statistically significantly to deepfake detection (ablation Δ > 0.02 with p < 0.05 for each).

**Finding 2:** The per-video gating mechanism reveals which modality dominates on which video. On TalkingFace forgeries with high arousal, emotion-temporal dominates. On FaceSwap with neutral expressions, quality dominates. On videos with rapid emotion transitions, emotion-temporal becomes critical.

**Finding 3:** Modality interaction patterns are informative: pairs of modalities with high interaction values (q × t, s × t) correlate with detection correctness, suggesting that **interaction between modalities** carries more signal than any modality alone.

Эти три finding'a напрямую отвечают на запрос научного руководителя.

---

## 10. Время

- Этапы 01-02 (prepare + train 5 folds): 1-1.5 часа
- Этапы 03-06 (evaluation + visualization + pilot): 30 минут
- Этап 07 (ablation × 3 configs × 5 folds): 1.5-2 часа
- Этап 08 (interaction analysis): 15 минут

**Всего: 3-4 часа compute time.**

---

## 11. Resume и monitoring (как в предыдущем SPEC)

Все правила из предыдущего SPEC v1 остаются:
- Per-fold atomic checkpoints (best.pt, last.pt, state.json, DONE)
- TensorBoard logging
- CSV training_curves.csv per epoch
- Console output каждые 5 эпох
- Per-fold summary после завершения
- RNG state restoration
- Config hash verification
- Corruption recovery через backup checkpoints
