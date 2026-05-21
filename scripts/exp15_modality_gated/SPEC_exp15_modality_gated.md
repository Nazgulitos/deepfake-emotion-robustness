# SPEC: Modality-Gated Fusion Network — Experiment 15

**Цель:** обучить и оценить собственную нейросетевую архитектуру для multimodal deepfake detection, которая (1) интерпретируемо показывает вклад каждой модальности на каждом видео и (2) служит ответом на feedback научного руководителя и куратора о необходимости "своей модели" с интерпретируемым механизмом взаимодействия между quality, emotion и detector сигналами.

**Контекст для агента:** диплом защищается через ~6 дней. Этот эксперимент — финальное дополнение. Любая ошибка должна остановить выполнение с понятным сообщением, а не молча проглатываться. Всё детерминированно (seed=42).

**Окружение:** A100 GPU доступен. PyTorch, scikit-learn, pandas, numpy уже стоят. Если чего-то не хватает — `pip install` с `--break-system-packages`.

---

## 1. Структура папки эксперимента

Весь новый код этого эксперимента должен лежать **в одной папке** `scripts/exp15_modality_gated/`. Никаких изменений в существующих модулях `src/`, `scripts/`, `notebooks/` — только новая папка.

```
scripts/exp15_modality_gated/
├── README.md                       # как запускать, что куда читается, куда пишется
├── config.yaml                     # все гиперпараметры в одном месте
├── 01_prepare_features.py          # собирает feature matrix из существующих CSV
├── 02_train_modality_gated.py      # обучение + 5-fold GroupKFold + сохранение модели
├── 03_evaluate_test.py             # OOF predictions + AUC + DeLong + bootstrap CI
├── 04_extract_gating_weights.py    # извлекает gating weights per video
├── 05_visualize_gating.py          # три figures: per-forgery, per-emotion, per-video
├── 06_pilot_holdout.py             # генерализация на pilot subset
├── model.py                        # архитектура ModalityGatedFusion
├── dataset.py                      # PyTorch Dataset + DataLoader helpers
├── utils.py                        # seeds, metrics, plotting helpers
└── outputs/                        # всё что генерируется — сюда (не в /outputs/results)
    ├── checkpoints/
    │   └── best_model.pt
    ├── predictions/
    │   ├── final_exp15_oof_predictions.csv
    │   └── pilot_exp15_predictions.csv
    ├── tables/
    │   ├── final_exp15_results.csv
    │   ├── final_exp15_gating_per_forgery.csv
    │   ├── final_exp15_gating_per_emotion.csv
    │   └── final_exp15_per_video_gating.csv
    ├── figures/
    │   ├── final_exp15_gating_per_forgery.png
    │   ├── final_exp15_gating_per_emotion.png
    │   ├── final_exp15_modality_dominance_examples.png
    │   ├── final_exp15_roc_overlay.png
    │   └── final_exp15_training_curves.png
    ├── stats/
    │   ├── final_exp15_delong_vs_ucf_only.json
    │   ├── final_exp15_delong_vs_ucf_quality.json
    │   └── final_exp15_permutation_tests.json
    ├── tensorboard/                # TensorBoard event files
    │   ├── fold_0/
    │   ├── fold_1/
    │   ├── fold_2/
    │   ├── fold_3/
    │   └── fold_4/
    └── logs/
        ├── run.log
        └── training_curves.csv     # метрики per epoch per fold
```

**Никаких outputs за пределы этой папки.** После эксперимента можно будет одной командой `cp` перенести нужные файлы в основную thesis-структуру.

---

## 2. Входные данные

Все читается **только** из существующих файлов проекта. Ничего не перегенерируется. Если файл не найден — fail loud с понятным сообщением.

| Источник | Путь | Что берём |
|---|---|---|
| Face manifest | `datasets/metadata/final_face_manifest.csv` | `video_id`, `label`, `identity`, `forgery_family` |
| Face manifest pilot | `datasets/metadata/pilot_face_manifest.csv` | то же для пилота |
| Emotion features | `datasets/emotion_annotated/metadata/final_video_emotion_features.csv` | EmoNet 8-d + EMONET-FACE BIG 200-d |
| Emotion features pilot | `datasets/emotion_annotated/metadata/pilot_video_emotion_features.csv` | то же для пилота |
| UCF scores | `datasets/detector_processed/final_ucf_scores.csv` | `video_score` per video |
| UCF scores pilot | `datasets/detector_processed/pilot_ucf_scores.csv` | то же для пилота |
| XceptionNet scores | `datasets/detector_processed/final_detector_scores.csv` | `video_score` |
| HuggingFace scores | `datasets/detector_processed/final_huggingface_scores.csv` | `video_score` |
| Quality features | если есть — из выходов Exp.12; если нет — вычислить из face manifest + UCF frame scores | `face_det_score_mean`, `face_size_mean`, `blur_mean`, `frame_count`, `ucf_score_variance` |

**Важно:** агент должен сначала `view` каждый из этих файлов и зафиксировать реальные имена колонок, потом писать код. Не предполагать ничего.

---

## 3. Архитектура — ModalityGatedFusion

Файл: `model.py`.

```python
class ModalityGatedFusion(nn.Module):
    """
    Three-branch architecture with learnable per-video gating over modalities.
    
    Modalities:
      M_d: detector score (scalar input)
      M_e: emotion descriptors (208-dim input)
      M_q: quality features (5-dim input)
    
    Each branch projects its input to a shared embedding dim (e.g. 16).
    A gating head computes per-video softmax weights over the three modalities.
    Each branch also produces its own scalar logit. Final prediction is a 
    gated mixture of the three branch logits.
    """
    
    def __init__(self, emotion_dim=208, quality_dim=5, embed_dim=16, dropout=0.2):
        super().__init__()
        
        # Per-modality embedders
        self.det_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emo_embed = nn.Sequential(
            nn.Linear(emotion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )
        self.qual_embed = nn.Sequential(
            nn.Linear(quality_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Per-modality logit heads (one scalar each)
        self.det_head = nn.Linear(embed_dim, 1)
        self.emo_head = nn.Linear(embed_dim, 1)
        self.qual_head = nn.Linear(embed_dim, 1)
        
        # Gating head — takes concatenated embeddings, produces 3 softmax weights
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )
    
    def forward(self, det, emo, qual):
        # Embeddings
        h_d = self.det_embed(det)         # (B, embed_dim)
        h_e = self.emo_embed(emo)         # (B, embed_dim)
        h_q = self.qual_embed(qual)       # (B, embed_dim)
        
        # Per-modality logits
        z_d = self.det_head(h_d).squeeze(-1)   # (B,)
        z_e = self.emo_head(h_e).squeeze(-1)   # (B,)
        z_q = self.qual_head(h_q).squeeze(-1)  # (B,)
        
        # Gating weights (softmax)
        h_concat = torch.cat([h_d, h_e, h_q], dim=-1)   # (B, embed_dim * 3)
        gate_logits = self.gate(h_concat)               # (B, 3)
        gate_weights = F.softmax(gate_logits, dim=-1)   # (B, 3)
        
        # Final mixed logit (gated combination)
        z_stacked = torch.stack([z_d, z_e, z_q], dim=-1)   # (B, 3)
        z_final = (gate_weights * z_stacked).sum(dim=-1)   # (B,)
        
        return {
            "logit": z_final,
            "gate_weights": gate_weights,
            "branch_logits": z_stacked,
        }
```

**Размер модели:** ~17K параметров. Безопасно для 800 видео.

**Параметры по умолчанию (в `config.yaml`):**

```yaml
seed: 42
device: cuda

# Architecture
embed_dim: 16
dropout: 0.2

# Training
batch_size: 32
n_epochs: 100
patience: 15
lr: 1e-3
weight_decay: 1e-4
optimizer: adamw

# Loss
pos_weight_auto: true   # compute from training set

# Cross-validation
n_folds: 5
group_col: identity

# Feature columns (define explicitly; agent must verify against actual files)
emotion_feature_prefix: ["emonet_", "emonetface_"]
quality_feature_cols:
  - face_det_score_mean
  - face_size_mean
  - blur_mean
  - frame_count
  - ucf_score_variance

# Detector to use as primary input
primary_detector: ucf

# Paths (relative to project root)
paths:
  face_manifest: datasets/metadata/final_face_manifest.csv
  face_manifest_pilot: datasets/metadata/pilot_face_manifest.csv
  emotion_features: datasets/emotion_annotated/metadata/final_video_emotion_features.csv
  emotion_features_pilot: datasets/emotion_annotated/metadata/pilot_video_emotion_features.csv
  ucf_scores: datasets/detector_processed/final_ucf_scores.csv
  ucf_scores_pilot: datasets/detector_processed/pilot_ucf_scores.csv
  output_root: scripts/exp15_modality_gated/outputs
```

---

## 4. Этапы выполнения

### Этап 01 — `01_prepare_features.py`

**Что делает:**
1. Читает face manifest, emotion features, UCF scores.
2. Inner join по `video_id`.
3. Вычисляет quality features. Если они уже посчитаны в Exp.12 — берёт оттуда. Если нет — считает заново:
   - `face_det_score_mean` — среднее по `det_score` колонке в face manifest
   - `face_size_mean` — среднее площади bbox
   - `blur_mean` — если есть колонка с Laplacian — берёт; если нет — пропускает feature
   - `frame_count` — количество строк per video_id
   - `ucf_score_variance` — стандартное отклонение frame-level UCF scores
4. Standardize все continuous features через `StandardScaler` (statistics только по training partition в каждом fold — но на этом этапе сохраняем raw, scaling в training script).
5. Сохраняет `outputs/predictions/final_feature_matrix.parquet` с колонками:
   - `video_id`, `label`, `identity`, `forgery_family`, `dominant_emotion`
   - `detector_score`
   - emotion features (208 колонок с префиксом)
   - quality features (5 колонок)
6. Повторяет для pilot subset → `outputs/predictions/pilot_feature_matrix.parquet`.

**Проверки в конце:**
- Никаких NaN в feature columns
- Размеры совпадают с ожидаемыми (final ≈800, pilot ≈200)
- Печатает `df.info()` обоих feature matrices

### Этап 02 — `02_train_modality_gated.py`

**Что делает:**
1. Читает `final_feature_matrix.parquet`.
2. 5-fold GroupKFold по `identity`. Тот же seed 42.
3. На каждом fold:
   - Разделяет на train/val (in-fold splitting на train portion, 80/20 random with seed)
   - StandardScaler fit на train, apply на val/test fold
   - Создаёт PyTorch Dataset + DataLoader
   - Обучает ModalityGatedFusion с early stopping на val AUC
   - Сохраняет best checkpoint per fold в `outputs/checkpoints/fold_{k}.pt`
   - Сохраняет OOF predictions для test fold в общий список
4. После всех folds:
   - Конкатенирует OOF predictions в `outputs/predictions/final_exp15_oof_predictions.csv` с колонками: `video_id`, `label`, `prediction`, `gate_det`, `gate_emo`, `gate_qual`, `branch_det_logit`, `branch_emo_logit`, `branch_qual_logit`, `forgery_family`, `dominant_emotion`
5. Печатает per-fold AUC + overall OOF AUC

**Loss:** `BCEWithLogitsLoss(pos_weight=...)` где `pos_weight = neg/pos` на training set.

**Early stopping:** на val AUC, patience=15 epochs.

**Логи:** stdout + `outputs/logs/run.log` с timestamp.

### Этап 03 — `03_evaluate_test.py`

**Что делает:**
1. Читает `final_exp15_oof_predictions.csv`.
2. Вычисляет:
   - Overall AUC с 2000-iteration bootstrap CI
   - ACC, F1, Precision, Recall на threshold=0.5
   - EER
3. Сравнивает с:
   - UCF only (читает из `datasets/detector_processed/final_ucf_scores.csv`)
   - UCF+quality (читает из Exp.12 outputs если есть)
4. DeLong's test между ModalityGated и UCF only → `outputs/stats/final_exp15_delong_vs_ucf_only.json`
5. DeLong vs UCF+quality (если доступен) → `outputs/stats/final_exp15_delong_vs_ucf_quality.json`
6. Permutation test (10000 iterations) ModalityGated vs UCF only
7. Сохраняет `outputs/tables/final_exp15_results.csv` с строкой:
   ```
   model, AUC, AUC_ci_low, AUC_ci_high, ACC, F1, Precision, Recall, EER, n
   ```
8. Печатает таблицу в консоль.

### Этап 04 — `04_extract_gating_weights.py`

**Что делает:**
1. Из OOF predictions берёт gate weights per video.
2. Группирует и считает средние gate weights:
   - **Per forgery family** → `final_exp15_gating_per_forgery.csv`:
     ```
     forgery_family, mean_gate_det, mean_gate_emo, mean_gate_qual, n
     ```
   - **Per dominant emotion** → `final_exp15_gating_per_emotion.csv` (с фильтром n≥10):
     ```
     dominant_emotion, mean_gate_det, mean_gate_emo, mean_gate_qual, n
     ```
   - **Top 10 examples где emotion доминирует** + **Top 10 где quality доминирует** → `final_exp15_per_video_gating.csv` (для слайдов / discussion).
3. Печатает summary в консоль.

### Этап 05 — `05_visualize_gating.py`

**Три figure (каждая в PNG, dpi=300, prefix `final_exp15_`):**

**Figure 1: Per-forgery modality dominance** (stacked horizontal bar chart)
```
FaceSwap     [det 0.20 | emo 0.30 | qual 0.50]
FaceReenact  [det 0.35 | emo 0.40 | qual 0.25]
TalkingFace  [det 0.30 | emo 0.45 | qual 0.25]
```
Цвета: detector=синий, emotion=оранжевый, quality=зелёный. Легенда. Title: "Modality contribution by forgery family".

**Figure 2: Per-emotion modality dominance** (тот же стиль, но строки — emotion классы с n≥10)

**Figure 3: ROC overlay** — линии для:
- UCF only (baseline)
- UCF+quality (XGBoost из Exp.12)
- ModalityGated (твоя новая модель)

Все читаются из существующих CSV + новых OOF predictions.

### Этап 06 — `06_pilot_holdout.py`

**Что делает:**
1. Загружает лучший fold checkpoint (по val AUC) — или ensemble из всех 5 folds.
2. Применяет к `pilot_feature_matrix.parquet` без re-training.
3. Сохраняет `outputs/predictions/pilot_exp15_predictions.csv`.
4. Вычисляет AUC + metrics + сравнение с UCF only на pilot.
5. Печатает таблицу `pilot_exp15_results`.

---

## 5. README.md в папке эксперимента

Должен содержать:
- Один параграф что это за эксперимент
- Команды для запуска end-to-end:
  ```bash
  cd scripts/exp15_modality_gated/
  python 01_prepare_features.py
  python 02_train_modality_gated.py
  python 03_evaluate_test.py
  python 04_extract_gating_weights.py
  python 05_visualize_gating.py
  python 06_pilot_holdout.py
  ```
- Описание ожидаемых выходов и их назначение
- Контакт/troubleshooting

---

## 6. Ожидаемые числовые результаты

Не как контрольные точки, а как sanity check. Если результаты сильно отличаются — что-то сломано.

- **Final OOF AUC:** ожидается в диапазоне 0.85–0.93. Если выше 0.95 — overfitting; если ниже 0.80 — что-то в featurization сломано.
- **Per-forgery gating:** ожидается что FaceSwap будет более quality-dominated (поскольку UCF на FaceSwap слаб), TalkingFace — более emotion-dominated.
- **Per-emotion gating:** ожидается что anger/elation будут более emotion-dominated (т.к. AUC baseline на них низкий, modality gating должен полагаться на quality).
- **Pilot AUC:** ожидается 0.90–0.97 (как UCF+emotion+quality в Exp.12).

---

## 7. Reproducibility и логи

Каждый script в начале выполнения:
1. Сетит seeds: `numpy`, `torch`, `random`, `cuda`
2. Записывает в `outputs/logs/run.log` метаданные: timestamp, git commit hash, python version, torch version, какой config был использован, какие пути читались

Все таблицы сохраняются также как `.tex` (через pandas `df.to_latex()`) в дополнение к `.csv` для удобства вставки в thesis.

---

## 8. Что НЕ делать

- Не модифицировать существующие файлы вне `scripts/exp15_modality_gated/`
- Не пересчитывать face detection, emotion annotation, UCF scores — всё уже есть
- Не использовать pilot subset для training или model selection (только final evaluation)
- Не добавлять fine-tuning UCF — это другой эксперимент
- Не пытаться обучать без identity-disjoint split
- Не глотать ошибки молча — если что-то не найдено, fail loud

---

## 9. Ограничение по времени

Максимум 6 часов compute time всего на all stages. Если training одного fold идёт больше 10 минут — что-то не так, остановить и поднять флаг.

---

## 10. После завершения

Выведи в консоль одной summary table:

```
================================================================
Exp.15 — Modality-Gated Fusion Network — Final Summary
================================================================
Final OOF AUC:           X.XXX (95% CI: X.XXX - X.XXX)
vs UCF only:             ΔAUC = +X.XXX (DeLong p = X.XXXe-XX)
vs UCF+quality:          ΔAUC = +X.XXX (DeLong p = X.XXXe-XX)
Pilot AUC:               X.XXX

Mean gating weights (overall):
  detector: X.XX
  emotion:  X.XX
  quality:  X.XX

Per-forgery dominant modality:
  FaceSwap:    [winning_modality]
  FaceReenact: [winning_modality]
  TalkingFace: [winning_modality]

All outputs saved to: scripts/exp15_modality_gated/outputs/
================================================================
```

---

## 11. Checkpointing and Recovery

Тренировка должна выдерживать падения. Если процесс упал на fold 3 из 5 — повторный запуск не должен начинаться с fold 0.

### 11.1 Per-fold checkpointing

В `outputs/checkpoints/` сохраняются:

```
checkpoints/
├── fold_0/
│   ├── best.pt              # лучший по val AUC, сохраняется каждый раз когда обновляется best
│   ├── last.pt              # последняя эпоха (для resume внутри fold)
│   ├── state.json           # epoch, best_val_auc, patience_counter, optimizer state ref
│   └── DONE                 # пустой файл; создаётся только когда fold полностью завершён
├── fold_1/...
└── fold_2/...
```

**Что в `state.json`:**

```json
{
    "fold": 0,
    "current_epoch": 23,
    "best_val_auc": 0.8467,
    "best_epoch": 13,
    "patience_counter": 10,
    "config_hash": "abc123...",   // хеш config.yaml; если меняли config — invalid checkpoint
    "torch_version": "2.1.0",
    "completed": false,
    "timestamp_last_save": "2026-05-20T15:23:11Z"
}
```

**Что в `last.pt`:**

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scaler_state_dict": ...,   # если используется AMP
    "scheduler_state_dict": ..., # если есть scheduler
    "scaler": ...,               # StandardScaler от sklearn (pickled)
    "rng_states": {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate()
    }
}
```

Сохранение RNG состояний нужно для **полной воспроизводимости** при resume.

### 11.2 Resume logic в `02_train_modality_gated.py`

В начале каждого fold:

```python
fold_dir = f"outputs/checkpoints/fold_{k}/"

if os.path.exists(f"{fold_dir}/DONE"):
    logger.info(f"[Fold {k}] Already completed. Skipping.")
    # Load OOF predictions from existing artifact
    continue

if os.path.exists(f"{fold_dir}/state.json"):
    state = json.load(open(f"{fold_dir}/state.json"))
    
    # Verify config didn't change
    current_config_hash = hash_config(config)
    if state["config_hash"] != current_config_hash:
        logger.warning(f"[Fold {k}] Config changed. Restarting fold from scratch.")
        shutil.rmtree(fold_dir)
        os.makedirs(fold_dir)
        start_epoch = 0
        # ... fresh init
    else:
        logger.info(f"[Fold {k}] Resuming from epoch {state['current_epoch']}")
        checkpoint = torch.load(f"{fold_dir}/last.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Restore RNG
        torch.set_rng_state(checkpoint["rng_states"]["torch"])
        torch.cuda.set_rng_state(checkpoint["rng_states"]["cuda"])
        np.random.set_state(checkpoint["rng_states"]["numpy"])
        random.setstate(checkpoint["rng_states"]["python"])
        start_epoch = state["current_epoch"] + 1
        best_val_auc = state["best_val_auc"]
        patience_counter = state["patience_counter"]
else:
    # First time — fresh init
    start_epoch = 0
    best_val_auc = 0.0
    patience_counter = 0
    # Save config hash
    state = {"config_hash": current_config_hash, "fold": k, ...}
```

### 11.3 Сохранение во время обучения

**Каждые N эпох** (по умолчанию N=1 — каждую эпоху):

```python
# Always save last.pt for resume
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "rng_states": {...},
    "scaler": scaler,
}, f"{fold_dir}/last.pt")

# Update state.json
state.update({
    "current_epoch": epoch,
    "best_val_auc": best_val_auc,
    "patience_counter": patience_counter,
    "timestamp_last_save": datetime.utcnow().isoformat() + "Z"
})
json.dump(state, open(f"{fold_dir}/state.json", "w"), indent=2)

# Save best.pt only when val AUC improves
if val_auc > best_val_auc:
    best_val_auc = val_auc
    state["best_epoch"] = epoch
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
        "val_auc": val_auc,
        "epoch": epoch,
    }, f"{fold_dir}/best.pt")
    patience_counter = 0
else:
    patience_counter += 1
```

**Atomic save:** записывать в `state.json.tmp` потом `os.rename` чтобы избежать corruption если процесс упадёт во время записи.

### 11.4 После завершения fold

```python
# Mark fold as complete
open(f"{fold_dir}/DONE", "w").close()

# Save OOF predictions for this fold immediately
oof_predictions_this_fold.to_csv(f"{fold_dir}/oof_predictions.csv", index=False)
```

Это критично: если падение случится **после** последнего fold, но **до** consolidation OOF — у нас всё равно есть per-fold предсказания.

### 11.5 Consolidation в конце

В конце `02_train_modality_gated.py`:

```python
# Verify all folds completed
for k in range(n_folds):
    if not os.path.exists(f"outputs/checkpoints/fold_{k}/DONE"):
        raise RuntimeError(f"Fold {k} did not complete. Resume the script.")

# Concatenate per-fold OOF predictions
all_oof = pd.concat([
    pd.read_csv(f"outputs/checkpoints/fold_{k}/oof_predictions.csv")
    for k in range(n_folds)
])
all_oof.to_csv("outputs/predictions/final_exp15_oof_predictions.csv", index=False)
```

### 11.6 Восстановление при corruption

Если `last.pt` загружается с ошибкой (corrupted) — попытаться загрузить `best.pt` и начать с эпохи `best.pt.epoch + 1`. Это потеря небольшого progress, но не всего fold.

```python
try:
    checkpoint = torch.load(f"{fold_dir}/last.pt")
except Exception as e:
    logger.warning(f"[Fold {k}] last.pt corrupted: {e}. Trying best.pt...")
    try:
        checkpoint = torch.load(f"{fold_dir}/best.pt")
        start_epoch = checkpoint["epoch"] + 1
    except Exception as e2:
        logger.error(f"[Fold {k}] Both checkpoints corrupted. Restarting fold.")
        shutil.rmtree(fold_dir)
        os.makedirs(fold_dir)
        start_epoch = 0
```

### 11.7 Резервная копия

Каждый раз когда сохраняется `best.pt`, копировать также в `outputs/checkpoints/backup/fold_{k}_best.pt`. Защита от case где checkpoint corrupted во время записи.

### 11.8 Команда полного сброса

В README документировать:

```bash
# Reset everything and start training from scratch
rm -rf scripts/exp15_modality_gated/outputs/checkpoints/
rm -rf scripts/exp15_modality_gated/outputs/predictions/
rm -rf scripts/exp15_modality_gated/outputs/logs/training_curves.csv
python 02_train_modality_gated.py

# Reset single fold (e.g. if it diverged)
rm -rf scripts/exp15_modality_gated/outputs/checkpoints/fold_2/
python 02_train_modality_gated.py
```

### 11.9 Behavior summary

| Сценарий | Что произойдёт при повторном запуске |
|---|---|
| Скрипт упал на fold 0 epoch 5 | Resume fold 0 с epoch 6 |
| Скрипт упал на fold 3 epoch 17 | Skip folds 0-2, resume fold 3 с epoch 18 |
| Скрипт убит после fold 2 | Skip folds 0-2, start fold 3 from scratch |
| Все 5 folds завершены | Skip training, run consolidation only |
| Config изменили после fold 1 | Fold 0 уже complete (skip), fold 1 restart from scratch |
| `last.pt` corrupted | Try `best.pt`, lose только последние эпохи |
| OOM на fold 0 | Падает с понятным сообщением, ничего не сохраняется |

### 11.10 Гарантии

- **Никаких "тихих" перезаписей.** Если конфиг изменился — явное warning.
- **Atomic writes** через `.tmp` + `os.rename` для всех state файлов.
- **Per-fold isolation** — падение одного fold не вредит другим.
- **Полная воспроизводимость** при resume — RNG state восстанавливается.

---

## 12. Monitoring Loss and Metrics During Training

В дополнение к выводу в stdout/log, training script должен логировать метрики **в трёх форматах**:

### 11.1 TensorBoard

Использовать `torch.utils.tensorboard.SummaryWriter`.

**Логируемые метрики per epoch per fold:**
- `train/loss` — средний BCE loss на эпохе
- `train/auc` — AUC на training partition
- `val/loss` — BCE loss на validation
- `val/auc` — AUC на validation
- `val/auc_best` — лучшее val AUC до текущей эпохи (для отслеживания early stopping)

**Логируемые скаляры per fold:**
- `fold_{k}/best_val_auc`
- `fold_{k}/test_oof_auc`
- `fold_{k}/n_epochs_trained`

**Логируемые histograms per epoch:**
- `gate_weights/detector` — распределение detector gate по val batch
- `gate_weights/emotion`
- `gate_weights/quality`

Это покажет **как gating развивается во время обучения** — очень полезно для интерпретации.

**Папка:** `scripts/exp15_modality_gated/outputs/tensorboard/`

**Как смотреть:**
```bash
tensorboard --logdir scripts/exp15_modality_gated/outputs/tensorboard/ --port 6006
```

### 11.2 CSV-лог метрик per epoch

В дополнение к TensorBoard писать чистый CSV для удобной обработки и вставки в thesis:

`outputs/logs/training_curves.csv` со строками:

```csv
fold,epoch,train_loss,train_auc,val_loss,val_auc,val_auc_best,lr,gate_det_mean,gate_emo_mean,gate_qual_mean
0,1,0.6234,0.5821,0.6101,0.6234,0.6234,0.001,0.34,0.33,0.33
0,2,0.5891,0.6234,0.5723,0.6789,0.6789,0.001,0.36,0.31,0.33
...
```

Это позволяет потом построить training curves в matplotlib для thesis.

### 11.3 Final training curves figure

После завершения training (в конце `02_train_modality_gated.py`) автоматически сгенерировать figure:

`outputs/figures/final_exp15_training_curves.png` — 2x2 subplot:
- (a) train/val loss per epoch для каждого fold (5 линий каждого цвета)
- (b) train/val AUC per epoch для каждого fold
- (c) Best val AUC progression — показывает где сработал early stopping для каждого fold
- (d) Mean gate weights через epochs — как сеть учится распределять веса между модальностями

Этот figure можно вставить в thesis Chapter 4 как доказательство правильного обучения.

### 11.4 Console output per epoch

Каждые 5 эпох в stdout печатать:

```
[Fold 1] Epoch  5/100 | train_loss=0.5123 train_auc=0.7234 | val_loss=0.4987 val_auc=0.7456 | gate=[d:0.28 e:0.41 q:0.31] | lr=1.00e-03
[Fold 1] Epoch 10/100 | train_loss=0.4234 train_auc=0.8012 | val_loss=0.4456 val_auc=0.8234 | gate=[d:0.25 e:0.42 q:0.33] | lr=1.00e-03
[Fold 1] Epoch 15/100 | train_loss=0.3891 train_auc=0.8456 | val_loss=0.4321 val_auc=0.8389 | gate=[d:0.24 e:0.43 q:0.33] | lr=1.00e-03
...
[Fold 1] Early stopping at epoch 28 | best_val_auc=0.8467 at epoch 13
```

При раннем останове показывать на какой эпохе он сработал и какой best val AUC.

### 11.5 Per-fold summary после каждого fold

После того как fold обучился — печатать compact summary:

```
================================
[Fold 1/5] Training complete
================================
  Best val AUC:       0.8467 (epoch 13)
  Test OOF AUC:       0.8523
  Epochs trained:     28/100
  Total time:         4m 32s
  Mean gate weights:  det=0.24  emo=0.43  qual=0.33
================================
```

### 11.6 Минимальные dependencies

Скорее всего уже стоит. На всякий случай в начале `02_train_modality_gated.py`:

```python
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")
```

Если TensorBoard недоступен — продолжать без него, но CSV-лог и figure обязательны.

---

## 13. Acceptance criteria

Эксперимент считается успешно выполненным, если:
1. Все 6 scripts отрабатывают без ошибок
2. OOF AUC лежит в ожидаемом диапазоне (0.85–0.93)
3. Per-forgery gating таблица заполнена для всех 3 семейств
4. Все три figures сгенерированы и читаемы
5. Final summary table напечатана
6. Все файлы в `outputs/` присутствуют согласно структуре в §1
7. Training curves figure (`final_exp15_training_curves.png`) сгенерирован и показывает корректные convergence patterns
8. `training_curves.csv` содержит метрики per epoch для всех 5 folds
9. **Resume test passed:** при искусственном прерывании после fold 2 и повторном запуске — folds 0-2 пропускаются, обучение продолжается с fold 3 без сброса прогресса