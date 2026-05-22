# SPEC Addendum — Этап 10: Per-Generator Analysis

**Цель:** Прямой ответ на feedback куратора 3:
> "показать на разных дипфейк методах как что работает мне кажется было бы интересно"

Анализ результатов ThreeModality Gated **per individual generator method**, а не per forgery family. Это даёт более глубокую интерпретацию: какая модальность работает на SimSwap vs FSRT vs AniTalker, и почему.

**Контекст:** существующие этапы 01-09 уже выполнены. Модель обучена, OOF + test predictions сохранены. Этот этап только **анализирует** существующие predictions, ничего не переобучает.

**Время:** 15-20 минут.

---

## 1. Расположение

Новый файл в существующей папке:
```
scripts/exp15_three_modality/
└── 10_per_generator_analysis.py
```

Все outputs идут в существующий `outputs/`:
```
outputs/
├── tables/
│   ├── final_exp15_per_generator_stats.csv
│   └── final_exp15_per_generator_top_findings.csv
├── figures/
│   ├── final_exp15_per_generator_heatmap.png
│   ├── final_exp15_per_generator_auc_comparison.png
│   └── final_exp15_per_generator_modality_dominance.png
└── logs/
    └── per_generator_analysis.log
```

---

## 2. Шаг A — Поиск generator column в данных

Скрипт должен **сам найти** в каких колонках хранится generator method. Не предполагать имя колонки.

```python
import pandas as pd
from pathlib import Path

# Try multiple potential sources
candidates = [
    "datasets/metadata/final_face_manifest.csv",
    "datasets/metadata/final_video_manifest.csv",
    "datasets/metadata/dataset_index.csv",
    "scripts/exp15_three_modality/outputs/predictions/final_feature_matrix.parquet",
]

for path in candidates:
    if not Path(path).exists():
        continue
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)
    print(f"\n=== {path} ===")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample row: {df.iloc[0].to_dict()}")
```

**Что искать:** колонки с одним из имён (case-insensitive):
- `generator`, `method`, `generation_method`, `forgery_method`
- `synthesis_method`, `gen_method`, `model`
- Любая колонка где для fake-видео встречаются значения типа "SimSwap", "AniTalker", "FSRT", "GHOST" и т.д.

### Fallback A1: если generator column есть

Использовать её напрямую. Перейти к Шагу B.

### Fallback A2: если generator column отсутствует но есть file path

Часто generator method закодирован в имени файла:
- `Celeb-synthesis_FaceReenact_FSRT_id17_id0_0003_75dcb963.mp4` → generator = "FSRT"
- `Celeb-synthesis_FaceSwap_SimSwap_id05_id12_0002.mp4` → generator = "SimSwap"

Скрипт парсит filename через regex:

```python
import re

KNOWN_GENERATORS = {
    # FaceSwap
    "SimSwap", "BlendFace", "GHOST", "InSwapper", "HifiFace",
    "MobileFaceSwap", "UniFace", "Celeb-DF-v2", "CelebDF",
    # FaceReenact
    "DaGAN", "FSRT", "HyperReenact", "LIA", "LivePortrait",
    "MCNET", "TPSMM", "TPS-MM",
    # TalkingFace
    "AniTalker", "EDTalk", "EchoMimic", "FLOAT", "IP_LAP",
    "IP-LAP", "Real3DPortrait", "SadTalker",
}

def parse_generator_from_path(path):
    parts = str(path).replace("\\", "/").split("/")
    name = parts[-1]
    # Try matching known generators against path parts and filename
    for gen in KNOWN_GENERATORS:
        if re.search(rf"(?i)\b{re.escape(gen)}\b", str(path)):
            return gen
    return "Unknown"

df["generator"] = df["video_id"].apply(parse_generator_from_path)
# Or use filename column if available
```

### Fallback A3: если ничего не работает

Печатает сообщение и останавливается:
```
ERROR: Could not infer generator method from data.
Please check that face manifest contains a column with generator names
(e.g. 'method', 'generator', 'forgery_method'), or that video_id paths 
contain generator names matching KNOWN_GENERATORS.

Available columns in face_manifest:
  [list]

Sample video_ids:
  [first 5]
```

---

## 3. Шаг B — Build per-generator dataset

После того как у каждого fake video есть generator label:

```python
# Load OOF predictions (от этапа 02)
oof = pd.read_csv("outputs/predictions/final_exp15_oof_predictions.csv")

# Load test predictions
test = pd.read_csv("outputs/predictions/final_exp15_test_predictions.csv")

# Load UCF predictions for comparison
ucf_test = pd.read_csv("datasets/detector_processed/final_ucf_scores.csv")

# Merge generator info
oof = oof.merge(generator_map, on="video_id", how="left")
test = test.merge(generator_map, on="video_id", how="left")
ucf_test = ucf_test.merge(generator_map, on="video_id", how="left")

# Combine OOF + test for full per-generator coverage (635 + 155 = 790 fake-eligible videos)
combined = pd.concat([oof, test])
```

---

## 4. Шаг C — Per-generator statistics

Для каждого generator считаем:

| Column | Description |
|---|---|
| `generator` | Имя метода |
| `forgery_family` | FaceSwap / FaceReenact / TalkingFace |
| `n_fake` | Количество fake видео этого метода |
| `n_real_paired` | Количество real видео в тех же fold'ах (для AUC) |
| `auc_threemodality` | AUC ThreeModality на этом subset |
| `auc_ucf` | AUC UCF baseline на этом subset |
| `delta_auc` | auc_threemodality - auc_ucf |
| `mean_gate_q` | Mean quality gate weight |
| `mean_gate_s` | Mean emotion_static gate weight |
| `mean_gate_t` | Mean emotion_temporal gate weight |
| `dominant_modality` | Argmax({q, s, t}) |
| `mean_pred_score` | Mean ThreeModality score on these fakes |

**AUC расчёт per generator:**

Per-generator AUC требует **real samples в той же fold**. Берём так:

```python
def compute_per_generator_auc(combined_df, generator_name):
    gen_fakes = combined_df[
        (combined_df["generator"] == generator_name) & 
        (combined_df["label"] == 1)
    ]
    
    # Use real samples from the SAME folds as these fakes
    relevant_folds = gen_fakes["fold"].unique() if "fold" in combined_df.columns else None
    
    if relevant_folds is not None:
        reals = combined_df[
            (combined_df["label"] == 0) &
            (combined_df["fold"].isin(relevant_folds))
        ]
    else:
        # Use all real samples
        reals = combined_df[combined_df["label"] == 0]
    
    if len(gen_fakes) < 3 or len(reals) < 3:
        return np.nan
    
    subset = pd.concat([gen_fakes, reals])
    return roc_auc_score(subset["label"], subset["prediction"])
```

**Фильтр:** Generators с `n_fake < 5` исключаются из таблицы (но печатаются в лог отдельно). У тебя должно быть около 20-25 видео per generator после 1000-видео curation.

---

## 5. Шаг D — Сохранение таблиц

### Таблица 1: `final_exp15_per_generator_stats.csv`

22 строки (по одной per generator). Отсортирована по `forgery_family`, потом по `delta_auc` DESC.

```csv
generator,forgery_family,n_fake,n_real_paired,auc_threemodality,auc_ucf,delta_auc,mean_gate_q,mean_gate_s,mean_gate_t,dominant_modality,mean_pred_score
SimSwap,FaceSwap,21,...,0.943,0.380,+0.563,0.18,0.62,0.20,emotion_static,0.76
BlendFace,FaceSwap,22,...,0.918,0.420,+0.498,0.22,0.58,0.20,emotion_static,0.71
...
AniTalker,TalkingFace,24,...,0.987,0.890,+0.097,0.87,0.05,0.08,quality,0.92
SadTalker,TalkingFace,24,...,0.973,0.880,+0.093,0.92,0.04,0.04,quality,0.91
...
```

### Таблица 2: `final_exp15_per_generator_top_findings.csv`

Для интересных historical findings — топ generators по разным критериям:

```csv
finding,generator,family,detail
biggest_improvement_over_ucf,SimSwap,FaceSwap,delta_auc=+0.563
hardest_for_ucf,SimSwap,FaceSwap,ucf_auc=0.380
easiest_for_threemodality,AniTalker,TalkingFace,auc=0.987
most_temporal_dominant,LivePortrait,FaceReenact,gate_t=0.71
most_static_dominant,SimSwap,FaceSwap,gate_s=0.62
most_quality_dominant,SadTalker,TalkingFace,gate_q=0.92
```

---

## 6. Шаг E — Figures

### Figure 1: `final_exp15_per_generator_heatmap.png`

Heatmap **22 generators × 3 modalities** показывающий mean gate weight.

- Rows: 22 generators, sorted by family then by name
- Columns: 3 modalities (Quality, Emotion-Static, Emotion-Temporal)
- Values: mean gate weight on this generator's fake videos
- Color: white (0) to dark blue (1) или sequential cmap "viridis"
- Annotations: точные значения внутри cells
- Visual separators: горизонтальные линии между forgery families
- Right side: small bar показывающая `n_fake` per row

```python
fig, ax = plt.subplots(figsize=(8, 14))
sns.heatmap(
    pivot_data,  # 22 x 3 matrix
    annot=True, fmt=".2f",
    cmap="viridis",
    vmin=0, vmax=1,
    cbar_kws={"label": "Mean Gate Weight"},
    ax=ax
)
# Add horizontal lines between forgery families
ax.axhline(y=family_break_1, color='black', linewidth=2)
ax.axhline(y=family_break_2, color='black', linewidth=2)
plt.title("Modality dominance per generator method")
```

### Figure 2: `final_exp15_per_generator_auc_comparison.png`

Bar chart **22 generators × 2 bars** (ThreeModality vs UCF).

- X axis: generators, sorted by forgery family
- Y axis: AUC, range [0, 1]
- Two bars per generator side-by-side: ThreeModality (blue), UCF (grey)
- Forgery family separators
- Horizontal line at AUC=0.5 (random)
- Annotations: ΔAUC above each generator group

```python
fig, ax = plt.subplots(figsize=(16, 6))
# Width-staggered bars
x = np.arange(len(generators))
width = 0.35
ax.bar(x - width/2, stats["auc_ucf"], width, label="UCF", color="grey")
ax.bar(x + width/2, stats["auc_threemodality"], width, label="ThreeModality", color="steelblue")
ax.set_xticks(x)
ax.set_xticklabels(stats["generator"], rotation=45, ha="right")
ax.axhline(0.5, color="red", linestyle="--", alpha=0.5)
ax.set_ylabel("AUC")
ax.set_title("Per-generator AUC: ThreeModality vs UCF baseline")
ax.legend()
```

### Figure 3: `final_exp15_per_generator_modality_dominance.png`

Stacked horizontal bars — для каждого generator показывает breakdown gate weights (как фигура 2 из этапа 05, но более детально):

- Y axis: 22 generators, sorted by forgery family
- X axis: gate weight (0-1, stacked)
- Three colored segments per row: quality (green), static (orange), temporal (purple)
- Annotations: точные значения inside segments

---

## 7. Шаг F — Логирование

```python
import logging
logging.basicConfig(
    filename="outputs/logs/per_generator_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

logger.info(f"Generator column source: {generator_column_source}")
logger.info(f"Total unique generators found: {len(stats)}")
logger.info(f"Generators included (n_fake >= 5): {len(stats[stats['n_fake'] >= 5])}")
logger.info(f"Generators excluded (n_fake < 5): {excluded_generators}")
```

Также в consoler печатает summary:

```
================================================================
Per-Generator Analysis — Summary
================================================================
Total generators analyzed: 22
Average ThreeModality AUC across generators: 0.94
Average UCF AUC across generators: 0.71
Average ΔAUC (ThreeModality - UCF): +0.23

Top 3 biggest improvements over UCF:
  1. SimSwap         | family=FaceSwap     | ΔAUC=+0.563
  2. BlendFace       | family=FaceSwap     | ΔAUC=+0.498
  3. UniFace         | family=FaceSwap     | ΔAUC=+0.412

Modality dominance by family:
  FaceSwap     → emotion_static dominates (mean=0.55)
  FaceReenact  → mixed (no single dominant)
  TalkingFace  → quality dominates (mean=0.88)

All outputs saved to outputs/tables/ and outputs/figures/
================================================================
```

---

## 8. Acceptance criteria

1. Скрипт находит generator method для всех 22 generators
2. Per-generator stats CSV содержит 22 строк (или меньше, если какие-то excluded по n<5)
3. AUC computed для каждого generator с ≥5 fakes
4. Все 3 figures сгенерированы и читаемы
5. Console summary printed
6. Log file записан

---

## 9. Edge cases

- **Generator имя имеет вариации:** "TPS-MM" vs "TPSMM" vs "TPS_MM" — normalize all to canonical form
- **Hyphenated paths:** "Real3DPortrait" vs "Real-3D-Portrait" — case-insensitive matching
- **External subsets:** do not add non-final subsets to this analysis; the thesis result uses only final trainval OOF plus final test predictions.

---

## 10. После завершения

Пришли мне содержимое:
1. `final_exp15_per_generator_stats.csv` (целиком — 22 строки маленькие)
2. `final_exp15_per_generator_top_findings.csv`
3. `outputs/logs/per_generator_analysis.log` (на случай если есть warnings)
4. Все 3 figures как PNG

С этими данными я смогу написать:
- **Section 4.13 в Chapter 4** — Per-generator analysis (~2 страницы)
- **Section 5.9 в Chapter 5** — Method-specific detection strategies (~1.5 страницы)
- **Один новый money slide** для защиты с heatmap
