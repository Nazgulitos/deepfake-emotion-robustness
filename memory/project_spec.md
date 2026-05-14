---
name: Thesis project SPEC and status
description: SPEC_v1.md goals, PR sequence, and what has been completed vs. pending
type: project
---

Thesis: deepfake detection robustness under emotion conditions. Repo: deepfake-emotion-robustness/.

**Why:** Master's thesis with 2-week deadline (from 2026-05-13). SPEC_v1.md defines all work.

## PR sequence (from SPEC §14)

| PR | Scope | Status |
|---|---|---|
| 1 | Cleanup, tests skeleton, configs/experiments.yaml, docs/data_schema.md | **DONE** (2026-05-13) |
| 2 | src/utils/splits.py, src/utils/run_metadata.py, src/analysis/statistical_tests.py | **DONE** (included in PR 1) |
| 3 | scripts/run_huggingface_detector.py → final_huggingface_scores.csv | TODO |
| 4 | Verify Exp.01–03 reproduce within 1% AUC | TODO |
| 5 | Exp.04b + Exp.05 (per-emotion subgroup) | TODO |
| 6 | Exp.06 (forgery × emotion cross-tab) | TODO |
| 7 | Exp.07 (DeLong, Spearman statistical tests) | TODO |
| 8 | Exp.08 (transformer detector + fusion) | TODO |
| 9 | Exp.09 (SHAP) + Exp.10 (UMAP) | TODO |
| 10 | Exp.11 (pilot holdout) + build_thesis_artifacts.py | TODO |

## Key schema notes (actual CSV columns differ from SPEC)
- SPEC `forgery_family` → actual `manipulation_family`
- SPEC `video_score` → actual `detector_score`
- SPEC `emonet_class` → actual `pred_emotion`
- `identity` is in manifests; join on `video_id` to recover in merged tables

## New files added in PR 1
- configs/experiments.yaml
- docs/data_schema.md
- src/analysis/__init__.py, subgroup_auc.py, statistical_tests.py, shap_helpers.py, umap_helpers.py
- src/utils/splits.py, run_metadata.py
- tests/test_metrics.py, test_splits.py, test_aggregation.py, test_statistical_tests.py
- Notebooks renamed: *__pilot.ipynb / *__final.ipynb convention
- Makefile: added exp04b–exp11, all-new, thesis-artifacts, test targets
- outputs/results/, outputs/thesis_artifacts/ directories

**How to apply:** Next PRs should pick up at PR 3 (run_huggingface_detector.py). Always use `manipulation_family` not `forgery_family`. Run `make test` to verify 33 tests pass.
