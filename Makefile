PYTHONPATH_ENV = PYTHONPATH=.
PY = .venv/bin/python
CONFIG ?= configs/base.yaml
DATE := $(shell date +%Y-%m-%d)

.PHONY: help build-subset extract-frames extract-faces annotate-emotion aggregate-emotion detect merge evaluate late-fusion all \
        huggingface transformer exp04b exp05 exp06 exp07 exp08 exp09 exp10 exp11 all-new thesis-artifacts test

help:
	@echo "Available targets:"
	@echo "  build-subset       Build curated subset manifest"
	@echo "  extract-frames     Extract frames from subset videos"
	@echo "  extract-faces      Detect/crop faces from frames"
	@echo "  annotate-emotion   Run frame-level emotion annotation"
	@echo "  aggregate-emotion  Aggregate video-level emotion features"
	@echo "  detect             Run baseline deepfake detector"
	@echo "  merge              Merge metadata tables"
	@echo "  evaluate           Evaluate detector by emotion conditions"
	@echo "  late-fusion        Run simple late-fusion baseline"
	@echo "  all                Run the full pipeline"

build-subset:
	$(PYTHONPATH_ENV) $(PY) scripts/build_subset.py --config $(CONFIG)

extract-frames:
	$(PYTHONPATH_ENV) $(PY) scripts/extract_frames.py --config $(CONFIG)

extract-faces:
	$(PYTHONPATH_ENV) $(PY) scripts/extract_faces.py --config $(CONFIG)

annotate-emotion:
	$(PYTHONPATH_ENV) $(PY) scripts/run_emotion_annotation.py --config $(CONFIG)

aggregate-emotion:
	$(PYTHONPATH_ENV) $(PY) scripts/aggregate_emotion_features.py --config $(CONFIG)

detect:
	$(PYTHONPATH_ENV) $(PY) scripts/run_deepfake_detector.py --config $(CONFIG)

merge:
	$(PYTHONPATH_ENV) $(PY) scripts/merge_metadata.py --config $(CONFIG)

evaluate:
	$(PYTHONPATH_ENV) $(PY) scripts/evaluate_by_emotion.py --config $(CONFIG)

late-fusion:
	$(PYTHONPATH_ENV) $(PY) scripts/run_late_fusion.py --config $(CONFIG)

all: build-subset extract-frames extract-faces annotate-emotion aggregate-emotion detect merge evaluate late-fusion

# ---------------------------------------------------------------------------
# New experiment targets
# ---------------------------------------------------------------------------

huggingface:
	$(PYTHONPATH_ENV) $(PY) scripts/run_huggingface_detector.py --subset final
	$(PYTHONPATH_ENV) $(PY) scripts/run_huggingface_detector.py --subset pilot

transformer:
	$(PYTHONPATH_ENV) $(PY) scripts/run_transformer_detector.py --subset final

exp04b:
	$(PYTHONPATH_ENV) $(PY) scripts/evaluate_by_emotion.py --exp_id exp04b --subset final

exp05:
	$(PYTHONPATH_ENV) $(PY) scripts/analyze_per_emotion_auc.py --exp_id exp05 --subset final

exp06:
	$(PYTHONPATH_ENV) $(PY) scripts/analyze_forgery_emotion_crosstab.py --exp_id exp06 --subset final

exp07:
	$(PYTHONPATH_ENV) $(PY) scripts/run_statistical_tests.py --exp_id exp07 --subset final

exp08:
	$(PYTHONPATH_ENV) $(PY) scripts/run_late_fusion.py --exp_id exp08 --detector transformer --subset final

exp09:
	$(PYTHONPATH_ENV) $(PY) scripts/analyze_shap.py --exp_id exp09 --subset final

exp10:
	$(PYTHONPATH_ENV) $(PY) scripts/analyze_umap.py --exp_id exp10 --subset final

exp11:
	$(PYTHONPATH_ENV) $(PY) scripts/validate_on_pilot.py --exp_id exp11

all-new: huggingface transformer exp04b exp05 exp06 exp07 exp08 exp09 exp10 exp11

thesis-artifacts:
	$(PYTHONPATH_ENV) $(PY) scripts/build_thesis_artifacts.py --date $(DATE)

test:
	$(PYTHONPATH_ENV) .venv/bin/pytest tests/ -v
