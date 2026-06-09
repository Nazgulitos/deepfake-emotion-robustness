UV = uv run
CONFIG ?= configs/base.yaml
DATE := $(shell date +%Y-%m-%d)

.PHONY: help build-subset extract-frames extract-faces annotate-emotion aggregate-emotion detect merge evaluate late-fusion all \
	huggingface transformer exp04b exp05 exp06 exp07 exp08 exp09 exp10 exp11 all-new thesis-artifacts test setup

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
	@echo "  setup              Sync the uv environment"

setup:
	uv sync

build-subset:
	$(UV) python scripts/build_subset.py --config $(CONFIG)

extract-frames:
	$(UV) python scripts/extract_frames.py --config $(CONFIG)

extract-faces:
	$(UV) python scripts/extract_faces.py --config $(CONFIG)

annotate-emotion:
	$(UV) python scripts/run_emotion_annotation.py --config $(CONFIG)

aggregate-emotion:
	$(UV) python scripts/aggregate_emotion_features.py --config $(CONFIG)

detect:
	$(UV) python scripts/run_deepfake_detector.py --config $(CONFIG)

merge:
	$(UV) python scripts/merge_metadata.py --config $(CONFIG)

evaluate:
	$(UV) python scripts/evaluate_by_emotion.py --config $(CONFIG)

late-fusion:
	$(UV) python scripts/run_late_fusion.py --config $(CONFIG)

all: build-subset extract-frames extract-faces annotate-emotion aggregate-emotion detect merge evaluate late-fusion

# ---------------------------------------------------------------------------
# New experiment targets
# ---------------------------------------------------------------------------

huggingface:
	$(UV) python scripts/run_huggingface_detector.py --subset final
	$(UV) python scripts/run_huggingface_detector.py --subset pilot

transformer:
	$(UV) python scripts/run_transformer_detector.py --subset final

exp04b:
	$(UV) python scripts/evaluate_by_emotion.py --exp_id exp04b --subset final

exp05:
	$(UV) python scripts/analyze_per_emotion_auc.py --exp_id exp05 --subset final

exp06:
	$(UV) python scripts/analyze_forgery_emotion_crosstab.py --exp_id exp06 --subset final

exp07:
	$(UV) python scripts/run_statistical_tests.py --exp_id exp07 --subset final

exp08:
	$(UV) python scripts/run_late_fusion.py --exp_id exp08 --detector transformer --subset final

exp09:
	$(UV) python scripts/analyze_shap.py --exp_id exp09 --subset final

exp10:
	$(UV) python scripts/analyze_umap.py --exp_id exp10 --subset final

exp11:
	$(UV) python scripts/validate_on_pilot.py --exp_id exp11

all-new: huggingface transformer exp04b exp05 exp06 exp07 exp08 exp09 exp10 exp11

thesis-artifacts:
	$(UV) python scripts/build_thesis_artifacts.py --date $(DATE)

test:
	$(UV) pytest tests/ -v
