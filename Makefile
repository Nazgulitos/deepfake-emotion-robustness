PYTHONPATH_ENV = PYTHONPATH=.
PY = python
CONFIG ?= configs/base.yaml

.PHONY: help build-subset extract-frames extract-faces annotate-emotion aggregate-emotion detect merge evaluate late-fusion all

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
