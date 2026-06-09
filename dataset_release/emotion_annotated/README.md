# Emotion-Annotated Dataset Release

This folder is the canonical release package for the emotion-annotated dataset artifact produced during the thesis.

Purpose
- Provide a compact, user-friendly handoff of the processed dataset (frame-level emotion predictions and video-level features) without including large raw media.
- Document how to install, verify, and cite the dataset.

What belongs here
- Processed CSV/Parquet artifacts (frame-level predictions, video-level features, manifests).
- A `DATASET_MANIFEST.csv` describing files included in the release (checksums, brief descriptions).
- A short `LICENSE` file (recommended: CC BY 4.0).

Layout
```
dataset_release/emotion_annotated/
  DATASET_MANIFEST.csv
  LICENSE
  README.md  <-- this file
  final_frame_emotion_predictions.csv
  final_video_emotion_features.csv
  final_face_manifest.csv
```

Quick usage
1. Download or copy the released artifact into the repository (outside raw media):

```bash
# unpack into datasets/emotion_annotated or symlink
mkdir -p datasets/emotion_annotated/metadata
# if you received a tarball: tar -xzf emotion_annotated_v1.tar.gz -C datasets/emotion_annotated/metadata
```

2. Point the repository config to the dataset location in `configs/paths.local.yaml`.
3. Run analysis or model code referencing `datasets/emotion_annotated/metadata`.

Packaging and hosting suggestions
- Do NOT store raw videos in GitHub. Host raw media on Zenodo, OSF, or institutional storage and provide a download manifest.
- For a release, create a compressed archive (no raw media) and upload to Zenodo to obtain a DOI.

Packaging example (from repository root):

```bash
# package processed artifacts only
tar -czf emotion_annotated_v1.tar.gz -C datasets/emotion_annotated metadata
sha256sum emotion_annotated_v1.tar.gz > emotion_annotated_v1.tar.gz.sha256
```

Citation
Please cite the thesis main report included in this repository (`PROJECT_REPORT.pdf`) and include the dataset DOI when published.

Contact
If you need access to raw media or the full processing pipeline, open an issue or email the project maintainer.
