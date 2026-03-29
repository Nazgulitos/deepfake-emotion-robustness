"""Face detection and crop extraction scaffolding."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def extract_faces_from_frames(
    frame_manifest: pd.DataFrame,
    face_output_dir: Path,
    detector_name: str,
) -> pd.DataFrame:
    """Prepare face manifest rows from frame-manifest inputs.

    TODO: Implement actual face detection and crop writing.
    """
    face_output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for _, item in frame_manifest.iterrows():
        video_id = str(item["video_id"])
        face_dir = face_output_dir / video_id
        face_dir.mkdir(parents=True, exist_ok=True)

        rows.append(
            {
                "video_id": video_id,
                "frame_dir": str(item["frame_dir"]),
                "face_dir": str(face_dir),
                "face_detector": detector_name,
                "num_face_crops": 0,
                "status": "todo_detect",
            }
        )

    return pd.DataFrame(rows)
