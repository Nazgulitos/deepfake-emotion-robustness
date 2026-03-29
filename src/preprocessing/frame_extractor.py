"""Frame extraction stage scaffolding.

This module intentionally provides a minimal, configurable structure.
Replace TODO sections with the chosen extraction backend (OpenCV/ffmpeg).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def extract_frames_from_manifest(
    subset_manifest: pd.DataFrame,
    frame_output_dir: Path,
    fps: float,
) -> pd.DataFrame:
    """Create frame-manifest rows for each video.

    TODO: Implement actual extraction and set num_frames accordingly.
    """
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for _, item in subset_manifest.iterrows():
        video_id = str(item["video_id"])
        video_path = Path(str(item["video_path"]))
        frame_dir = frame_output_dir / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        rows.append(
            {
                "video_id": video_id,
                "video_path": str(video_path),
                "frame_dir": str(frame_dir),
                "fps": fps,
                "num_frames": 0,
                "status": "todo_extract",
            }
        )

    return pd.DataFrame(rows)
