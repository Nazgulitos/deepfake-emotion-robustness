"""Frame-level emotion annotation scaffolding."""

from __future__ import annotations

import pandas as pd


def annotate_faces(
    face_manifest: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Create placeholder frame-level emotion predictions.

    TODO: Integrate pretrained FER model inference.
    """
    rows: list[dict[str, object]] = []
    for _, item in face_manifest.iterrows():
        rows.append(
            {
                "video_id": str(item["video_id"]),
                "frame_idx": -1,
                "emotion_class": "unknown",
                "valence": pd.NA,
                "arousal": pd.NA,
                "confidence": pd.NA,
                "model_name": model_name,
                "status": "todo_infer",
            }
        )

    return pd.DataFrame(rows)
