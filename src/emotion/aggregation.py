"""Aggregate frame-level emotions into video-level descriptors."""

from __future__ import annotations

import math

import pandas as pd


def _entropy(labels: pd.Series) -> float:
    if labels.empty:
        return float("nan")
    probs = labels.value_counts(normalize=True)
    return float(-(probs * probs.map(lambda p: math.log(p + 1e-12))).sum())


def _transition_rate(labels: pd.Series) -> float:
    if len(labels) <= 1:
        return 0.0
    transitions = (labels.shift(1) != labels).sum() - 1
    transitions = max(transitions, 0)
    return float(transitions / (len(labels) - 1))


def aggregate_emotion_features(
    frame_predictions: pd.DataFrame,
    neutral_label: str = "neutral",
) -> pd.DataFrame:
    """Compute required video-level emotional descriptors."""
    required = {"video_id", "emotion_class", "valence", "arousal"}
    missing = required - set(frame_predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    features: list[dict[str, float | str]] = []
    for video_id, group in frame_predictions.groupby("video_id"):
        labels = group["emotion_class"].dropna().astype(str)
        arousal = pd.to_numeric(group["arousal"], errors="coerce")
        valence = pd.to_numeric(group["valence"], errors="coerce")

        dominant_emotion = labels.mode().iloc[0] if not labels.empty else "unknown"

        features.append(
            {
                "video_id": str(video_id),
                "dominant_emotion": dominant_emotion,
                "mean_valence": float(valence.mean()) if not valence.empty else float("nan"),
                "mean_arousal": float(arousal.mean()) if not arousal.empty else float("nan"),
                "max_arousal": float(arousal.max()) if not arousal.empty else float("nan"),
                "emotion_entropy": _entropy(labels),
                "emotion_transition_rate": _transition_rate(labels.reset_index(drop=True)),
                "arousal_variation": float(arousal.std()) if not arousal.empty else float("nan"),
                "neutral_ratio": float((labels == neutral_label).mean()) if not labels.empty else 0.0,
            }
        )

    return pd.DataFrame(features)
