"""Tests for src/emotion/aggregation.py — descriptor math on hand-crafted input."""

import math

import numpy as np
import pandas as pd
import pytest

from src.emotion.aggregation import aggregate_emotion_features


def _make_frames(video_id: str, emotions: list[str], arousals: list[float], valences: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "video_id": video_id,
        "emotion_class": emotions,
        "arousal": arousals,
        "valence": valences,
    })


def test_dominant_emotion_is_mode():
    frames = _make_frames("v1", ["happy", "happy", "sad"], [0.1, 0.2, 0.3], [0.5, 0.6, 0.4])
    result = aggregate_emotion_features(frames)
    assert result.loc[0, "dominant_emotion"] == "happy"


def test_mean_arousal_correct():
    frames = _make_frames("v1", ["happy", "sad"], [0.2, 0.4], [0.5, 0.5])
    result = aggregate_emotion_features(frames)
    assert result.loc[0, "mean_arousal"] == pytest.approx(0.3)


def test_max_arousal_correct():
    frames = _make_frames("v1", ["happy", "sad"], [0.2, 0.9], [0.5, 0.5])
    result = aggregate_emotion_features(frames)
    assert result.loc[0, "max_arousal"] == pytest.approx(0.9)


def test_neutral_ratio_all_neutral():
    frames = _make_frames("v1", ["neutral", "neutral"], [0.1, 0.1], [0.0, 0.0])
    result = aggregate_emotion_features(frames, neutral_label="neutral")
    assert result.loc[0, "neutral_ratio"] == pytest.approx(1.0)


def test_neutral_ratio_none_neutral():
    frames = _make_frames("v1", ["happy", "sad"], [0.1, 0.2], [0.3, 0.4])
    result = aggregate_emotion_features(frames, neutral_label="neutral")
    assert result.loc[0, "neutral_ratio"] == pytest.approx(0.0)


def test_entropy_uniform_is_higher_than_concentrated():
    uniform_frames = _make_frames("v1", ["a", "b", "c", "d"], [0.1] * 4, [0.0] * 4)
    concentrated_frames = _make_frames("v2", ["a", "a", "a", "b"], [0.1] * 4, [0.0] * 4)
    result_u = aggregate_emotion_features(uniform_frames)
    result_c = aggregate_emotion_features(concentrated_frames)
    assert result_u.loc[0, "emotion_entropy"] > result_c.loc[0, "emotion_entropy"]


def test_transition_rate_no_change_is_zero():
    frames = _make_frames("v1", ["happy", "happy", "happy"], [0.1] * 3, [0.5] * 3)
    result = aggregate_emotion_features(frames)
    assert result.loc[0, "emotion_transition_rate"] == pytest.approx(0.0)


def test_transition_rate_all_change():
    frames = _make_frames("v1", ["a", "b", "c", "d"], [0.1] * 4, [0.5] * 4)
    result = aggregate_emotion_features(frames)
    assert result.loc[0, "emotion_transition_rate"] == pytest.approx(1.0)


def test_multiple_videos():
    frames = pd.concat([
        _make_frames("v1", ["happy", "sad"], [0.3, 0.7], [0.5, 0.2]),
        _make_frames("v2", ["angry", "angry"], [0.8, 0.9], [0.1, 0.1]),
    ], ignore_index=True)
    result = aggregate_emotion_features(frames)
    assert len(result) == 2
    assert set(result["video_id"]) == {"v1", "v2"}


def test_missing_column_raises():
    frames = pd.DataFrame({"video_id": ["v1"], "emotion_class": ["happy"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        aggregate_emotion_features(frames)
