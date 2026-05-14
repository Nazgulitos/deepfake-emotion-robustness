"""Tests for src/utils/splits.py — identity-disjoint split guarantees."""

import numpy as np
import pandas as pd
import pytest

from src.utils.splits import assert_identities_disjoint, group_kfold_splits, identity_disjoint_split


def _make_df(n_identities: int = 30, videos_per_identity: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    identities = [f"id_{i:03d}" for i in range(n_identities)]
    rows = []
    for iid in identities:
        for _ in range(videos_per_identity):
            rows.append({"video_id": f"{iid}_v{_}", "identity": iid, "label": rng.integers(0, 2)})
    return pd.DataFrame(rows)


def test_no_identity_appears_in_two_splits():
    df = _make_df()
    out = identity_disjoint_split(df)
    assert_identities_disjoint(out)  # raises if violated


def test_all_rows_have_split_assigned():
    df = _make_df()
    out = identity_disjoint_split(df)
    assert out["split"].notna().all()
    assert set(out["split"]).issubset({"train", "val", "test"})


def test_train_is_largest_split():
    df = _make_df(n_identities=20, videos_per_identity=10)
    out = identity_disjoint_split(df, fracs=(0.7, 0.15))
    counts = out["split"].value_counts()
    assert counts["train"] > counts.get("val", 0)
    assert counts["train"] > counts.get("test", 0)


def test_group_kfold_yields_correct_n_folds():
    df = _make_df(n_identities=10, videos_per_identity=5)
    folds = list(group_kfold_splits(df, n_splits=5))
    assert len(folds) == 5


def test_group_kfold_no_group_overlap():
    df = _make_df(n_identities=10, videos_per_identity=5)
    for train_idx, val_idx in group_kfold_splits(df, n_splits=5):
        train_groups = set(df.iloc[train_idx]["identity"])
        val_groups = set(df.iloc[val_idx]["identity"])
        assert train_groups.isdisjoint(val_groups), "Identity overlap in GroupKFold fold"


def test_assert_identities_disjoint_raises_on_overlap():
    df = pd.DataFrame({
        "video_id": ["a", "b", "c"],
        "identity": ["id_1", "id_1", "id_2"],
        "split": ["train", "test", "train"],
    })
    with pytest.raises(AssertionError, match="overlap"):
        assert_identities_disjoint(df)


def test_assert_identities_disjoint_passes_clean():
    df = pd.DataFrame({
        "video_id": ["a", "b", "c"],
        "identity": ["id_1", "id_2", "id_3"],
        "split": ["train", "val", "test"],
    })
    assert_identities_disjoint(df)  # should not raise
