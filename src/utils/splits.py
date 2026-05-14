"""GroupKFold helpers ensuring identity-disjoint train/val/test splits."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def identity_disjoint_split(
    df: pd.DataFrame,
    identity_col: str = "identity",
    split_col: str = "split",
    fracs: tuple[float, float] = (0.7, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/val/test splits so no identity appears in more than one split.

    Args:
        df: Input dataframe with at least identity_col rows.
        identity_col: Column holding subject identity.
        split_col: Name of the output column to add/overwrite.
        fracs: (train_frac, val_frac); remainder goes to test.
        seed: Random seed for reproducibility.

    Returns:
        Copy of df with split_col set.
    """
    train_frac, val_frac = fracs
    rng = np.random.default_rng(seed)
    identities = np.array(df[identity_col].dropna().unique(), dtype=object)
    rng.shuffle(identities)

    n = len(identities)
    n_train = int(np.floor(n * train_frac))
    n_val = int(np.floor(n * val_frac))

    train_ids = set(identities[:n_train])
    val_ids = set(identities[n_train : n_train + n_val])

    def _assign(row: pd.Series) -> str:
        iid = row[identity_col]
        if iid in train_ids:
            return "train"
        if iid in val_ids:
            return "val"
        return "test"

    out = df.copy()
    out[split_col] = df.apply(_assign, axis=1)
    return out


def group_kfold_splits(
    df: pd.DataFrame,
    group_col: str = "identity",
    n_splits: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) pairs from GroupKFold on group_col.

    Guarantees that rows with the same group are never split across train/val.

    Args:
        df: Input dataframe.
        group_col: Column to use as groups.
        n_splits: Number of folds.

    Yields:
        Pairs of integer index arrays (train indices, val indices).
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col].values
    x_dummy = np.zeros(len(df))
    for train_idx, val_idx in gkf.split(x_dummy, groups=groups):
        yield train_idx, val_idx


def assert_identities_disjoint(
    df: pd.DataFrame,
    identity_col: str = "identity",
    split_col: str = "split",
) -> None:
    """Raise AssertionError if any identity appears in more than one split."""
    cross = df.groupby(identity_col)[split_col].nunique()
    overlap = cross[cross > 1]
    if not overlap.empty:
        raise AssertionError(
            f"Identity/split overlap detected for {len(overlap)} identities: "
            f"{overlap.index.tolist()[:10]}"
        )
