"""PyTorch Dataset and DataLoader helpers for Exp.15."""

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class ModalityDataset(Dataset):
    """
    Video-level dataset that returns (det, emo, qual, label) tensors.

    Args:
        df:              Feature matrix DataFrame (already scaled).
        det_col:         Name of the detector score column.
        emo_cols:        List of emotion feature column names.
        qual_cols:       List of quality feature column names.
        label_col:       Name of binary label column (0/1 int).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        det_col: str,
        emo_cols: List[str],
        qual_cols: List[str],
        label_col: str = "label_int",
    ):
        self.det = torch.tensor(df[det_col].values, dtype=torch.float32).unsqueeze(1)
        self.emo = torch.tensor(df[emo_cols].values, dtype=torch.float32)
        self.qual = torch.tensor(df[qual_cols].values, dtype=torch.float32)
        self.labels = torch.tensor(df[label_col].values, dtype=torch.float32)
        self.video_ids = df["video_id"].values if "video_id" in df.columns else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.det[idx], self.emo[idx], self.qual[idx], self.labels[idx]


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    det_col: str,
    emo_cols: List[str],
    qual_cols: List[str],
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    def _worker_init(worker_id):
        np.random.seed(seed + worker_id)

    train_ds = ModalityDataset(train_df, det_col, emo_cols, qual_cols)
    val_ds = ModalityDataset(val_df, det_col, emo_cols, qual_cols)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        worker_init_fn=_worker_init,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader
