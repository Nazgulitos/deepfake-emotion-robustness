"""Dataset classes for ThreeModalityGated experiment."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ThreeModalityDataset(Dataset):
    def __init__(self, df, qual_cols, emo_static_cols, emo_temporal_cols,
                 label_col="label_int"):
        self.X_q = df[qual_cols].values.astype(np.float32)
        self.X_s = df[emo_static_cols].values.astype(np.float32)
        self.X_t = df[emo_temporal_cols].values.astype(np.float32)
        self.y = df[label_col].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_q[idx]),
            torch.from_numpy(self.X_s[idx]),
            torch.from_numpy(self.X_t[idx]),
            torch.tensor(self.y[idx]),
        )


class TwoModalityDataset(Dataset):
    """Two-branch dataset for ablation variants."""

    def __init__(self, df, cols_a, cols_b, label_col="label_int"):
        self.X_a = df[cols_a].values.astype(np.float32)
        self.X_b = df[cols_b].values.astype(np.float32)
        self.y = df[label_col].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_a[idx]),
            torch.from_numpy(self.X_b[idx]),
            torch.tensor(self.y[idx]),
        )


def make_loaders(train_df, val_df, qual_cols, emo_static_cols, emo_temporal_cols,
                 batch_size: int = 32):
    train_ds = ThreeModalityDataset(train_df, qual_cols, emo_static_cols, emo_temporal_cols)
    val_ds = ThreeModalityDataset(val_df, qual_cols, emo_static_cols, emo_temporal_cols)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def make_ablation_loaders(train_df, val_df, cols_a, cols_b, batch_size: int = 32):
    train_ds = TwoModalityDataset(train_df, cols_a, cols_b)
    val_ds = TwoModalityDataset(val_df, cols_a, cols_b)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
