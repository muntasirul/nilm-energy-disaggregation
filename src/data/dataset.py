"""PyTorch datasets + house/time-based splitting for NILM."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.build_windows import center_indices


@dataclass
class NormStats:
    agg_scale: float
    agg_mean: float
    agg_std: float
    app_scale: float


def compute_norm_stats(agg: np.ndarray, app: np.ndarray, mode: str = "max") -> NormStats:
    if mode == "zscore":
        return NormStats(1.0, float(agg.mean()), float(agg.std() + 1e-8), float(app.std() + 1e-8))
    return NormStats(float(np.abs(agg).max() + 1e-8), 0.0, 1.0, float(np.abs(app).max() + 1e-8))


class NilmWindowDataset(Dataset):
    """Seq2Point dataset: input = aggregate window [1, W], target = appliance
    power at the window midpoint (scalar, watts, in normalized space)."""

    def __init__(
        self,
        df: pd.DataFrame,
        appliance: str,
        window_size: int,
        stats: NormStats | None = None,
        normalize: str = "max",
    ):
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd (Seq2Point needs a midpoint)")
        if appliance not in df.columns:
            raise ValueError(f"appliance '{appliance}' not in columns: {list(df.columns)}")

        self.window_size = window_size
        agg = df["aggregate"].to_numpy(dtype=np.float32)
        app = df[appliance].to_numpy(dtype=np.float32)

        self.stats = stats or compute_norm_stats(agg, app, normalize)
        self.agg = (
            (agg - self.stats.agg_mean) / self.stats.agg_std
            if normalize == "zscore"
            else agg / self.stats.agg_scale
        )
        self.app = app / self.stats.app_scale

        self.idx = center_indices(len(self.agg), window_size)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        c = int(self.idx[i])
        half = self.window_size // 2
        x = self.agg[c - half : c + half + 1]
        y = self.app[c]
        return (
            torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0),  # [1, W]
            torch.tensor(y, dtype=torch.float32),
        )


def split_by_time(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    """Chronological split of a single-house DataFrame."""
    n = len(df)
    t1, t2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]


def load_house_df(path) -> pd.DataFrame:
    return pd.read_parquet(path)
