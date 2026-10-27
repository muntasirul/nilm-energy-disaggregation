"""Sliding-window helpers for Seq2Point-style NILM.

Seq2Point maps an input window of aggregate power -> the appliance power at
the window's midpoint. So for window_size W (odd), target index t covers
input [t - W//2, t + W//2].
"""

import numpy as np


def center_indices(n: int, window_size: int) -> np.ndarray:
    """Valid midpoint indices for a series of length n with window W."""
    half = window_size // 2
    return np.arange(half, n - half)


def windowed_center_view(x: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (windows, midpoints):
      windows   -> [num_windows, W] rows of x
      midpoints -> [num_windows] x values at each window's center
    Uses strided views so no data is copied (read-only semantics).
    """
    n = len(x)
    if n < window_size:
        raise ValueError(f"series length {n} < window_size {window_size}")
    idx = center_indices(n, window_size)
    starts = idx - window_size // 2
    windows = np.lib.stride_tricks.sliding_window_view(x, window_size)[starts]
    return windows, x[idx]
