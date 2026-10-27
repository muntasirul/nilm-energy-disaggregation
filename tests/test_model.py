import numpy as np
import pandas as pd
import torch

from src.data.dataset import NilmWindowDataset, split_by_time
from src.features.build_windows import center_indices, windowed_center_view
from src.models.seq2point import Seq2Point


def make_df(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="3s")
    agg = rng.random(n).astype("float32") * 2000
    app = np.clip(agg * 0.1 + rng.normal(0, 30, n), 0, None).astype("float32")
    return pd.DataFrame({"aggregate": agg, "refrigerator": app}, index=idx)


def test_model_forward_shape():
    W = 99
    m = Seq2Point(window_size=W)
    out = m(torch.zeros(4, 1, W))
    assert out.shape == (4, 1)


def test_model_backward():
    W = 99
    m = Seq2Point(window_size=W)
    x = torch.rand(8, 1, W, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None


def test_dataset_shapes_and_midpoint():
    df = make_df()
    W = 99
    ds = NilmWindowDataset(df, "refrigerator", W)
    assert len(ds) == len(df) - W + 1
    x, y = ds[0]
    assert x.shape == (1, W)
    assert y.dim() == 0


def test_center_indices():
    idx = center_indices(10, 5)
    assert idx[0] == 2 and idx[-1] == 7 and len(idx) == 6


def test_windowed_center_view():
    x = np.arange(10, dtype=np.float32)
    w, mid = windowed_center_view(x, 5)
    assert w.shape == (6, 5)
    np.testing.assert_array_equal(mid, x[2:8])
    np.testing.assert_array_equal(w[0], x[0:5])


def test_split_by_time():
    df = make_df()
    tr, va, te = split_by_time(df, 0.7, 0.15)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr.index.max() < va.index.min() < te.index.min() or len(va) == 0
