import numpy as np
import pandas as pd

from src.data.preprocess import parse_labels


def test_parse_labels(tmp_path):
    p = tmp_path / "labels.dat"
    p.write_text("1 mains\n2 mains\n3 oven\n4 oven\n5 refrigerator\n")
    labels = parse_labels(p)
    assert labels == {1: "mains", 2: "mains", 3: "oven", 4: "oven", 5: "refrigerator"}


def test_load_channel(tmp_path):
    from src.data.preprocess import load_channel

    p = tmp_path / "channel_1.dat"
    p.write_text("1300000000 120\n1300000001 121\n1300000002 119\n")
    s = load_channel(p)
    assert len(s) == 3
    assert list(s.values) == [120, 121, 119]
    assert s.index.is_monotonic_increasing
