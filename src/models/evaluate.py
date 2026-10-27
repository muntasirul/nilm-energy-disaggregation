"""Evaluate a trained Seq2Point checkpoint on the test split.

Usage:  python -m src.models.evaluate [--checkpoint models/seq2point_best.pt]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.dataset import NilmWindowDataset, NormStats
from src.models.seq2point import Seq2Point
from src.models.train import build_datasets, pick_device

CONFIG = Path("config/config.yaml")


def metrics(y_true: np.ndarray, y_pred: np.ndarray, on_threshold: float) -> dict:
    mae = np.abs(y_pred - y_true).mean()
    sae = abs(y_pred.sum() - y_true.sum()) / (y_true.sum() + 1e-8)
    nde = ((y_pred - y_true) ** 2).sum() / ((y_true**2).sum() + 1e-8)
    f1 = f1_score(y_true > on_threshold, y_pred > on_threshold)
    return {"mae_watts": float(mae), "sae": float(sae), "nde": float(nde), "f1_on_off": float(f1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--max-windows", type=int, default=200_000)
    p.add_argument("--plot", default="reports/test_day.png")
    args = p.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    ckpt_path = args.checkpoint or cfg["evaluate"]["checkpoint"]
    thr = cfg["evaluate"]["on_threshold_watts"]

    _, _, test_ds, appliance = build_datasets(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = Seq2Point(window_size=ckpt["window_size"]).eval()
    model.load_state_dict(ckpt["state_dict"])
    device = pick_device()
    model.to(device)

    dl = DataLoader(test_ds, batch_size=2048, shuffle=False)
    preds, trues = [], []
    n = 0
    with torch.no_grad():
        for x, y in dl:
            preds.append(model(x.to(device)).cpu().squeeze(-1))
            trues.append(y)
            n += len(x)
            if n >= args.max_windows:
                break

    y_pred = torch.cat(preds).numpy() * test_ds.stats.app_scale
    y_true = torch.cat(trues).numpy()[: len(y_pred)] * test_ds.stats.app_scale
    y_pred = np.clip(y_pred, 0, None)

    m = metrics(y_true, y_pred, thr)
    print(f"appliance: {appliance}")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")

    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame([m], index=[appliance]).to_csv("reports/metrics.csv")

    step = max(1, len(y_true) // 20_000)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_true[::step], label="true", lw=0.8)
    ax.plot(y_pred[::step], label="pred", lw=0.8, alpha=0.8)
    ax.set_title(f"{appliance} — true vs predicted power")
    ax.set_ylabel("watts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120)
    print(f"plot -> {args.plot}")


if __name__ == "__main__":
    main()
