"""Train Seq2Point on preprocessed REDD parquet files.

Usage:  python -m src.models.train
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import NilmWindowDataset, NormStats, load_house_df, split_by_time
from src.models.seq2point import build_model

CONFIG = Path("config/config.yaml")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_appliance(df: pd.DataFrame, candidates: list[str]) -> str:
    for a in candidates:
        if a in df.columns:
            return a
    raise ValueError(f"none of {candidates} found in columns {list(df.columns)}")


def build_datasets(cfg: dict):
    interim = Path(cfg["data"]["interim_dir"])
    houses = sorted(interim.glob("house_*.parquet"))
    if not houses:
        raise SystemExit("No interim parquet files. Run: make preprocess")

    strategy = cfg["data"]["split"]["strategy"]
    candidates = cfg["data"]["appliances"]

    if strategy == "by_house":
        sp = cfg["data"]["split"]
        splits = {"train": [], "val": [], "test": []}
        names = {"train": sp["train"], "val": sp["val"], "test": sp["test"]}
        frames = {}
        for key, house_list in names.items():
            for h in house_list:
                f = interim / f"{h}.parquet"
                if f.exists():
                    frames.setdefault(key, []).append(load_house_df(f))
        if not frames.get("train"):
            raise SystemExit("No train houses found on disk; download them first")
        train_df = pd.concat(frames["train"]).sort_index()
        val_df = pd.concat(frames["val"]).sort_index() if frames.get("val") else train_df
        test_df = pd.concat(frames["test"]).sort_index() if frames.get("test") else val_df
    else:
        df = load_house_df(houses[0])
        train_df, val_df, test_df = split_by_time(
            df,
            cfg["data"]["split"]["train_frac"],
            cfg["data"]["split"]["val_frac"],
        )

    appliance = pick_appliance(train_df, candidates)
    print(f"target appliance: {appliance}")

    W = cfg["model"]["window_size"]
    normalize = cfg["model"].get("normalize", "max")
    train_ds = NilmWindowDataset(train_df, appliance, W, normalize=normalize)
    val_ds = NilmWindowDataset(val_df, appliance, W, stats=train_ds.stats, normalize=normalize)
    test_ds = NilmWindowDataset(test_df, appliance, W, stats=train_ds.stats, normalize=normalize)
    return train_ds, val_ds, test_ds, appliance


@torch.no_grad()
def mae(model, loader, device) -> float:
    model.eval()
    errs = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        errs.append((model(x).squeeze(-1) - y).abs())
    return torch.cat(errs).mean().item()


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    tcfg = cfg["train"]
    torch.manual_seed(tcfg["seed"])
    random.seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    device = pick_device() if tcfg["device"] == "auto" else torch.device(tcfg["device"])
    print(f"device: {device}")

    train_ds, val_ds, _, appliance = build_datasets(cfg)
    train_dl = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=0, drop_last=True
    )
    val_dl = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0)
    print(f"train={len(train_ds)} windows  val={len(val_ds)} windows")

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    loss_fn = nn.MSELoss()

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "seq2point_best.pt"

    run = None
    if tcfg.get("mlflow_tracking", False):
        try:
            import mlflow

            mlflow.set_experiment("nilm")
            run = mlflow.start_run()
            mlflow.log_params(
                {"appliance": appliance, "window": cfg["model"]["window_size"], **tcfg}
            )
        except ImportError:
            print("mlflow not installed; continuing without tracking")

    best_val = float("inf")
    patience = 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        tot = 0.0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}/{tcfg['epochs']}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(-1), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)

        train_loss = tot / len(train_ds)
        val_mae = mae(model, val_dl, device)
        print(f"epoch {epoch}: train_mse={train_loss:.5f} val_mae={val_mae:.5f}")

        if run is not None:
            mlflow.log_metrics({"train_mse": train_loss, "val_mae": val_mae}, step=epoch)

        if val_mae < best_val:
            best_val, patience = val_mae, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "appliance": appliance,
                    "window_size": cfg["model"]["window_size"],
                    "normalize": cfg["model"].get("normalize", "max"),
                    "stats": vars(train_ds.stats),
                    "val_mae": val_mae,
                },
                ckpt_path,
            )
        else:
            patience += 1
            if patience >= tcfg["early_stop_patience"]:
                print(f"early stop at epoch {epoch} (best val_mae={best_val:.5f})")
                break

    meta = {"appliance": appliance, "stats": vars(train_ds.stats), "best_val_mae": best_val}
    (ckpt_dir / "seq2point_meta.json").write_text(json.dumps(meta, indent=2))
    if run is not None:
        mlflow.log_artifact(str(ckpt_path))
        mlflow.end_run()
    print(f"saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
