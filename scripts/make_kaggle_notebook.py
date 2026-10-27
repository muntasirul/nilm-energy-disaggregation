"""Generate notebooks/kaggle_nilm_training.ipynb (self-contained Kaggle session).

Run:  python3 scripts/make_kaggle_notebook.py
"""

import json
from pathlib import Path

OUT = Path(__file__).parent.parent / "notebooks" / "kaggle_nilm_training.ipynb"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip().splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip().splitlines(keepends=True),
    }


cells = [
    md("""
# NILM — Energy Disaggregation (Seq2Point CNN)

Train a **Seq2Point CNN** to disaggregate a single appliance's power from the
aggregate smart-meter signal, using the **REDD low-frequency dataset**.

- **Data:** `github.com/wuhanstudio/redd-lowfreq` (mirror; MIT's original host is offline)
- **Target:** `refrigerator` (present in all 6 REDD houses; change `APPLIANCE` to try others)
- **Split:** chronological within `house_1` by default — set `SPLIT="by_house"` and add houses
  to `HOUSES` for the generalization experiment (train on 2 homes, test on an unseen one)
- **Metrics:** MAE (watts), F1 (on/off detection), SAE, NDE

**Runtime:** set *Accelerator → GPU T4* for much faster training.
"""),
    code("""
import json
import random
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

DATA_DIR = Path("/kaggle/working/redd")
"""),
    md("## 1. Download REDD low_freq data"),
    code("""
REPO, BRANCH = "wuhanstudio/redd-lowfreq", "main"
API = f"https://api.github.com/repos/{REPO}/contents"

# house_1 alone is ~270 MB. For the by-house experiment:
# HOUSES = ["house_1", "house_3", "house_6"]
HOUSES = ["house_1"]

def list_files(house):
    with urllib.request.urlopen(f"{API}/{house}?ref={BRANCH}", timeout=30) as r:
        return json.loads(r.read())

def download(url, dest):
    if dest.exists():
        return
    req = urllib.request.Request(url, headers={"User-Agent": "kaggle"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)

for house in HOUSES:
    d = DATA_DIR / house
    d.mkdir(parents=True, exist_ok=True)
    files = [f for f in list_files(house) if f["type"] == "file"]
    for f in tqdm(files, desc=house):
        download(f["download_url"], d / f["name"])
print("done")
"""),
    md("""## 2. Parse `.dat` files → DataFrame

Each `channel_N.dat` is `timestamp watts` per line. Channels 1–2 are the mains
legs (summed → `aggregate`); `labels.dat` maps other channels to appliances.
"""),
    code("""
SAMPLE_S = 3
MAINS = (1, 2)

def parse_labels(p):
    labels = {}
    for line in p.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            labels[int(parts[0])] = parts[1].strip().lower().replace(" ", "_")
    return labels

def load_channel(p):
    df = pd.read_csv(p, sep=r"\\s+", header=None, names=["ts", "power"])
    s = pd.Series(df.power.astype("float32").values,
                  index=pd.to_datetime(df.ts, unit="s"))
    return s[~s.index.duplicated()].sort_index()

def load_house(house_dir):
    labels = parse_labels(house_dir / "labels.dat")
    channels = {c: load_channel(house_dir / f"channel_{c}.dat") for c in labels
                if (house_dir / f"channel_{c}.dat").exists()}

    idx = channels[MAINS[0]].index
    for c in MAINS[1:]:
        idx = idx.union(channels[c].index)
    df = pd.DataFrame(index=idx.sort_values())

    df["aggregate"] = sum(channels[c].reindex(df.index).ffill().fillna(0) for c in MAINS)
    for c, name in labels.items():
        if c in MAINS or c not in channels:
            continue
        s = channels[c].reindex(df.index).ffill().fillna(0)
        df[name] = df[name] + s if name in df.columns else s

    return df.clip(lower=0).resample(f"{SAMPLE_S}s").mean().ffill().fillna(0)

frames = {h: load_house(DATA_DIR / h) for h in HOUSES}
house = frames[HOUSES[0]]
print(house.shape); house.head()
"""),
    md("## 3. Quick EDA"),
    code("""
day = house.iloc[:28_800]  # ~24 h at 3 s

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(day.index, day.aggregate, lw=0.5); axes[0].set_ylabel("watts")
axes[0].set_title("Aggregate")
axes[1].plot(day.index, day["refrigerator"], lw=0.6, color="tab:orange")
axes[1].set_title("refrigerator"); axes[1].set_ylabel("watts")
plt.tight_layout(); plt.show()

cols = [c for c in house.columns if c != "aggregate"]
duty = house[cols].gt(50).mean().sort_values(ascending=False).rename("duty_cycle")
duty.to_frame().style.format("{:.1%}")
"""),
    md("""## 4. Windowed dataset

Seq2Point: an aggregate window of `W` samples predicts the appliance power at
the window's **midpoint**.
"""),
    code("""
W = 599                    # ~30 min at 3 s sampling; must be odd
APPLIANCE = "refrigerator"
SPLIT = "by_time"          # "by_house" for the generalization experiment

class NilmDataset(Dataset):
    def __init__(self, df, appliance, W, stats=None):
        agg = df["aggregate"].to_numpy(np.float32)
        app = df[appliance].to_numpy(np.float32)
        self.s = stats or (np.abs(agg).max() + 1e-8, np.abs(app).max() + 1e-8)
        self.agg, self.app, self.half = agg / self.s[0], app / self.s[1], W // 2
        self.idx = np.arange(self.half, len(agg) - self.half)
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        c = self.idx[i]
        x = np.ascontiguousarray(self.agg[c - self.half : c + self.half + 1])
        return torch.from_numpy(x).unsqueeze(0), torch.tensor(self.app[c])

if SPLIT == "by_house":
    assert len(HOUSES) >= 2, "download >=2 houses for by_house split"
    train_df = pd.concat(frames[h] for h in HOUSES[:-1])
    test_df = frames[HOUSES[-1]]
    cut = int(len(train_df) * 0.85)
    train_df, val_df = train_df.iloc[:cut], train_df.iloc[cut:]
else:
    n = len(house)
    train_df = house.iloc[: int(n * 0.7)]
    val_df   = house.iloc[int(n * 0.7) : int(n * 0.85)]
    test_df  = house.iloc[int(n * 0.85) :]

train_ds = NilmDataset(train_df, APPLIANCE, W)
val_ds   = NilmDataset(val_df,   APPLIANCE, W, stats=train_ds.s)
test_ds  = NilmDataset(test_df,  APPLIANCE, W, stats=train_ds.s)

BATCH = 256
train_dl = DataLoader(train_ds, BATCH, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_dl   = DataLoader(val_ds, 1024, shuffle=False, num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds, 1024, shuffle=False, num_workers=2, pin_memory=True)
print(f"train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")
"""),
    md("## 5. Seq2Point model"),
    code("""
class Seq2Point(nn.Module):
    def __init__(self, W, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 30, 10, padding="same"), nn.ReLU(),
            nn.Conv1d(30, 30, 8, padding="same"), nn.ReLU(),
            nn.Conv1d(30, 40, 6, padding="same"), nn.ReLU(),
            nn.Conv1d(40, 50, 5, padding="same"), nn.ReLU(),
            nn.Conv1d(50, 50, 5, padding="same"), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(50 * W, 1024), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(1024, 1),
        )
    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)

model = Seq2Point(W).to(DEVICE)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
"""),
    md("## 6. Train"),
    code("""
EPOCHS, LR, PATIENCE = 15, 1e-3, 4

opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
loss_fn = nn.MSELoss()
best, wait, history = float("inf"), 0, []

@torch.no_grad()
def val_mae():
    model.eval()
    errs = [(model(x.to(DEVICE)) - y.to(DEVICE)).abs() for x, y in val_dl]
    return torch.cat(errs).mean().item()

for epoch in range(1, EPOCHS + 1):
    model.train()
    tot = 0.0
    for x, y in tqdm(train_dl, desc=f"epoch {epoch}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward(); opt.step()
        tot += loss.item() * len(x)

    vm = val_mae()
    history.append({"epoch": epoch, "train_mse": tot / len(train_ds), "val_mae": vm})
    print(f"epoch {epoch}: train_mse={history[-1]['train_mse']:.5f} val_mae={vm:.5f}")

    if vm < best:
        best, wait = vm, 0
        torch.save({"state_dict": model.state_dict(), "W": W, "appliance": APPLIANCE,
                    "agg_scale": train_ds.s[0], "app_scale": train_ds.s[1]},
                   "/kaggle/working/seq2point_best.pt")
    else:
        wait += 1
        if wait >= PATIENCE:
            print("early stop"); break

pd.DataFrame(history)
"""),
    md("## 7. Evaluate on held-out test data"),
    code("""
THR = 50  # watts -> appliance counts as "on"

ckpt = torch.load("/kaggle/working/seq2point_best.pt", map_location=DEVICE,
                  weights_only=False)
model.load_state_dict(ckpt["state_dict"]); model.eval()

preds, trues = [], []
with torch.no_grad():
    for x, y in test_dl:
        preds.append(model(x.to(DEVICE)).cpu()); trues.append(y)

y_pred = np.clip(torch.cat(preds).numpy() * test_ds.s[1], 0, None)
y_true = torch.cat(trues).numpy() * test_ds.s[1]

mae = np.abs(y_pred - y_true).mean()
sae = abs(y_pred.sum() - y_true.sum()) / y_true.sum()
nde = ((y_pred - y_true) ** 2).sum() / (y_true ** 2).sum()
tp = ((y_pred > THR) & (y_true > THR)).sum()
fp = ((y_pred > THR) & (y_true <= THR)).sum()
fn = ((y_pred <= THR) & (y_true > THR)).sum()
f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)

print(f"{APPLIANCE}: MAE={mae:.1f} W  SAE={sae:.3f}  NDE={nde:.3f}  F1={f1:.3f}")

n = 3000
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(y_true[:n], label="true", lw=0.8)
ax.plot(y_pred[:n], label="pred", lw=0.8, alpha=0.8)
ax.set_title(f"{APPLIANCE} — predicted vs true")
ax.set_ylabel("watts"); ax.legend()
plt.tight_layout(); plt.savefig("/kaggle/working/pred_vs_true.png"); plt.show()
"""),
    md("""
## Next steps

- **More appliances:** train one model per appliance (`washer_dryer`, `dishwasher`, `microwave`)
- **By-house generalization:** set `HOUSES = ["house_1","house_3","house_6"]` + `SPLIT="by_house"`
- **Back in the repo:** the trained checkpoint drops into `models/` and works with
  `src/serving/app.py` (FastAPI) and `src/dashboard/app.py` (Streamlit)
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {"accelerator": "gpu", "dataSources": [], "dockerImageVersionId": None,
                   "isInternetEnabled": True, "language": "python", "sourceType": "notebook"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(cells)} cells)")
