"""Parse REDD low_freq .dat files into clean per-house Parquet files.

REDD low_freq layout (per house dir):
    labels.dat           -> "N appliance_name" per line
    channel_1.dat        -> "timestamp power" per line (mains leg 1)
    channel_2.dat        -> "timestamp power" per line (mains leg 2)
    channel_3+.dat       -> appliance channels (~3s sampling)

Output: one Parquet per house in data/interim/, indexed by timestamp,
columns = aggregate + one column per appliance (same-named channels summed).

Usage:
    python -m src.data.preprocess
"""

import re
from pathlib import Path

import pandas as pd
import yaml

CONFIG = Path("config/config.yaml")


def parse_labels(labels_path: Path) -> dict[int, str]:
    """labels.dat: '<channel> <appliance>' per line -> {channel: appliance}."""
    labels = {}
    for line in labels_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\d+)\s+(.+)", line)
        if m:
            labels[int(m.group(1))] = m.group(2).strip().lower().replace(" ", "_")
    return labels


def load_channel(path: Path) -> pd.Series:
    """A channel_N.dat file: 'unix_timestamp watts' whitespace-separated."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["ts", "power"],
        dtype={"ts": "int64", "power": "float32"},
    )
    s = pd.Series(df["power"].values, index=pd.to_datetime(df["ts"], unit="s"))
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_house(house_dir: Path, mains_channels=(1, 2), sample_seconds: int = 3) -> pd.DataFrame:
    """Parse one REDD house dir -> wide DataFrame resampled to `sample_seconds`."""
    labels = parse_labels(house_dir / "labels.dat")

    channels: dict[int, pd.Series] = {}
    for ch, name in labels.items():
        f = house_dir / f"channel_{ch}.dat"
        if f.exists():
            channels[ch] = load_channel(f)

    # union of all timestamps -> common grid
    all_ts = channels[mains_channels[0]].index
    for ch in mains_channels[1:]:
        all_ts = all_ts.union(channels[ch].index)

    df = pd.DataFrame(index=all_ts.sort_values())

    # aggregate = sum of mains legs
    agg = pd.Series(0.0, index=df.index)
    for ch in mains_channels:
        agg = agg.add(channels[ch].reindex(df.index).ffill().fillna(0), fill_value=0)
    df["aggregate"] = agg

    # appliances: sum channels sharing a label (e.g. two ovens)
    for ch, name in labels.items():
        if ch in mains_channels or ch not in channels:
            continue
        s = channels[ch].reindex(df.index).ffill().fillna(0)
        df[name] = df[name] + s if name in df.columns else s

    df = df.clip(lower=0)
    df = df.resample(f"{sample_seconds}s").mean().ffill().fillna(0)
    return df


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    raw = Path(cfg["data"]["raw_dir"])
    interim = Path(cfg["data"]["interim_dir"])
    interim.mkdir(parents=True, exist_ok=True)
    sample_seconds = cfg["data"]["sample_seconds"]
    mains = tuple(cfg["data"].get("mains_channels", [1, 2]))

    houses = sorted(d for d in raw.iterdir() if d.is_dir() and (d / "labels.dat").exists())
    if not houses:
        raise SystemExit(f"No REDD houses found under {raw}. Run: make data")

    for house_dir in houses:
        print(f"Parsing {house_dir.name} ...")
        df = load_house(house_dir, mains, sample_seconds)
        out = interim / f"{house_dir.name}.parquet"
        df.to_parquet(out)
        print(f"  -> {out}  shape={df.shape}  cols={list(df.columns)[:8]}...")


if __name__ == "__main__":
    main()
