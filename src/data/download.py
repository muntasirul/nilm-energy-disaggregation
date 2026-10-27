"""Download REDD low-frequency data.

MIT's original REDD host (redd.csail.mit.edu) is no longer reachable, so we
use the `wuhanstudio/redd-lowfreq` GitHub mirror, which contains the actual
REDD low_freq .dat files for all 6 houses.

Usage:
    python -m src.data.download --house house_1      # one house (v1 default)
    python -m src.data.download --all                # all 6 houses
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = "wuhanstudio/redd-lowfreq"
BRANCH = "main"
API = f"https://api.github.com/repos/{REPO}/contents"
HOUSES = [f"house_{i}" for i in range(1, 7)]


def list_house_files(house: str) -> list[dict]:
    """Return [{name, download_url, size}] for all files in a house dir."""
    url = f"{API}/{house}?ref={BRANCH}"
    with urllib.request.urlopen(url, timeout=30) as r:
        items = json.loads(r.read().decode())
    return [it for it in items if it.get("type") == "file"]


def download_file(url: str, dest: Path, retries: int = 3) -> None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nilm-dl"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            return
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1}/{retries} after error: {e}")
            time.sleep(2 * (attempt + 1))


def download_house(house: str, out_dir: Path) -> None:
    dest_dir = out_dir / house
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = list_house_files(house)
    total_mb = sum(f.get("size", 0) for f in files) / 1e6
    print(f"[{house}] {len(files)} files, ~{total_mb:.0f} MB -> {dest_dir}")

    for i, f in enumerate(files, 1):
        dest = dest_dir / f["name"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            print(f"  ({i}/{len(files)}) skip {f['name']} (already complete)")
            continue
        print(f"  ({i}/{len(files)}) {f['name']} ({f['size']/1e6:.1f} MB)", flush=True)
        download_file(f["download_url"], dest)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--house", default="house_1", help="e.g. house_1")
    p.add_argument("--all", action="store_true", help="download all 6 houses")
    p.add_argument("--out", default="data/raw/low_freq", help="output dir")
    args = p.parse_args()

    houses = HOUSES if args.all else [args.house]
    out_dir = Path(args.out)

    for house in houses:
        download_house(house, out_dir)

    print("Done. Next: python -m src.data.preprocess")


if __name__ == "__main__":
    sys.exit(main())
