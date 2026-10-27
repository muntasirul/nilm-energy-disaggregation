# NILM — Energy Disaggregation

Turn a single household smart-meter signal into a **per-appliance** energy
breakdown — "your fridge used 8.2 kWh this week" — using deep learning
(Seq2Point CNN), the REDD dataset, FastAPI, and a Streamlit dashboard.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full plan, metrics, and roadmap.

## Quickstart

```bash
make setup        # venv + deps
source .venv/bin/activate
make data         # downloads REDD house_1 (~270 MB, GitHub mirror)
make preprocess   # .dat -> parquet
make train        # trains Seq2Point on refrigerator by default
make evaluate     # metrics + plot on the test split
make serve        # FastAPI -> http://localhost:8000/docs
make dashboard    # Streamlit -> http://localhost:8501
```

Or in Docker (after training, so `models/` exists):

```bash
docker compose up --build
```

## Data

- **REDD low_freq** — downloaded from `github.com/wuhanstudio/redd-lowfreq`
  (MIT's original host `redd.csail.mit.edu` is defunct).
- **UK-DALE / REFIT** — require free registration; see PROJECT_PLAN.md §6.

## Repo layout

```
config/config.yaml     all knobs (window size, appliance, split, lr, ...)
src/data/              download + REDD parser + torch Dataset
src/features/          sliding-window helpers
src/models/            Seq2Point CNN, train loop (MLflow), evaluation
src/serving/           FastAPI /disaggregate
src/dashboard/         Streamlit UI
tests/                 pytest suite
notebooks/             Kaggle training notebook (self-contained)
```

## Citation

Kolter, J. Z., & Johnson, M. J. (2011). *REDD: A public data set for energy
disaggregation research.* SustKDD workshop.
