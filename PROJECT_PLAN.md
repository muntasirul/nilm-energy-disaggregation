# NILM — Non-Intrusive Load Monitoring (Energy Disaggregation)

**Project Plan & Step-by-Step Implementation Guide**

> **One-liner:** Turn a single household smart-meter signal into a per-appliance energy breakdown, so users can see exactly what is consuming electricity and where to save.

---

## 1. The Real Problem

Most homes and small businesses see only **one total electricity number** on their bill or smart meter. They have no idea whether that spike came from the washing machine, the fridge, the AC, or a faulty device. Without per-appliance visibility:

- Consumers can't identify energy waste or high-consumption appliances.
- Utilities can't offer personalized efficiency advice or load-shifting incentives.
- Smart-home products can't act intelligently on "what is on right now."

**NILM (Non-Intrusive Load Monitoring)** solves this without installing a sensor on every appliance. It **disaggregates** the aggregate power signal into its constituent appliances using machine learning — a single meter in, per-appliance insights out.

---

## 2. Goals & Success Metrics

### Primary goals
1. Ingest raw smart-meter datasets and produce a clean, uniform training dataset.
2. Train a deep sequence model that predicts per-appliance power from aggregate power.
3. Serve the model behind an API and visualize results in a dashboard.

### Success metrics (targets)
| Metric | Target | Notes |
|---|---|---|
| Per-appliance MAE | < 30 W (varies by appliance) | Mean absolute error of predicted vs. true appliance power |
| Per-appliance F1 (on/off) | > 0.6 for high-draw appliances | Whether the appliance is on, using a threshold |
| SAE (Signal Aggregate Error) | < 0.3 | Relative error on total assigned energy |
| Inference latency | < 50 ms per window | Realistic for a smart-meter API |
| End-to-end pipeline | 100% reproducible | One command from raw data → trained model |

> Start with **one appliance** (recommended: **washing machine** or **kettle**) to prove the pipeline, then extend to 3–5 appliances.

---

## 3. Skills Demonstrated (CV Mapping)

| CV Claim | Where it shows up |
|---|---|
| **AI/ML Engineer** | Seq2Point/temporal-CNN model design, training loop, hyperparameter tuning, evaluation, model selection |
| **Data Analyst** | EDA notebooks, consumption patterns, anomaly detection, per-appliance distributions, visualization |
| **Software Engineer** | Clean repo structure, config-driven code, FastAPI service, React/Streamlit dashboard, tests, CI, Docker |
| **MLOps** | Data versioning, experiment tracking, model registry, containerization, monitoring |

---

## 4. High-Level Architecture

```
Raw meter data (UK-DALE / REFIT / REDD)
        │
        ▼
┌─────────────────┐
│  Data ingestion  │  ETL: download → parse → clean → resample → store (Parquet)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  EDA + analysis  │  Jupyter notebooks (load curves, distributions, correlations)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  Preprocessing   │  Normalize, sliding windows, train/val/test split (by HOUSE)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  Model training  │  Seq2Point CNN / temporal CNN (PyTorch) + experiment tracking
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  Evaluation      │  MAE, F1, SAE, NDE, per-appliance plots
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  Serving (FastAPI) │  POST /disaggregate → per-appliance power over time
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  Dashboard        │  Streamlit/React UI: "your fridge used X kWh this week"
└─────────────────┘
```

---

## 5. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Ecosystem |
| Deep learning | PyTorch | Flexibility for custom CNN/sequence models |
| Data processing | pandas, numpy, scipy | Time-series handling, resampling, filtering |
| Data storage | Parquet (local), optionally DVC | Efficient columnar storage + versioning |
| Experiment tracking | MLflow (or Weights & Biases) | Log metrics, params, artifacts |
| API | FastAPI + uvicorn | Fast, typed, auto-docs |
| Dashboard | Streamlit (fast) or React + Recharts (polish) | Visualize disaggregation |
| Testing | pytest | Unit + integration tests |
| Packaging | Docker + docker-compose | Reproducible deployment |
| CI | GitHub Actions | Lint, test, build |

---

## 6. Data Sources (verified)

| Dataset | Homes | Access | Status |
|---|---|---|---|
| **REDD low_freq** | 6 US homes | GitHub mirror `wuhanstudio/redd-lowfreq` — direct download, no auth | ✅ **verified working** (house_1 = ~273 MB, 21 `.dat` files) |
| **UK-DALE** | 5 UK homes | `data.ukedc.rl.ac.uk` — free CEDA registration required | 🔒 gated |
| **REFIT** | 20 UK homes | `pureportal.strath.ac.uk` — free account required | 🔒 gated |

> **Important:** MIT's original REDD host (`redd.csail.mit.edu`) is offline.
> The GitHub mirror is a byte-for-byte copy of `low_freq` and is what
> `src/data/download.py` uses. `refrigerator` and `washer_dryer` exist in all
> 6 houses — good default targets.

> **Tip:** Do NOT commit raw data (it's in `.gitignore`). Verify license terms
> before publishing derived data; publish code + models, link data sources.

---

## 7. Repository Structure

```
nilm-energy-disaggregation/
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── Makefile                      # one-command workflows
├── config/
│   └── config.yaml               # paths, model hyperparams, appliance list
├── data/
│   ├── raw/                      # downloaded source files (gitignored)
│   ├── interim/                  # cleaned per-house parquet
│   └── processed/                # windowed tensors / datasets
├── notebooks/
│   ├── 01_explore_ukdale.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_eval.ipynb
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   └── dataset.py
│   ├── features/
│   │   └── build_windows.py
│   ├── models/
│   │   ├── seq2point.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── serving/
│   │   └── app.py               # FastAPI
│   └── dashboard/
│       └── app.py               # Streamlit
├── tests/
│   ├── test_preprocess.py
│   ├── test_dataset.py
│   └── test_model.py
├── mlruns/                      # MLflow artifacts (gitignored)
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## 8. Step-by-Step Process

### Phase 0 — Environment Setup (Day 1)
1. Create repo, virtualenv, and install dependencies.
   ```bash
   git init nilm-energy-disaggregation && cd nilm-energy-disaggregation
   python -m venv .venv && source .venv/bin/activate
   pip install torch pandas numpy scipy matplotlib plotly fastapi uvicorn streamlit mlflow pytest
   pip freeze > requirements.txt
   ```
2. Add `.gitignore` (ignore `data/raw/`, `mlruns/`, `.venv/`, `__pycache__/`).
3. Initialize `config/config.yaml` with paths + hyperparameters.
4. Set up GitHub repo, push initial commit.

---

### Phase 1 — Data Ingestion & ETL
1. **Request/download** UK-DALE House 1 (and later House 2, 5).
2. **Parse** the source format (UK-DALE is CSV/npz) into pandas DataFrames:
   - `aggregate` (whole-house power, resampled to 6 s)
   - one column per `appliance` (e.g., washing machine, kettle, fridge, microwave, dishwasher)
3. **Clean:** drop NaNs, align timestamps, handle missing intervals, clip negative power to 0.
4. **Resample** to a uniform 6 s grid (forward-fill, then mean where needed).
5. **Store** as Parquet per house: `data/interim/house_1.parquet`.
6. **Validate:** assert no NaNs, monotonic timestamps, consistent sampling rate.

**Deliverable:** `src/data/download.py` + `src/data/preprocess.py` + a clean Parquet file.

---

### Phase 2 — Exploratory Data Analysis (Data Analyst proof)
Create `notebooks/01_explore_ukdale.ipynb`:
1. Plot **aggregate load over a day/week** to find patterns.
2. Plot **per-appliance load curves** and duty cycles.
3. Compute per-appliance **on/off statistics** (how often, how long, average power when on).
4. **Correlation** between aggregate and each appliance.
5. Detect **anomalies** (spikes, long gaps, unusual zeros).
6. Produce a summary: "fridge is always-on (~50 W base), washing machine is bursty (~2 kW peaks)."

**Deliverable:** a polished EDA notebook with clear visualizations (this is a portfolio artifact by itself).

---

### Phase 3 — Preprocessing & Windowing
1. **Normalize** aggregate and appliance signals (e.g., to [0,1] by max, or z-score).
2. **Build sliding windows** (Seq2Point style):
   - Input window length `W = 599` samples (≈ 1 hour at 6 s) → output = the **midpoint** sample's appliance power.
3. **Split by HOUSE** (never split within a home):
   - Train: House 1
   - Validation: House 2
   - Test: House 5 (unseen home → proves generalization)
4. **Shuffle** training windows, batch, and wrap in a PyTorch `Dataset`/`DataLoader`.

**Deliverable:** `src/features/build_windows.py` + `src/data/dataset.py` + unit tests.

---

### Phase 4 — Model Training
**Model: Seq2Point CNN** (from Kelly & Knottenbelt, "Neural NILM").

Suggested architecture:
```
Input: [batch, 1, W]
  Conv1D(1 → 30, kernel=10, stride=1) + ReLU
  Conv1D(30 → 30, kernel=8, stride=1) + ReLU
  Conv1D(30 → 40, kernel=6, stride=1) + ReLU
  Conv1D(40 → 50, kernel=5, stride=1) + ReLU
  Conv1D(50 → 50, kernel=5, stride=1) + ReLU
  Flatten → Dense(1024) → ReLU → Dropout(0.2)
  Dense(1)   # predicted appliance power at the midpoint
```

1. Implement the model in `src/models/seq2point.py`.
2. Write a **training loop** (`train.py`):
   - Loss: **MSE** (or MAE/Huber for robustness).
   - Optimizer: Adam, `lr = 1e-3`.
   - Early stopping on validation MAE, checkpoint best model.
3. **Track experiments** with MLflow: log `lr`, `W`, `batch_size`, per-epoch loss/MAE, and save model artifacts.
4. Train **one model per appliance** (transfer learning optional later).

**Deliverable:** training script + logged experiments + saved model checkpoints.

---

### Phase 5 — Evaluation
Use `src/models/evaluate.py` on the **unseen house (House 5)**:
1. Compute **MAE**, **RMSE** on appliance power.
2. Compute **F1** for on/off detection (threshold the output, e.g., >50 W = "on").
3. Compute **SAE** and **NDE** (normalized disaggregation error).
4. Plot **true vs. predicted** appliance power over a sample day.
5. Visualize the **aggregate vs. sum-of-appliances** to sanity-check energy accounting.

**Deliverable:** evaluation report + plots in `notebooks/03_model_eval.ipynb`.

---

### Phase 6 — Model Serving (FastAPI)
1. Export the trained model to **ONNX** or **TorchScript**.
2. Build `src/serving/app.py`:
   - `POST /disaggregate` — accepts a window of aggregate power (or a CSV upload), returns per-appliance power time series.
   - `GET /health` — readiness probe.
   - `GET /docs` — auto Swagger UI.
3. Load model once at startup; add input validation (Pydantic).
4. Test with a real 6 s aggregate signal.

**Deliverable:** a running API with auto-generated docs.

---

### Phase 7 — Dashboard
Build `src/dashboard/app.py` (Streamlit for speed, React for polish):
1. Upload or load a sample meter file.
2. Call the `/disaggregate` endpoint.
3. Show:
   - Stacked/overlaid **per-appliance power** chart.
   - **Energy share** donut chart (kWh per appliance per day/week).
   - Estimated **cost** per appliance (multiply kWh by a rate).
4. Highlight the top energy consumer ("Your fridge used 8.2 kWh this week").

**Deliverable:** a visually compelling demo (the strongest part of a portfolio).

---

### Phase 8 — MLOps, Tests & CI
1. Add **pytest** tests for preprocessing, dataset, and model forward pass.
2. **Dockerize:** `Dockerfile` for the API, `docker-compose.yml` to run API + dashboard together.
3. Add **GitHub Actions** CI: lint (ruff/black), type-check (mypy optional), run pytest, build Docker image.
4. Add **data versioning** (DVC) for processed Parquet + models (optional but impressive).
5. Write a **Makefile**: `make data`, `make train`, `make test`, `make serve`.

**Deliverable:** reproducible, testable, containerized project.

---

### Phase 9 — Documentation & Portfolio Polish
1. **README.md** with:
   - Problem statement + approach (with a simple architecture diagram).
   - Dataset citation + license note.
   - How to run (setup → train → serve → dashboard).
   - Results table (MAE/F1 per appliance).
   - Demo GIF/screenshots (before/after).
2. Add a **blog-style writeup** or LinkedIn post explaining the problem and your approach.
3. Ensure the repo is clean, commented where valuable, and has clear commit history.

---

## 9. Milestones & Suggested Timeline

| Week | Focus | Outcome |
|---|---|---|
| 1 | Setup + data ingestion + EDA | Clean Parquet + EDA notebook |
| 2 | Preprocessing + first model (one appliance) | Working Seq2Point baseline |
| 3 | Extend to 3–5 appliances + evaluate | Results table + plots |
| 4 | FastAPI + dashboard | End-to-end working demo |
| 5 | Tests, Docker, CI, README, polish | Portfolio-ready repo |

---

## 10. Deliverables Checklist

- [ ] Clean, versioned dataset pipeline (`make data`)
- [ ] EDA notebook with strong visuals
- [ ] Trained Seq2Point model(s) with logged experiments (MLflow)
- [ ] Evaluation report (MAE, F1, SAE, NDE) on an unseen house
- [ ] FastAPI inference endpoint
- [ ] Interactive dashboard
- [ ] pytest suite + GitHub Actions CI
- [ ] Docker + docker-compose
- [ ] Polished README with demo

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| UK-DALE access/license delays | Start with REFIT or REDD; use a public subset |
| Model overfits to one home | Split by house; test on an unseen home; add dropout + L2 |
| Low signal-to-noise for low-power appliances | Start with high-draw appliances (kettle, washing machine) |
| Large data / slow training | Use 6 s data (not 16 kHz); cap window count; use a GPU or Colab |
| Scope creep | Lock v1 to one appliance, then iterate |

---

## 12. Stretch Goals (after v1 works)

- **Multi-task model:** one model predicting all appliances simultaneously.
- **Transfer learning:** train on UK homes, fine-tune on US homes (cross-domain).
- **Real-time streaming:** consume a live meter feed instead of batch windows.
- **On-device edge inference:** quantize/export to run on a smart meter.
- **Unsupervised pre-training** (BERT-style masking on load curves).