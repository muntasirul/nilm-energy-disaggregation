"""Streamlit dashboard: visualize NILM disaggregation.

Run:  streamlit run src/dashboard/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml

from src.features.build_windows import center_indices
from src.models.seq2point import Seq2Point

st.set_page_config(page_title="NILM Energy Disaggregation", layout="wide")
st.title("NILM — Energy Disaggregation")
st.caption("Aggregate smart-meter power in → per-appliance power out.")


@st.cache_resource
def load_model():
    cfg = yaml.safe_load(Path("config/config.yaml").read_text())
    ckpt = torch.load(cfg["evaluate"]["checkpoint"], map_location="cpu", weights_only=False)
    model = Seq2Point(window_size=ckpt["window_size"]).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


try:
    model, meta = load_model()
except Exception as e:
    st.error(f"No trained checkpoint found ({e}). Train first with `make train`.")
    st.stop()

W = meta["window_size"]
agg_scale = meta["stats"]["agg_scale"]
app_scale = meta["stats"]["app_scale"]

src = st.sidebar.radio("Input source", ["Sample parquet", "Upload CSV"])
df = None

if src == "Sample parquet":
    interim = Path("data/interim")
    files = sorted(interim.glob("house_*.parquet"))
    if not files:
        st.warning("No data/interim/*.parquet. Run `make data && make preprocess`.")
        st.stop()
    f = st.sidebar.selectbox("House", files, format_func=lambda p: p.stem)
    df = pd.read_parquet(f)
    n = len(df)
    lo, hi = st.sidebar.slider("Time slice", 0, n - W, (0, min(20_000, n - W)))
    df = df.iloc[lo : lo + hi]
else:
    up = st.sidebar.file_uploader("CSV with an 'aggregate' column", type=["csv"])
    if up is None:
        st.stop()
    df = pd.read_csv(up)

if "aggregate" not in df.columns:
    st.error("Need an 'aggregate' column.")
    st.stop()

agg = df["aggregate"].to_numpy(dtype=np.float32)
if len(agg) < W:
    st.error(f"Need >= {W} samples; got {len(agg)}.")
    st.stop()

norm = agg / agg_scale
idx = center_indices(len(norm), W)
windows = np.stack([norm[c - W // 2 : c + W // 2 + 1] for c in idx])
batch = torch.from_numpy(windows).unsqueeze(1)

preds = []
with torch.no_grad():
    for i in range(0, len(batch), 4096):
        preds.append(model(batch[i : i + 4096]).squeeze(-1))
y = np.clip(torch.cat(preds).numpy() * app_scale, 0, None)
mid_ts = df.index[idx] if isinstance(df.index, pd.DatetimeIndex) else idx

appliance = meta["appliance"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=mid_ts, y=y, name=f"{appliance} (predicted)", line=dict(width=1)))
if appliance in df.columns:
    true = df[appliance].to_numpy(dtype=np.float32)[idx]
    fig.add_trace(go.Scatter(x=mid_ts, y=true, name=f"{appliance} (true)", line=dict(width=1)))
fig.add_trace(
    go.Scatter(
        x=df.index, y=agg, name="aggregate", line=dict(width=0.5, color="gray"), opacity=0.5
    )
)
fig.update_layout(height=480, yaxis_title="watts", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

sample_s = 3
kwh = y.sum() * sample_s / 3600 / 1000
c1, c2, c3 = st.columns(3)
c1.metric(f"Predicted {appliance} energy", f"{kwh:.2f} kWh")
c2.metric("Peak predicted power", f"{y.max():.0f} W")
c3.metric("Windows disaggregated", f"{len(y):,}")
