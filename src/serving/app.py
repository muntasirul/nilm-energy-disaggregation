"""FastAPI inference service for NILM disaggregation.

POST /disaggregate
    body: {"aggregate": [watt, watt, ...]}  (length >= window_size)
    resp: {"appliance": str, "predicted_power": [watt, ...]}

The model is loaded once at startup from models/seq2point_best.pt.
"""

from pathlib import Path

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.build_windows import center_indices
from src.models.seq2point import Seq2Point

app = FastAPI(title="NILM Disaggregation API", version="0.1.0")

_model = None
_meta = None


def load_checkpoint():
    global _model, _meta
    if _model is not None:
        return
    cfg = yaml.safe_load(Path("config/config.yaml").read_text())
    ckpt_path = Path(cfg["evaluate"]["checkpoint"])
    if not ckpt_path.exists():
        raise RuntimeError(f"checkpoint not found: {ckpt_path}. Train first: make train")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Seq2Point(window_size=ckpt["window_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _model, _meta = model, ckpt


class DisaggregateRequest(BaseModel):
    aggregate: list[float] = Field(..., description="aggregate power samples (watts)")


class DisaggregateResponse(BaseModel):
    appliance: str
    window_size: int
    n_outputs: int
    predicted_power: list[float]


@app.on_event("startup")
def _startup():
    load_checkpoint()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/disaggregate", response_model=DisaggregateResponse)
def disaggregate(req: DisaggregateRequest):
    x = np.asarray(req.aggregate, dtype=np.float32)
    W = _meta["window_size"]
    if len(x) < W:
        raise HTTPException(422, f"need at least {W} samples, got {len(x)}")

    agg = x / _meta["stats"]["agg_scale"]
    idx = center_indices(len(agg), W)
    windows = np.stack([agg[c - W // 2 : c + W // 2 + 1] for c in idx])
    batch = torch.from_numpy(windows).unsqueeze(1)  # [N, 1, W]

    preds = []
    with torch.no_grad():
        for i in range(0, len(batch), 4096):
            preds.append(_model(batch[i : i + 4096]).squeeze(-1))
    y = torch.cat(preds).numpy() * _meta["stats"]["app_scale"]
    y = np.clip(y, 0, None)

    return DisaggregateResponse(
        appliance=_meta["appliance"],
        window_size=W,
        n_outputs=len(y),
        predicted_power=[round(float(v), 2) for v in y],
    )
