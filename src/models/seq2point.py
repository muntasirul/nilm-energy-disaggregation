"""Seq2Point CNN for NILM (Kelly & Knottenbelt, "Neural NILM", 2015).

Input : [batch, 1, W]  aggregate power window
Output: [batch, 1]     appliance power at the window midpoint
"""

import torch
import torch.nn as nn


class Seq2Point(nn.Module):
    def __init__(self, window_size: int = 599, dropout: float = 0.2):
        super().__init__()
        self.window_size = window_size
        self.features = nn.Sequential(
            nn.Conv1d(1, 30, kernel_size=10, padding="same"),
            nn.ReLU(),
            nn.Conv1d(30, 30, kernel_size=8, padding="same"),
            nn.ReLU(),
            nn.Conv1d(30, 40, kernel_size=6, padding="same"),
            nn.ReLU(),
            nn.Conv1d(40, 50, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv1d(50, 50, kernel_size=5, padding="same"),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * window_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(cfg: dict) -> nn.Module:
    return Seq2Point(
        window_size=cfg["model"]["window_size"],
        dropout=cfg["model"].get("dropout", 0.2),
    )
