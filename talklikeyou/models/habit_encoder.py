from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CyclicPositionalEncoding(nn.Module):
    def __init__(self, feature_dim: int, period: int = 100, max_length: int = 600):
        super().__init__()
        encoding = torch.zeros(period, feature_dim)
        position = torch.arange(period, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, feature_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / feature_dim)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        repeats = max_length // period + 1
        self.register_buffer("pe", encoding.unsqueeze(0).repeat(1, repeats, 1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.pe[:, : values.shape[1]])


class AttentionPooling(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.W = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Mish(),
            nn.Linear(feature_dim, 1),
        )
        self.softmax = F.softmax

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = self.softmax(self.W(sequence).squeeze(-1), dim=-1).unsqueeze(-1)
        return torch.sum(sequence * weights, dim=1)


class ClassificationHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.dense = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = self.dropout(features)
        values = torch.tanh(self.dense(values))
        return self.out_proj(self.dropout(values))


class HabitEncoder(nn.Module):
    """Encode a 100-frame lip-motion reference into the unified habit space."""

    def __init__(
        self,
        motion_dim: int,
        feature_dim: int,
        habit_dim: int,
        person_num: int,
        reference_frames: int = 100,
    ):
        super().__init__()
        self.vertice_map = nn.Linear(motion_dim, feature_dim)
        self.PPE = CyclicPositionalEncoding(feature_dim, period=reference_frames)
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=4,
            dim_feedforward=2 * feature_dim,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.vertice_map_r = nn.Linear(feature_dim, habit_dim)
        self.cls = ClassificationHead(habit_dim, person_num)
        self.pooling = AttentionPooling(habit_dim)

    def forward(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        motion = motion.reshape(motion.shape[0], motion.shape[1], -1)
        features = self.vertice_map(motion)
        features = self.transformer_encoder(self.PPE(features))
        habit = self.pooling(self.vertice_map_r(features))
        return habit, self.cls(habit)
