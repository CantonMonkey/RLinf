"""Aligned metric-depth local-controller helpers for G7.8."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from streamvln_passability_g76 import (
    THRESHOLDS,
    GeometryPassabilityNet,
    binary_metrics,
    candidate_geometry,
    select_threshold,
    validate_scene_split,
)


DEPTH_MAX_METERS = 10.0


class DepthPassabilityNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.GELU(),
        )
        self.geometry = nn.Sequential(nn.Linear(4, 32), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(160, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, depth: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        metric = depth.unsqueeze(1).float().clamp_(0.0, DEPTH_MAX_METERS)
        visual = self.depth_encoder(metric.div_(DEPTH_MAX_METERS))
        geometry_feature = self.geometry(geometry)
        return self.head(torch.cat((visual, geometry_feature), dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class DepthBatch:
    depth: torch.Tensor
    geometry: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> "DepthBatch":
        return DepthBatch(
            depth=self.depth.to(device),
            geometry=self.geometry.to(device),
            labels=self.labels.to(device),
        )


def make_depth_batch(
    rows: Sequence[dict], *, require_two_classes: bool = True
) -> DepthBatch:
    if not rows:
        raise ValueError("depth passability batch cannot be empty")
    depth = np.stack(
        [np.asarray(row["metric_depth"], dtype=np.float32) for row in rows]
    )
    if depth.ndim != 3 or tuple(depth.shape[1:]) != (64, 64):
        raise ValueError("metric depth must be float32 [N,64,64]")
    if (
        not np.isfinite(depth).all()
        or float(depth.min()) < 0.0
        or float(depth.max()) > DEPTH_MAX_METERS + 1e-4
    ):
        raise ValueError("metric depth is outside configured meters")
    geometry = np.stack(
        [
            candidate_geometry(row["heading_degrees"], row["distance_m"])
            for row in rows
        ]
    )
    labels = np.asarray([bool(row["unsafe"]) for row in rows], dtype=np.float32)
    if require_two_classes and labels.min() == labels.max():
        raise ValueError("depth passability batch must contain both classes")
    return DepthBatch(
        depth=torch.from_numpy(depth.copy()),
        geometry=torch.from_numpy(geometry),
        labels=torch.from_numpy(labels),
    )


def train_depth_model(
    model: nn.Module,
    train_rows: Sequence[dict],
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> dict:
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("depth training configuration is invalid")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    batch = make_depth_batch(train_rows).to(device)
    model = model.to(device)
    positives = float(batch.labels.sum().item())
    negatives = float(batch.labels.numel() - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("depth training requires both classes")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    positive_weight = torch.tensor(negatives / positives, device=device)
    losses = []
    model.train()
    for _ in range(epochs):
        logits = model(batch.depth, batch.geometry)
        loss = F.binary_cross_entropy_with_logits(
            logits, batch.labels, pos_weight=positive_weight
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("depth passability loss became nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "positive_weight": float(positive_weight.detach().cpu()),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "finite": all(math.isfinite(value) for value in losses),
    }


@torch.inference_mode()
def predict_depth_scores(
    model: nn.Module,
    rows: Sequence[dict],
    *,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("depth batch size must be positive")
    model.eval()
    scores = []
    for start in range(0, len(rows), batch_size):
        batch = make_depth_batch(
            rows[start : start + batch_size], require_two_classes=False
        ).to(device)
        scores.append(
            model(batch.depth, batch.geometry).sigmoid().detach().cpu().numpy()
        )
    return np.concatenate(scores).astype(np.float64)


def per_scene_metrics(rows, scores, threshold):
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(rows),):
        raise ValueError("per-scene score count mismatch")
    result = {}
    scenes = sorted({row["scene"] for row in rows})
    for scene in scenes:
        indices = [index for index, row in enumerate(rows) if row["scene"] == scene]
        labels = [rows[index]["unsafe"] for index in indices]
        if len(set(labels)) != 2:
            raise ValueError("each G7.8 dev scene must contain both classes")
        result[scene] = binary_metrics(
            labels, values[np.asarray(indices, dtype=np.int64)], threshold
        )
    return result


def threshold_sensitivity(rows, scores):
    labels = [row["unsafe"] for row in rows]
    values = np.asarray(scores, dtype=np.float64)
    return {
        "overall": [
            binary_metrics(labels, values, threshold) for threshold in THRESHOLDS
        ],
        "by_scene": {
            scene: [
                binary_metrics(
                    [row["unsafe"] for row in rows if row["scene"] == scene],
                    values[
                        np.asarray(
                            [
                                index
                                for index, row in enumerate(rows)
                                if row["scene"] == scene
                            ],
                            dtype=np.int64,
                        )
                    ],
                    threshold,
                )
                for threshold in THRESHOLDS
            ]
            for scene in sorted({row["scene"] for row in rows})
        },
    }


__all__ = [
    "DepthPassabilityNet",
    "GeometryPassabilityNet",
    "per_scene_metrics",
    "predict_depth_scores",
    "select_threshold",
    "threshold_sensitivity",
    "train_depth_model",
    "validate_scene_split",
]
