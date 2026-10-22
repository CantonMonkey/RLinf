"""Observation-conditioned local passability helpers for G7.6."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


THRESHOLDS = tuple(round(value / 20.0, 2) for value in range(1, 20))


def candidate_geometry(heading_degrees: float, distance_m: float) -> np.ndarray:
    heading = math.radians(float(heading_degrees))
    distance = float(distance_m)
    if not math.isfinite(heading) or distance not in (0.5, 0.75, 1.0):
        raise ValueError("candidate geometry is outside the G7.4 macro contract")
    return np.asarray(
        [math.sin(heading), math.cos(heading), distance, distance * distance],
        dtype=np.float32,
    )


def validate_scene_split(train_scenes: Sequence[str], dev_scenes: Sequence[str]) -> None:
    train = tuple(train_scenes)
    dev = tuple(dev_scenes)
    if (
        len(train) != 6
        or len(dev) != 2
        or len(set(train)) != 6
        or len(set(dev)) != 2
        or set(train) & set(dev)
        or any(not isinstance(scene, str) or not scene for scene in train + dev)
    ):
        raise ValueError("G7.6 requires a disjoint 6-scene train / 2-scene dev split")


class GeometryPassabilityNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, rgb: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        del rgb
        return self.network(geometry).squeeze(-1)


class RGBPassabilityNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
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

    def forward(self, rgb: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        image = rgb.permute(0, 3, 1, 2).float().div(255.0)
        visual = self.visual(image)
        geometry_feature = self.geometry(geometry)
        return self.head(torch.cat((visual, geometry_feature), dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class PassabilityBatch:
    rgb: torch.Tensor
    geometry: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> "PassabilityBatch":
        return PassabilityBatch(
            rgb=self.rgb.to(device),
            geometry=self.geometry.to(device),
            labels=self.labels.to(device),
        )


def make_batch(rows: Sequence[dict], *, require_two_classes: bool = True) -> PassabilityBatch:
    if not rows:
        raise ValueError("passability batch cannot be empty")
    rgb = np.stack([np.asarray(row["rgb"], dtype=np.uint8) for row in rows])
    if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (64, 64, 3):
        raise ValueError("passability RGB must be uint8 [N,64,64,3]")
    geometry = np.stack(
        [
            candidate_geometry(row["heading_degrees"], row["distance_m"])
            for row in rows
        ]
    )
    labels = np.asarray([bool(row["unsafe"]) for row in rows], dtype=np.float32)
    if require_two_classes and labels.min() == labels.max():
        raise ValueError("passability batch must contain safe and unsafe candidates")
    return PassabilityBatch(
        rgb=torch.from_numpy(rgb.copy()),
        geometry=torch.from_numpy(geometry),
        labels=torch.from_numpy(labels),
    )


def train_model(
    model: nn.Module,
    train_rows: Sequence[dict],
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> dict:
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("passability training configuration is invalid")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    batch = make_batch(train_rows).to(device)
    model = model.to(device)
    positives = float(batch.labels.sum().item())
    negatives = float(batch.labels.numel() - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("passability training requires both classes")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    positive_weight = torch.tensor(negatives / positives, device=device)
    losses = []
    model.train()
    for _ in range(epochs):
        logits = model(batch.rgb, batch.geometry)
        loss = F.binary_cross_entropy_with_logits(
            logits, batch.labels, pos_weight=positive_weight
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("passability loss became nonfinite")
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
def predict_scores(
    model: nn.Module,
    rows: Sequence[dict],
    *,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    model.eval()
    scores = []
    for start in range(0, len(rows), batch_size):
        batch = make_batch(rows[start : start + batch_size], require_two_classes=False).to(device)
        scores.append(
            model(batch.rgb, batch.geometry).sigmoid().detach().cpu().numpy()
        )
    return np.concatenate(scores).astype(np.float64)


def binary_metrics(labels: Iterable[bool], scores: Iterable[float], threshold: float) -> dict:
    y = np.asarray(list(labels), dtype=np.bool_)
    s = np.asarray(list(scores), dtype=np.float64)
    if (
        y.ndim != 1
        or s.shape != y.shape
        or y.size < 2
        or y.min() == y.max()
        or not np.isfinite(s).all()
        or not 0.0 < threshold < 1.0
    ):
        raise ValueError("binary passability metrics require finite two-class inputs")
    predicted = s >= float(threshold)
    tp = int(np.logical_and(predicted, y).sum())
    fn = int(np.logical_and(~predicted, y).sum())
    tn = int(np.logical_and(~predicted, ~y).sum())
    fp = int(np.logical_and(predicted, ~y).sum())
    unsafe_recall = tp / (tp + fn)
    safe_recall = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    positive_scores = s[y]
    negative_scores = s[~y]
    comparisons = (
        (positive_scores[:, None] > negative_scores[None, :]).mean()
        + 0.5 * (positive_scores[:, None] == negative_scores[None, :]).mean()
    )
    return {
        "threshold": float(threshold),
        "count": int(y.size),
        "unsafe_count": int(y.sum()),
        "safe_count": int((~y).sum()),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "unsafe_recall": float(unsafe_recall),
        "safe_recall": float(safe_recall),
        "balanced_accuracy": float(0.5 * (unsafe_recall + safe_recall)),
        "unsafe_precision": float(precision),
        "auroc": float(comparisons),
    }


def select_threshold(
    labels: Sequence[bool],
    scores: Sequence[float],
    *,
    min_unsafe_recall: float,
    min_safe_recall: float,
) -> dict:
    if not 0.0 <= min_unsafe_recall <= 1.0 or not 0.0 <= min_safe_recall <= 1.0:
        raise ValueError("passability recall gates must be probabilities")
    rows = [binary_metrics(labels, scores, threshold) for threshold in THRESHOLDS]
    feasible = [
        row
        for row in rows
        if row["unsafe_recall"] >= min_unsafe_recall
        and row["safe_recall"] >= min_safe_recall
    ]
    if feasible:
        chosen = max(
            feasible,
            key=lambda row: (
                row["balanced_accuracy"],
                row["safe_recall"],
                row["threshold"],
            ),
        )
    else:
        chosen = max(
            rows,
            key=lambda row: (
                min(
                    row["unsafe_recall"] / max(min_unsafe_recall, 1e-12),
                    row["safe_recall"] / max(min_safe_recall, 1e-12),
                ),
                row["balanced_accuracy"],
                row["threshold"],
            ),
        )
    return {
        **chosen,
        "recall_gate_passed": bool(feasible),
        "grid": list(THRESHOLDS),
    }
