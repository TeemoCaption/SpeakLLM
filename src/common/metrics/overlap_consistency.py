"""Overlap consistency metric for duplex evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class OverlapMetricConfig:
    conflict_margin: float = 0.2
    smoothing: int = 5


def compute_overlap_penalty(
    reference: Iterable[float],
    prediction: Iterable[float],
    config: OverlapMetricConfig | None = None,
) -> float:
    cfg = config or OverlapMetricConfig()
    ref = np.array(list(reference))
    pred = np.array(list(prediction))
    if ref.size != pred.size:
        raise ValueError("Reference and prediction must have equal length")
    diff = np.clip(pred - ref - cfg.conflict_margin, 0.0, 1.0)
    if cfg.smoothing > 1:
        kernel = np.ones(cfg.smoothing) / cfg.smoothing
        diff = np.convolve(diff, kernel, mode="same")
    return float(diff.mean())
