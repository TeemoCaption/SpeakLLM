"""Latency metrics for streaming evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class LatencyWindow:
    max_samples: int = 1000


class LatencyTracker:
    def __init__(self, window: LatencyWindow | None = None) -> None:
        self.window = window or LatencyWindow()
        self.samples: List[float] = []

    def add(self, value_ms: float) -> None:
        self.samples.append(value_ms)
        if len(self.samples) > self.window.max_samples:
            self.samples = self.samples[-self.window.max_samples :]

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "mean": 0.0}
        arr = np.array(self.samples)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(np.mean(arr)),
        }
