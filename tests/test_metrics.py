from __future__ import annotations

from common.metrics.latency import LatencyTracker
from common.metrics.overlap_consistency import compute_overlap_penalty
from common.metrics.tone_f1 import tone_f1_score


def test_latency_tracker_summary() -> None:
    tracker = LatencyTracker()
    for value in [100, 200, 300]:
        tracker.add(value)
    summary = tracker.summary()
    assert summary["p50"] == 200


def test_overlap_penalty_zero_when_match() -> None:
    ref = [0.0, 0.0, 1.0]
    pred = [0.0, 0.0, 1.0]
    assert compute_overlap_penalty(ref, pred) == 0.0


def test_tone_f1_perfect_match() -> None:
    tones = ["<tone1>", "<tone2>"]
    assert tone_f1_score(tones, tones) == 1.0
