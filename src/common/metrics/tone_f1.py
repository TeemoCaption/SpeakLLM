"""Tone classification F1 for tonal alignment checks."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Tuple


def tone_f1_score(predictions: Iterable[str], references: Iterable[str]) -> float:
    pred_counter = Counter()
    ref_counter = Counter()
    match_counter = Counter()
    for pred, ref in zip(predictions, references):
        pred_counter[pred] += 1
        ref_counter[ref] += 1
        if pred == ref:
            match_counter[pred] += 1
    if not ref_counter:
        return 0.0
    f1_total = 0.0
    classes = set(ref_counter.keys()).union(pred_counter.keys())
    for cls in classes:
        tp = match_counter[cls]
        fp = pred_counter[cls] - tp
        fn = ref_counter[cls] - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_total += f1
    return f1_total / len(classes)
