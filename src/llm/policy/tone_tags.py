"""Tone tag utilities for tone-aware alignment."""
from __future__ import annotations

from typing import List


TONE_TAGS = ["<tone1>", "<tone2>", "<tone3>", "<tone4>", "<tone0>", "<pause>"]
LANG_TAGS = ["<lang_zh>", "<lang_en>", "<lang_yue>"]


def inject_tone_tags(tokens: List[str], tones: List[str], pauses: List[int]) -> List[str]:
    augmented: List[str] = []
    tone_iter = iter(tones)
    pause_iter = iter(pauses)
    for token in tokens:
        augmented.append(token)
        tone = next(tone_iter, None)
        if tone:
            augmented.append(tone)
        pause = next(pause_iter, 0)
        if pause:
            augmented.append("<pause>")
    return augmented


def language_prefix(lang: str) -> str:
    tag = f"<lang_{lang}>"
    if tag not in LANG_TAGS:
        raise ValueError(f"Unsupported language tag: {lang}")
    return tag
