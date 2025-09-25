"""Voice activity detection helpers wrapping webrtcvad."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

import torch
import webrtcvad


@dataclass
class VADConfig:
    aggressiveness: int = 2
    sample_rate: int = 16000
    frame_ms: int = 20
    min_speech_frames: int = 10
    max_pause_frames: int = 6


class StreamingVAD:
    """Streaming VAD that yields (is_speech, frame_tensor)."""

    def __init__(self, config: VADConfig) -> None:
        self.config = config
        self.vad = webrtcvad.Vad(config.aggressiveness)
        self.frame_samples = int(config.sample_rate * config.frame_ms / 1000)
        self.speech_run = 0
        self.pause_run = 0

    def _frame_to_bytes(self, frame: torch.Tensor) -> bytes:
        frame16 = (frame.clamp(-1, 1) * 32767.0).short().tobytes()
        return frame16

    def process_stream(self, frames: Iterable[torch.Tensor]) -> Iterator[Tuple[bool, torch.Tensor]]:
        for frame in frames:
            if frame.numel() != self.frame_samples:
                raise ValueError("Unexpected frame length for VAD")
            decision = self.vad.is_speech(self._frame_to_bytes(frame), self.config.sample_rate)
            if decision:
                self.speech_run += 1
                self.pause_run = 0
            else:
                self.pause_run += 1
                if self.pause_run > self.config.max_pause_frames:
                    self.speech_run = 0
            yield decision, frame

    def detect_segments(self, frames: Iterable[torch.Tensor]) -> List[Tuple[int, int]]:
        """Return start/end frame indices for detected speech segments."""
        segments: List[Tuple[int, int]] = []
        active = False
        start_idx = 0
        for idx, (decision, _) in enumerate(self.process_stream(frames)):
            if decision and not active and self.speech_run >= self.config.min_speech_frames:
                active = True
                start_idx = idx - self.config.min_speech_frames + 1
            elif not decision and active and self.pause_run > self.config.max_pause_frames:
                active = False
                segments.append((start_idx, idx))
        if active:
            segments.append((start_idx, idx))  # type: ignore[name-defined]
        return segments
