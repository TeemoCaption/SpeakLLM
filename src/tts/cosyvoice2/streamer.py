"""Streaming wrapper for CosyVoice2 inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Optional

import torch


@dataclass
class CosyVoiceStreamerConfig:
    checkpoint: str = "cosvoice/cosyvoice2-0.5b"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    output_chunk_ms: int = 320
    sample_rate: int = 24000


class CosyVoiceStreamer:
    def __init__(self, config: CosyVoiceStreamerConfig | None = None) -> None:
        self.config = config or CosyVoiceStreamerConfig()
        self.model = None  # Lazy load to avoid heavy import on module load

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        try:
            from cosyvoice2 import CosyVoice2
        except ImportError as exc:  # pragma: no cover - dependency optional
            raise RuntimeError("CosyVoice2 package is required for streaming") from exc
        self.model = CosyVoice2.from_pretrained(self.config.checkpoint).to(
            self.config.device, dtype=self.config.dtype
        )
        self.model.eval()

    @torch.inference_mode()
    def generate_stream(
        self,
        text_tokens: Iterable[int],
        prosody_embedding: Optional[torch.Tensor] = None,
    ) -> Generator[torch.Tensor, None, None]:
        self._ensure_model()
        assert self.model is not None
        chunk_samples = int(self.config.sample_rate * self.config.output_chunk_ms / 1000)
        buffer = torch.empty(0, device=self.config.device, dtype=self.config.dtype)
        for chunk in self.model.stream(text_tokens, prosody_embedding=prosody_embedding):
            buffer = torch.cat((buffer, chunk), dim=0)
            while buffer.numel() >= chunk_samples:
                yield buffer[:chunk_samples].to(torch.float32).cpu()
                buffer = buffer[chunk_samples:]
        if buffer.numel():
            yield buffer.to(torch.float32).cpu()
