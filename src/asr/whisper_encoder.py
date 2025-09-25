"""Frozen Whisper encoder wrapper used for feature extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration


@dataclass
class WhisperEncoderConfig:
    checkpoint: str = "openai/whisper-medium"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    chunk_size: int = 1280


class WhisperEncoder:
    def __init__(self, config: WhisperEncoderConfig | None = None) -> None:
        self.config = config or WhisperEncoderConfig()
        self.model = WhisperForConditionalGeneration.from_pretrained(self.config.checkpoint)
        self.encoder = self.model.model.encoder
        self.encoder.requires_grad_(False)
        self.model.to(self.config.device, dtype=self.config.dtype)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.config.checkpoint)

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        features = self.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = features.input_features.to(self.config.device, dtype=self.config.dtype)
        encoder_outputs = self.encoder(input_features)
        return encoder_outputs.last_hidden_state.squeeze(0)

    def chunk_encode(self, audio: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """Chunk streaming encoding for low-latency inference."""
        step = self.config.chunk_size
        chunks = []
        for start in range(0, audio.size(-1), step):
            segment = audio[..., start : start + step]
            encoded = self.encode(segment, sampling_rate)
            chunks.append(encoded)
        return torch.cat(chunks, dim=0)
