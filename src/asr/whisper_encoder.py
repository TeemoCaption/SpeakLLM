"""Frozen Whisper encoder wrapper used for feature extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer


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
        self.tokenizer = WhisperTokenizer.from_pretrained(self.config.checkpoint)

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

    @torch.inference_mode()
    def transcribe(self, audio: torch.Tensor, sampling_rate: int) -> tuple[str, str | None]:
        features = self.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = features.input_features.to(self.config.device, dtype=self.config.dtype)
        forced_decoder_ids = self.model.generation_config.forced_decoder_ids
        self.model.generation_config.forced_decoder_ids = None
        try:
            generated_ids = self.model.generate(input_features=input_features)
        finally:
            self.model.generation_config.forced_decoder_ids = forced_decoder_ids
        sequence = generated_ids[0]
        text = self.tokenizer.decode(sequence, skip_special_tokens=True).strip()
        lid: str | None = None
        if sequence.numel() > 1:
            token = self.tokenizer.convert_ids_to_tokens(int(sequence[1]))
            if isinstance(token, str) and token.startswith("<|") and token.endswith("|>"):
                lid = token[2:-2]
        return text, lid
