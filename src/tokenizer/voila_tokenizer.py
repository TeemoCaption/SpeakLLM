"""Wrapper utilities for the Voila neural audio tokenizer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import AutoProcessor


@dataclass
class VoilaConfig:
    checkpoint: str = "maitrix-org/Voila-Tokenizer"
    semantic_level: int = 0
    acoustic_levels: int = 7


class VoilaTokenizer:
    def __init__(self, config: VoilaConfig | None = None) -> None:
        self.config = config or VoilaConfig()
        self.processor = AutoProcessor.from_pretrained(self.config.checkpoint)

    def encode(self, audio: torch.Tensor, sampling_rate: int) -> Dict[str, torch.Tensor]:
        """Return RVQ codes for semantic and acoustic layers."""
        with torch.inference_mode():
            processed = self.processor(audio=audio, sampling_rate=sampling_rate, return_tensors="pt")
        codes = processed.get("codes")
        if codes is None:
            raise RuntimeError("Voila processor did not return RVQ codes")
        semantic = codes[:, :, self.config.semantic_level]
        acoustic = codes[:, :, 1 : 1 + self.config.acoustic_levels]
        return {"semantic": semantic, "acoustic": acoustic}

    def decode(self, codes: Dict[str, torch.Tensor], sampling_rate: int) -> torch.Tensor:
        stacked = torch.cat((codes["semantic"].unsqueeze(-1), codes["acoustic"]), dim=-1)
        inputs = {"codes": stacked}
        with torch.inference_mode():
            decoder = getattr(self.processor, "audio_decoder", None)
            if decoder is None:
                raise RuntimeError("Voila processor missing audio_decoder attribute")
            waveform = decoder(**inputs, sampling_rate=sampling_rate)
        return waveform.squeeze(0)

    def pad_codes(self, codes: torch.Tensor, max_length: int) -> torch.Tensor:
        if codes.size(1) >= max_length:
            return codes[:, :max_length]
        pad_size = max_length - codes.size(1)
        pad = torch.full((codes.size(0), pad_size), fill_value=-1, dtype=codes.dtype, device=codes.device)
        return torch.cat((codes, pad), dim=1)
