"""Half-duplex pipeline primarily used for debugging."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from asr.whisper_encoder import WhisperEncoder, WhisperEncoderConfig
from asr.connector.model import Connector, ConnectorConfig
from llm.qwen.load_qwen import QwenLoader, QwenLoadConfig
from tts.cosyvoice2.streamer import CosyVoiceStreamer, CosyVoiceStreamerConfig


@dataclass
class HalfDuplexConfig:
    whisper: WhisperEncoderConfig = WhisperEncoderConfig()
    connector: ConnectorConfig = ConnectorConfig()
    qwen: QwenLoadConfig = QwenLoadConfig(lora_r=8)
    cosyvoice: CosyVoiceStreamerConfig = CosyVoiceStreamerConfig()
    sample_rate: int = 16000


class HalfDuplexPipeline:
    def __init__(self, config: HalfDuplexConfig | None = None) -> None:
        self.config = config or HalfDuplexConfig()
        self.encoder = WhisperEncoder(self.config.whisper)
        self.connector = Connector(self.config.connector).to(self.config.qwen.device)
        self.qwen = QwenLoader(self.config.qwen)
        self.tts = CosyVoiceStreamer(self.config.cosyvoice)

    @torch.inference_mode()
    def run(self, waveform: torch.Tensor, sampling_rate: int) -> List[torch.Tensor]:
        hidden = self.encoder.encode(waveform, sampling_rate).unsqueeze(0)
        hidden = self.connector(hidden)
        outputs = self.qwen.model.generate(inputs_embeds=hidden.to(self.qwen.model.device), max_new_tokens=128)
        text = self.qwen.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Assistant: {text}")
        tokens = self.qwen.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        return list(self.tts.generate_stream(tokens))
