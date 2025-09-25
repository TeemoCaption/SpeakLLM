"""Full duplex pipeline coordinating streaming ASR, LLM, and TTS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable

import torch

from asr.whisper_encoder import WhisperEncoder, WhisperEncoderConfig
from asr.connector.model import Connector, ConnectorConfig
from common.vad import StreamingVAD, VADConfig
from llm.policy.duplex_controller import DuplexController, DuplexControllerConfig
from llm.qwen.load_qwen import QwenLoader, QwenLoadConfig
from tts.cosyvoice2.streamer import CosyVoiceStreamer, CosyVoiceStreamerConfig


@dataclass
class FullDuplexConfig:
    whisper: WhisperEncoderConfig = WhisperEncoderConfig()
    connector: ConnectorConfig = ConnectorConfig()
    vad: VADConfig = VADConfig()
    controller: DuplexControllerConfig = DuplexControllerConfig()
    qwen: QwenLoadConfig = QwenLoadConfig(lora_r=8)
    cosyvoice: CosyVoiceStreamerConfig = CosyVoiceStreamerConfig()


class FullDuplexPipeline:
    def __init__(self, config: FullDuplexConfig | None = None) -> None:
        self.config = config or FullDuplexConfig()
        self.encoder = WhisperEncoder(self.config.whisper)
        self.connector = Connector(self.config.connector).to(self.config.qwen.device)
        self.qwen = QwenLoader(self.config.qwen)
        self.controller = DuplexController(self.config.controller).to(self.config.qwen.device)
        self.tts = CosyVoiceStreamer(self.config.cosyvoice)
        self.vad = StreamingVAD(self.config.vad)

    @torch.inference_mode()
    def run_stream(
        self,
        audio_frames: Iterable[torch.Tensor],
        sampling_rate: int,
    ) -> Generator[torch.Tensor, None, None]:
        buffer = []
        for is_speech, frame in self.vad.process_stream(audio_frames):
            if is_speech:
                buffer.append(frame)
            elif buffer:
                utterance = torch.cat(buffer)
                buffer.clear()
                response_chunks = self._process_utterance(utterance, sampling_rate)
                for chunk in response_chunks:
                    yield chunk
        if buffer:
            utterance = torch.cat(buffer)
            for chunk in self._process_utterance(utterance, sampling_rate):
                yield chunk

    def _process_utterance(self, waveform: torch.Tensor, sampling_rate: int) -> Iterable[torch.Tensor]:
        hidden = self.encoder.encode(waveform, sampling_rate).unsqueeze(0)
        hidden = self.connector(hidden)
        _ = self.controller(hidden.mean(dim=1))
        action, probs = self.controller.sample_action(hidden.mean(dim=1))
        if action == 1:  # hold
            return []
        outputs = self.qwen.model.generate(inputs_embeds=hidden.to(self.qwen.model.device), max_new_tokens=128)
        text = self.qwen.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Assistant: {text} (controller={action}, p={probs.tolist()})")
        tokens = self.qwen.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        return list(self.tts.generate_stream(tokens))
