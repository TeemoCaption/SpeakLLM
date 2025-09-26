"""Full duplex pipeline coordinating streaming ASR, LLM, and TTS."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Generator, Iterable

import torch
import torch.nn.functional as F

from asr.connector.model import Connector, ConnectorConfig
from asr.lid_head import LIDHead, LIDHeadConfig
from asr.whisper_encoder import WhisperEncoder, WhisperEncoderConfig
from common.vad import StreamingVAD, VADConfig
from llm.policy.duplex_controller import DuplexController, DuplexControllerConfig
from llm.qwen.load_qwen import QwenLoader, QwenLoadConfig
from peft import PeftModel
from tts.cosyvoice2.streamer import CosyVoiceStreamer, CosyVoiceStreamerConfig


@dataclass
class FullDuplexConfig:
    whisper: WhisperEncoderConfig = WhisperEncoderConfig()
    connector: ConnectorConfig = ConnectorConfig()
    vad: VADConfig = VADConfig()
    controller: DuplexControllerConfig = DuplexControllerConfig()
    qwen: QwenLoadConfig = QwenLoadConfig(lora_r=8)
    cosyvoice: CosyVoiceStreamerConfig = CosyVoiceStreamerConfig()
    lid_head: LIDHeadConfig | None = None


class FullDuplexPipeline:
    def __init__(self, config: FullDuplexConfig | None = None) -> None:
        self.config = config or FullDuplexConfig()
        self.encoder = WhisperEncoder(self.config.whisper)
        self.connector = Connector(self.config.connector).to(self.config.qwen.device)
        self.qwen = QwenLoader(self.config.qwen)
        self.controller = DuplexController(self.config.controller).to(self.config.qwen.device)
        self.tts = CosyVoiceStreamer(self.config.cosyvoice)
        self.vad = StreamingVAD(self.config.vad)

        self.lid_head: LIDHead | None = None
        if self.config.lid_head is not None:
            lid_cfg = self.config.lid_head
            self.lid_head = LIDHead(lid_cfg)
            if lid_cfg.ckpt_path:
                checkpoint = torch.load(lid_cfg.ckpt_path, map_location="cpu")
                state = checkpoint.get("state_dict", checkpoint)
                self.lid_head.load_state_dict(state)
            self.lid_head.eval()
            self.lid_head.requires_grad_(False)

        self._current_lang = "zh"
        self._last_lang = "zh"
        self._hysteresis = max(1, self.lid_head.cfg.hysteresis_threshold) if self.lid_head else 1
        self._votes: Deque[str] = deque(maxlen=self._hysteresis)
        self._lid_frames: Deque[torch.Tensor] = deque()
        self._lid_samples = 0
        self._stride_samples = 0

        self._set_adapter(self._current_lang)

    @staticmethod
    def _choose_lang_tag(text: str, lid: str | None) -> str:
        if lid:
            lid_lower = lid.lower()
            if lid_lower.startswith("zh"):
                return "zh"
            if lid_lower.startswith("en"):
                return "en"
        stripped = text.strip()
        if not stripped:
            return "zh"
        en_chars = sum(1 for ch in stripped if ch.isascii() and ch.isalpha())
        ratio = en_chars / max(1, len(stripped))
        return "en" if ratio > 0.7 else "zh"

    def _set_adapter(self, lang: str) -> None:
        model = self.qwen.model
        if not isinstance(model, PeftModel):
            return
        if hasattr(model, "enable_adapter_layers"):
            model.enable_adapter_layers()
        if lang.startswith("zh"):
            adapter_name = "zh"
        elif lang.startswith("en"):
            adapter_name = "en"
        else:
            if hasattr(model, "disable_adapter_layers"):
                model.disable_adapter_layers()
            return
        if adapter_name in model.peft_config:
            model.set_adapter(adapter_name)

    @torch.inference_mode()
    def run_stream(
        self,
        audio_frames: Iterable[torch.Tensor],
        sampling_rate: int,
    ) -> Generator[torch.Tensor, None, None]:
        buffer: list[torch.Tensor] = []
        self._lid_frames.clear()
        self._lid_samples = 0
        self._stride_samples = 0
        self._votes.clear()

        frame_ms = int(self.config.vad.frame_ms)
        window_samples = max(1, int(frame_ms * sampling_rate / 1000))
        stride_samples = window_samples
        min_confidence = 0.0
        if self.lid_head is not None:
            window_ms = max(frame_ms, self.lid_head.cfg.lid_window_ms)
            stride_ms = max(frame_ms, self.lid_head.cfg.lid_stride_ms)
            window_samples = max(window_samples, int(window_ms * sampling_rate / 1000))
            stride_samples = max(1, int(stride_ms * sampling_rate / 1000))
            min_confidence = self.lid_head.cfg.min_confidence

        for is_speech, frame in self.vad.process_stream(audio_frames):
            if is_speech:
                buffer.append(frame)
                if self.lid_head is not None:
                    self._lid_frames.append(frame)
                    self._lid_samples += frame.numel()
                    self._stride_samples += frame.numel()
                    while self._lid_frames and self._lid_samples > window_samples:
                        removed = self._lid_frames.popleft()
                        self._lid_samples -= removed.numel()
                    if self._stride_samples >= stride_samples:
                        self._stride_samples = 0
                        window_audio = torch.cat(list(self._lid_frames)) if self._lid_frames else frame
                        self._maybe_switch_adapter(window_audio, sampling_rate, min_confidence)
            elif buffer:
                utterance = torch.cat(buffer)
                buffer.clear()
                response_chunks = self._process_utterance(utterance, sampling_rate)
                for chunk in response_chunks:
                    yield chunk

    def _process_utterance(self, waveform: torch.Tensor, sampling_rate: int) -> Iterable[torch.Tensor]:
        encoded = self.encoder.encode(waveform, sampling_rate).unsqueeze(0)
        lang_tag = self._detect_language(encoded, waveform, sampling_rate)
        self._set_adapter(lang_tag)
        hidden = self.connector(encoded)
        _ = self.controller(hidden.mean(dim=1))
        action, probs = self.controller.sample_action(hidden.mean(dim=1))
        if action == 1:  # hold
            return []
        outputs = self.qwen.model.generate(inputs_embeds=hidden.to(self.qwen.model.device), max_new_tokens=128)
        text = self.qwen.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Assistant: {text} (lang={lang_tag}, controller={action}, p={probs.tolist()})")
        tokens = self.qwen.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        return list(self.tts.generate_stream(tokens))

    def _detect_language(
        self,
        encoded: torch.Tensor,
        waveform: torch.Tensor,
        sampling_rate: int,
    ) -> str:
        if self.lid_head is not None:
            lang, confidence = self._detect_from_head(encoded)
            if confidence >= self.lid_head.cfg.min_confidence:
                self._last_lang = lang
                return lang
        transcript, lid = self.encoder.transcribe(waveform, sampling_rate)
        lang = self._choose_lang_tag(transcript, lid)
        self._last_lang = lang
        return lang

    def _maybe_switch_adapter(self, window_audio: torch.Tensor, sampling_rate: int, min_confidence: float) -> None:
        if self.lid_head is None:
            return
        encoded = self.encoder.encode(window_audio, sampling_rate).unsqueeze(0)
        candidate, confidence = self._detect_from_head(encoded)
        if confidence < min_confidence:
            candidate = self._last_lang
        self._votes.append(candidate)
        if len(self._votes) == self._votes.maxlen and len(set(self._votes)) == 1:
            new_lang = self._votes[0]
            if new_lang != self._current_lang:
                self._current_lang = new_lang
                self._last_lang = new_lang
                self._set_adapter(new_lang)

    def _detect_from_head(self, encoded: torch.Tensor) -> tuple[str, float]:
        assert self.lid_head is not None
        encoded = encoded.to(self.lid_head.cfg.device)
        logits = self.lid_head(encoded)
        probs = F.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax().item())
        lang = self.lid_head.languages[idx]
        confidence = float(probs[idx].item())
        return lang, confidence
