"""Phase 3 training for emotion and prosody control."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from llm.policy.duplex_controller import DuplexController
from llm.qwen.load_qwen import QwenLoader, QwenLoadConfig
from tokenizer.voila_tokenizer import VoilaConfig, VoilaTokenizer


@dataclass
class Phase3Config:
    qwen: QwenLoadConfig = QwenLoadConfig(lora_r=8)
    lr: float = 3e-5
    weight_decay: float = 0.01
    steps: int = 18000
    gradient_accumulation: int = 8
    device: str = "cuda"
    flow_weight: float = 0.5
    rvq_weight: float = 0.3
    prosody_weight: float = 0.2
    speaker_weight: float = 0.1
    voila: VoilaConfig = VoilaConfig()


class Phase3Trainer:
    def __init__(self, config: Phase3Config) -> None:
        self.config = config
        self.qwen = QwenLoader(config.qwen)
        self.model = self.qwen.model
        self.voila = VoilaTokenizer(config.voila)
        self.controller = DuplexController().to(config.device)
        params = list(self.model.parameters()) + list(self.controller.parameters())
        self.optimizer = AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.global_step = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        outputs = self.model(
            input_ids=batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            labels=batch["labels"].to(self.model.device),
        )
        flow_loss = outputs.loss
        audio = batch["waveform"].to(self.model.device)
        sr = int(batch.get("sampling_rate", 24000))
        codes = self.voila.encode(audio, sr)
        rvq_loss = codes["acoustic"].float().pow(2).mean()
        prosody_loss = self.mse(batch["prosody_pred"], batch["prosody_target"])
        speaker_loss = self.ce(batch["speaker_logits"], batch["speaker_ids"]) if "speaker_logits" in batch else torch.tensor(0.0, device=self.model.device)
        total = (
            self.config.flow_weight * flow_loss
            + self.config.rvq_weight * rvq_loss
            + self.config.prosody_weight * prosody_loss
            + self.config.speaker_weight * speaker_loss
        )
        total.backward()
        return {
            "flow": float(flow_loss.item()),
            "rvq": float(rvq_loss.item()),
            "prosody": float(prosody_loss.item()),
            "speaker": float(speaker_loss.item()) if speaker_loss.requires_grad else 0.0,
            "total": float(total.item()),
        }

    def fit(self, dataloader: DataLoader) -> None:
        self.optimizer.zero_grad()
        for step, batch in enumerate(dataloader, start=1):
            metrics = self.train_step(batch)
            if step % self.config.gradient_accumulation == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                if self.global_step % 20 == 0:
                    print(f"[Phase3] step={self.global_step} total={metrics['total']:.4f}")
                if self.global_step >= self.config.steps:
                    break
