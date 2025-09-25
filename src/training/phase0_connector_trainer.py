"""Phase 0 training loop for Whisper connector distillation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from asr.connector.losses import ctc_loss, dtw_loss, kl_divergence, roundtrip_consistency_loss
from asr.connector.model import Connector, ConnectorConfig
from llm.qwen.load_qwen import QwenLoader, QwenLoadConfig


@dataclass
class Phase0Config:
    connector: ConnectorConfig = ConnectorConfig()
    qwen: QwenLoadConfig = QwenLoadConfig(lora_r=0)
    lr_connector: float = 1e-4
    lr_heads: float = 1e-5
    steps: int = 5000
    device: str = "cuda"
    vocab_size: int = 4000


class Phase0Trainer:
    def __init__(self, config: Phase0Config) -> None:
        self.config = config
        self.connector = Connector(config.connector).to(config.device)
        self.teacher = QwenLoader(config.qwen).model
        self.teacher.requires_grad_(False)
        self.ctc_head = nn.Linear(config.connector.hidden_dim, config.vocab_size).to(config.device)
        parameters = list(self.connector.parameters()) + list(self.ctc_head.parameters())
        self.opt = AdamW(parameters, lr=config.lr_connector)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.connector.train()
        hidden = self.connector(batch["features"].to(self.config.device))
        student_logits = self.ctc_head(hidden)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs_embeds=hidden).logits  # type: ignore[call-arg]
        losses = {
            "kl": kl_divergence(student_logits, teacher_logits),
            "ctc": ctc_loss(student_logits, batch["targets"], batch["logit_lengths"], batch["target_lengths"]),
        }
        if "target_embeddings" in batch:
            losses["dtw"] = dtw_loss(hidden.squeeze(0), batch["target_embeddings"].squeeze(0))
        if "asr_logits" in batch and "tts_logits" in batch:
            losses["roundtrip"] = roundtrip_consistency_loss(batch["asr_logits"], batch["tts_logits"])
        total = sum(losses.values())
        self.opt.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self.connector.parameters(), 1.0)
        self.opt.step()
        log = {name: float(loss.item()) for name, loss in losses.items()}
        log["total"] = float(total.item()) if isinstance(total, torch.Tensor) else float(total)
        return log

    def fit(self, dataloader: DataLoader) -> None:
        for step, batch in enumerate(dataloader, start=1):
            metrics = self.train_step(batch)
            if step >= self.config.steps:
                break
            if step % 50 == 0:
                print(f"[Phase0] step={step} total={metrics['total']:.4f}")
