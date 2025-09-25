"""Phase 1 semi-duplex supervised fine-tuning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from llm.qwen.load_qwen import QwenLoader, QwenLoadConfig
from llm.qwen.lora import save_lora


@dataclass
class Phase1Config:
    qwen: QwenLoadConfig = QwenLoadConfig(lora_r=8)
    lr_lora: float = 1e-4
    weight_decay: float = 0.01
    steps: int = 20000
    device: str = "cuda"
    gradient_accumulation: int = 8


class Phase1Trainer:
    def __init__(self, config: Phase1Config) -> None:
        self.config = config
        self.loader = QwenLoader(config.qwen)
        self.model = self.loader.model
        self.tokenizer = self.loader.tokenizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr_lora, weight_decay=config.weight_decay)
        self.global_step = 0

    def _forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = batch["input_ids"].to(self.model.device)
        attention = batch["attention_mask"].to(self.model.device)
        labels = batch["labels"].to(self.model.device)
        outputs = self.model(input_ids=inputs, attention_mask=attention, labels=labels)
        return outputs.loss

    def fit(self, dataloader: DataLoader) -> None:
        self.model.train()
        self.optimizer.zero_grad()
        for step, batch in enumerate(dataloader, start=1):
            loss = self._forward(batch) / self.config.gradient_accumulation
            loss.backward()
            if step % self.config.gradient_accumulation == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                if self.global_step % 50 == 0:
                    print(f"[Phase1] step={self.global_step} loss={loss.item():.4f}")
                if self.global_step >= self.config.steps:
                    break

    def save(self, output_dir: str) -> None:
        if hasattr(self.model, "peft_config"):
            save_lora(self.model, output_dir)
        else:
            self.model.save_pretrained(output_dir)
