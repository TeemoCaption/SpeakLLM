"""Phase 2 full duplex training loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from asr.connector.losses import ctc_loss, dtw_loss
from asr.connector.model import Connector, ConnectorConfig
from common.metrics.overlap_consistency import compute_overlap_penalty
from llm.policy.duplex_controller import DuplexController, DuplexControllerConfig
from llm.qwen.load_qwen import QwenLoader, QwenLoadConfig
from tokenizer.voila_tokenizer import VoilaConfig, VoilaTokenizer


@dataclass
class Phase2Config:
    connector: ConnectorConfig = ConnectorConfig()
    qwen: QwenLoadConfig = QwenLoadConfig(lora_r=8)
    controller: DuplexControllerConfig = DuplexControllerConfig()
    lr: float = 5e-5
    weight_decay: float = 0.01
    steps: int = 24000
    gradient_accumulation: int = 8
    device: str = "cuda"
    ctc_weight: float = 0.4
    dtw_weight: float = 0.4
    ce_weight: float = 0.8
    overlap_weight: float = 0.3
    rvq_weight: float = 0.2
    voila: VoilaConfig = VoilaConfig()


class Phase2Trainer:
    def __init__(self, config: Phase2Config) -> None:
        self.config = config
        self.connector = Connector(config.connector).to(config.device)
        self.controller = DuplexController(config.controller).to(config.device)
        self.qwen = QwenLoader(config.qwen)
        self.model = self.qwen.model
        self.voila = VoilaTokenizer(config.voila)
        params = list(self.connector.parameters()) + list(self.controller.parameters()) + list(self.model.parameters())
        self.opt = AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.global_step = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.connector.train()
        self.controller.train()
        hidden = self.connector(batch["audio_features"].to(self.config.device))
        outputs = self.model(inputs_embeds=hidden, labels=batch["labels"].to(self.config.device))
        ce = outputs.loss
        ctc = ctc_loss(outputs.logits, batch["targets"], batch["logit_lengths"], batch["target_lengths"])
        dtw = dtw_loss(hidden.squeeze(0), batch["target_embeddings"].squeeze(0))
        overlap = torch.tensor(0.0, device=self.config.device)
        if "overlap_reference" in batch and "overlap_prediction" in batch:
            penalty = compute_overlap_penalty(
                batch["overlap_reference"].tolist(), batch["overlap_prediction"].tolist()
            )
            overlap = torch.tensor(penalty, device=self.config.device)
        rvq = torch.tensor(0.0, device=self.config.device)
        if "waveform" in batch:
            audio = batch["waveform"].to(self.config.device)
            sr = int(batch.get("sampling_rate", 24000))
            codes = self.voila.encode(audio, sr)
            rvq = codes["acoustic"].float().abs().mean()
        total = (
            self.config.ce_weight * ce
            + self.config.ctc_weight * ctc
            + self.config.dtw_weight * dtw
            + self.config.overlap_weight * overlap
            + self.config.rvq_weight * rvq
        )
        total.backward()
        return {
            "ce": float(ce.item()),
            "ctc": float(ctc.item()),
            "dtw": float(dtw.item()),
            "overlap": float(overlap.item()),
            "rvq": float(rvq.item()),
            "total": float(total.item()),
        }

    def fit(self, dataloader: DataLoader) -> None:
        self.opt.zero_grad()
        for step, batch in enumerate(dataloader, start=1):
            metrics = self.train_step(batch)
            if step % self.config.gradient_accumulation == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()
                self.global_step += 1
                if self.global_step % 20 == 0:
                    print(f"[Phase2] step={self.global_step} total={metrics['total']:.4f}")
                if self.global_step >= self.config.steps:
                    break
