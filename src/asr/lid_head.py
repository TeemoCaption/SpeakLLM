from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class LIDHeadConfig:
    in_dim: int = 1024
    languages: List[str] | None = None
    ckpt_path: str = ""
    lid_window_ms: int = 800
    lid_stride_ms: int = 300
    hysteresis_threshold: int = 2
    min_confidence: float = 0.6
    device: str = "cpu"


class LIDHead(nn.Module):
    def __init__(self, cfg: LIDHeadConfig):
        super().__init__()
        self.cfg = cfg
        languages = cfg.languages or []
        if not languages:
            raise ValueError("LIDHead requires至少一個語言標籤")
        self.languages = list(languages)
        self.proj = nn.Linear(cfg.in_dim, len(self.languages))
        self.to(cfg.device)

    def forward(self, enc_feats: torch.Tensor) -> torch.Tensor:
        if enc_feats.dim() == 2:
            enc_feats = enc_feats.unsqueeze(0)
        x = enc_feats.mean(dim=1)
        return self.proj(x)
