"""Connector projecting Whisper features into Qwen token embeddings."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ConnectorConfig:
    input_dim: int = 768
    hidden_dim: int = 384
    downsample: int = 3
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1


class Downsample(nn.Module):
    def __init__(self, ratio: int) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        if self.ratio <= 1:
            return x
        B, T, C = x.shape
        pad = (self.ratio - T % self.ratio) % self.ratio
        if pad:
            x = nn.functional.pad(x, (0, 0, 0, pad))
            T = x.size(1)
        x = x.view(B, T // self.ratio, self.ratio, C)
        return x.mean(dim=2)


class Connector(nn.Module):
    def __init__(self, config: ConnectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or ConnectorConfig()
        self.downsample = Downsample(self.config.downsample)
        self.proj = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)
        self.norm = nn.LayerNorm(self.config.hidden_dim)

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.downsample(features)
        x = self.proj(x)
        if attention_mask is not None and attention_mask.ndim == 2:
            attn_mask = ~attention_mask.bool()
        else:
            attn_mask = None
        x = self.encoder(x, src_key_padding_mask=attn_mask)
        return self.norm(x)
