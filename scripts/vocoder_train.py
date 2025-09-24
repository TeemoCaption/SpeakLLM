"""Lightweight placeholder trainer for the audio decoder/vocoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


class CodeDataset(Dataset):
    def __init__(self, manifest_path: Path):
        self.records = [json.loads(line) for line in manifest_path.open("r", encoding="utf-8")]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        record = self.records[idx]
        return torch.tensor(record["audio_codes"], dtype=torch.long)


class SimpleDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_model * 4, batch_first=True),
            num_layers=12,
        )
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.embed(tokens)
        x = self.transformer(x)
        return self.proj(x)


def load_config_section(config_path: Optional[str], section: str) -> Optional[dict]:
    if not config_path:
        return None
    if yaml is None:
        raise ImportError("PyYAML is required to load config sections")
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if section in data:
        return data[section]
    if "stages" in data and section in data["stages"]:
        return data["stages"][section]
    raise KeyError(f"Section '{section}' not found in {config_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple audio decoder placeholder")
    parser.add_argument("--manifest", type=str, required=True, help="Sequence manifest with audio_codes field")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--config", type=str, default=None, help="Optional train config (e.g., configs/train.yml)")
    parser.add_argument("--section", type=str, default="audio_decoder", help="Section key inside the config file")
    parser.add_argument("--batch_size", type=int, default=16, help="Override batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Override learning rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_section = load_config_section(args.config, args.section)

    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs

    if cfg_section:
        batch_size = cfg_section.get("batching", {}).get("batch_size", batch_size)
        lr = cfg_section.get("optimizer", {}).get("lr", lr)
        steps = cfg_section.get("train_steps")
        if isinstance(steps, int) and steps > 0:
            epochs = max(steps // 1000, 1)

    dataset = CodeDataset(Path(args.manifest))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleDecoder(vocab_size=8192, d_model=896)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            logits = model(batch)
            loss = criterion(logits.view(-1, logits.size(-1)), batch.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"epoch {epoch + 1}/{epochs} - loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
