import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class CodeDataset(Dataset):
    def __init__(self, manifest_path: Path):
        self.records = [json.loads(line) for line in manifest_path.open("r", encoding="utf-8")]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
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

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.transformer(x)
        return self.proj(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train audio decoder placeholder")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    dataset = CodeDataset(Path(args.manifest))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleDecoder(vocab_size=8192, d_model=896)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        for batch in dataloader:
            logits = model(batch)
            loss = criterion(logits.view(-1, logits.size(-1)), batch.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    main()
