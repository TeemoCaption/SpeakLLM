from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from datasets import load_dataset

from src.asr.whisper_encoder import WhisperEncoder, WhisperEncoderConfig
from src.asr.lid_head import LIDHead, LIDHeadConfig


TARGET_SR = 16000
BATCH_SIZE = 8
STEPS = 2000


def _resample_waveform(sample: dict) -> torch.Tensor:
    waveform = torch.tensor(sample["audio"]["array"], dtype=torch.float32)
    src_sr = int(sample["audio"]["sampling_rate"])
    if src_sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, src_sr, TARGET_SR)
    return waveform


def _infinite_stream(name: str, config: str, split: str, label: int) -> Generator[Tuple[torch.Tensor, int], None, None]:
    while True:
        dataset = load_dataset(name, config, split=split, streaming=True, trust_remote_code=True)
        for sample in dataset:
            yield _resample_waveform(sample), label


def _encode_batch(encoder: WhisperEncoder, waveforms: Iterable[torch.Tensor]) -> torch.Tensor:
    features = []
    max_len = 0
    for waveform in waveforms:
        feat = encoder.encode(waveform.to(encoder.config.device), TARGET_SR)
        features.append(feat.cpu())
        max_len = max(max_len, feat.size(0))
    if not features:
        raise ValueError("批次內沒有語音樣本")
    dim = features[0].size(1)
    batch = torch.zeros(len(features), max_len, dim, dtype=features[0].dtype)
    for idx, feat in enumerate(features):
        batch[idx, : feat.size(0)] = feat
    return batch


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = WhisperEncoder(WhisperEncoderConfig(device=device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32))
    encoder.model.eval()
    for param in encoder.model.parameters():
        param.requires_grad = False

    lid_cfg = LIDHeadConfig(
        in_dim=1024,
        languages=["zh", "en"],
        ckpt_path="checkpoints/lid_head_zh_en.pt",
        device=device,
    )
    lid_head = LIDHead(lid_cfg)
    optimizer = optim.AdamW(lid_head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    zh_stream = _infinite_stream("google/fleurs", "zh_tw", "train", 0)
    en_stream = _infinite_stream("mozilla-foundation/common_voice_17_0", "en", "train", 1)
    streams = [zh_stream, en_stream]
    per_lang = BATCH_SIZE // len(streams)

    for step in range(1, STEPS + 1):
        batch_waveforms: list[torch.Tensor] = []
        batch_labels: list[int] = []
        for idx, stream in enumerate(streams):
            for _ in range(per_lang):
                waveform, label = next(stream)
                batch_waveforms.append(waveform)
                batch_labels.append(label)
        while len(batch_waveforms) < BATCH_SIZE:
            waveform, label = next(streams[0])
            batch_waveforms.append(waveform)
            batch_labels.append(label)

        features = _encode_batch(encoder, batch_waveforms).to(device)
        logits = lid_head(features)
        labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            probs = F.softmax(logits, dim=-1).mean(dim=0)
            print(f"[LID] step {step}/{STEPS} loss={loss.item():.4f} probs={probs.detach().cpu().tolist()}")

    ckpt_path = Path(lid_cfg.ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": lid_head.state_dict(), "cfg": lid_cfg.__dict__}, ckpt_path)
    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()
