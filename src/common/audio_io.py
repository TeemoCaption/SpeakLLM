"""Audio IO utilities shared by ASR, TTS, and streaming runtime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio


@dataclass
class ResampleConfig:
    target_sr: int
    dtype: torch.dtype = torch.float32


def load_audio(path: str, target_sr: int | None = None) -> Tuple[torch.Tensor, int]:
    """Load audio with optional resampling."""
    wav, sr = sf.read(path, dtype="float32")
    tensor = torch.from_numpy(wav).float()
    if tensor.ndim == 2:
        tensor = tensor.mean(dim=1)
    if target_sr and sr != target_sr:
        tensor = torchaudio.functional.resample(tensor, sr, target_sr)
        sr = target_sr
    return tensor, sr


def save_audio(path: str, audio: torch.Tensor, sample_rate: int) -> None:
    audio_np = audio.detach().cpu().numpy()
    sf.write(path, audio_np, sample_rate)


def resample_audio(audio: torch.Tensor, orig_sr: int, config: ResampleConfig) -> torch.Tensor:
    """Resample to the configured rate."""
    if orig_sr == config.target_sr:
        return audio.to(config.dtype)
    resampled = torchaudio.functional.resample(audio, orig_sr, config.target_sr)
    return resampled.to(config.dtype)


def frame_audio(audio: torch.Tensor, frame_size: int, hop_size: int) -> torch.Tensor:
    """Frame audio into overlapping chunks."""
    return audio.unfold(0, frame_size, hop_size)


def overlap_add(frames: torch.Tensor, hop_size: int) -> torch.Tensor:
    """Reconstruct audio via overlap-add."""
    frame_size = frames.size(-1)
    output_length = hop_size * (frames.size(0) - 1) + frame_size
    result = torch.zeros(output_length, dtype=frames.dtype, device=frames.device)
    for i, frame in enumerate(frames):
        start = i * hop_size
        result[start : start + frame_size] += frame
    return result


def chunk_stream(audio_iter: Iterable[torch.Tensor], chunk_size: int) -> Iterator[torch.Tensor]:
    """Yield fixed-size chunks from streaming audio tensors."""
    buffer: List[torch.Tensor] = []
    buffered = 0
    for piece in audio_iter:
        buffer.append(piece)
        buffered += piece.numel()
        while buffered >= chunk_size:
            merged = torch.cat(buffer)
            yield merged[:chunk_size]
            buffer = [merged[chunk_size:]] if merged.numel() > chunk_size else []
            buffered = buffer[0].numel() if buffer else 0
    if buffer:
        yield torch.cat(buffer)


def compute_rms(audio: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(audio ** 2)))


def detect_energy_spikes(audio: torch.Tensor, sample_rate: int, threshold_db: float = -35.0) -> np.ndarray:
    """Return mask where instantaneous energy crosses the threshold."""
    eps = 1e-8
    energy = audio.pow(2).unfold(0, sample_rate // 50, sample_rate // 100).mean(dim=-1)
    db = 10 * torch.log10(energy + eps)
    mask = db > threshold_db
    return mask.cpu().numpy()
