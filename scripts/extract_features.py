import argparse
import json
from pathlib import Path

import torch
import torchaudio


def compute_log_mel(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    return torchaudio.compliance.kaldi.fbank(
        waveform,
        sample_frequency=sample_rate,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        window_type="hamming",
        dither=0.0,
    )


def process_item(entry: dict, output_dir: Path) -> None:
    waveform, sr = torchaudio.load(entry["path"])
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        sr = 16000
    mel = compute_log_mel(waveform, sr)
    uid = Path(entry["path"]).stem
    output_path = output_dir / f"{uid}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mel, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract log-mel features for Whisper encoder")
    parser.add_argument("--manifest", type=str, required=True, help="Manifest JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write torch tensors")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, "r", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)
            process_item(entry, output_dir)


if __name__ == "__main__":
    main()
