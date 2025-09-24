import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def mix_audio(foreground_path: Path, background_path: Path, snr_db: float) -> np.ndarray:
    fg, sr = sf.read(foreground_path)
    bg, _ = sf.read(background_path)
    min_len = min(len(fg), len(bg))
    fg = fg[:min_len]
    bg = bg[:min_len]
    fg_power = np.mean(fg ** 2)
    bg_power = np.mean(bg ** 2)
    scale = np.sqrt(fg_power / (bg_power * (10 ** (snr_db / 10))))
    return fg + scale * bg, sr


def main():
    parser = argparse.ArgumentParser(description="Synthesize barge-in training examples")
    parser.add_argument("--foreground", type=str, required=True, help="Directory of clean user speech")
    parser.add_argument("--background", type=str, required=True, help="Directory of background speech/noise")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()

    fg_files = list(Path(args.foreground).glob("*.wav"))
    bg_files = list(Path(args.background).glob("*.wav"))
    output_dir = Path(args.output)
    audio_dir = output_dir / "wav"
    manifest_path = output_dir / "barge_in.jsonl"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    with manifest_path.open("w", encoding="utf-8") as fout:
        for idx in range(args.count):
            fg = rng.choice(fg_files)
            bg = rng.choice(bg_files)
            mix, sr = mix_audio(fg, bg, snr_db=rng.uniform(0, 10))
            wav_path = audio_dir / f"mix_{idx:06d}.wav"
            sf.write(wav_path, mix, sr, subtype="PCM_16")
            record = {
                "path": str(wav_path),
                "text": "",
                "lang": "multi",
                "dataset": "duplex_synthetic",
                "split": "train",
                "label": "barge_in",
            }
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
