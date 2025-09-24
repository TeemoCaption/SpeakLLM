import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from voila_tokenizer import VoilaTokenizer  # type: ignore
except ImportError:
    VoilaTokenizer = None


def quantize_waveform(tokenizer, audio_path: Path) -> dict:
    audio, sr = sf.read(audio_path)
    if sr != tokenizer.sample_rate:
        raise ValueError(f"Expected sample rate {tokenizer.sample_rate}, got {sr}")
    codes, scales = tokenizer.encode(audio)
    return {"codes": codes, "scales": scales}


def process_manifest(manifest_path: Path, tokenizer_path: Path, output_dir: Path) -> None:
    if VoilaTokenizer is None:
        raise ImportError("Install Voila tokenizer package or add it to PYTHONPATH.")
    tokenizer = VoilaTokenizer.from_pretrained(tokenizer_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)
            audio_path = Path(entry["path"])
            quantized = quantize_waveform(tokenizer, audio_path)
            uid = audio_path.stem
            np.savez_compressed(output_dir / f"{uid}.npz", **quantized)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize audio into Voila tokenizer codes")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    process_manifest(Path(args.manifest), Path(args.tokenizer), Path(args.output_dir))


if __name__ == "__main__":
    main()
