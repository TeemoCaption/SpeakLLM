import argparse
import json
from pathlib import Path

import soundfile as sf
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download Common Voice subset and export WAV files")
    parser.add_argument("--version", type=str, default="mozilla-foundation/common_voice_17_0")
    parser.add_argument("--subset", type=str, default="zh-TW")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.version, args.subset, split=args.split)
    output_dir = Path(args.output)
    audio_dir = output_dir / "wav"
    manifest_path = output_dir / f"{args.split}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as fout:
        for idx, item in enumerate(dataset):
            wav_path = audio_dir / f"{idx:08d}.wav"
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav_path, item["audio"]["array"], item["audio"]["sampling_rate"], subtype="PCM_16")
            record = {
                "path": str(wav_path),
                "text": item.get("sentence", ""),
                "lang": args.subset,
                "speaker_id": item.get("client_id", "unknown"),
                "dataset": "common_voice",
                "split": args.split,
            }
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
