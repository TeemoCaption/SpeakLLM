import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import soundfile as sf


def validate_audio(path: Path, target_sr: int = 16000) -> Dict:
    data, sr = sf.read(path)
    if sr != target_sr:
        raise ValueError(f"Expected sample rate {target_sr}, got {sr} for {path}")
    return {"duration": len(data) / sr, "sample_rate": sr}


def convert_manifest(manifest_in: Path, manifest_out: Path, lang: str, dataset: str) -> None:
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_in.open("r", encoding="utf-8") as fin, manifest_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            audio_path = Path(record["path"])
            meta = validate_audio(audio_path)
            record.update({"lang": record.get("lang", lang), "dataset": dataset})
            record.update(meta)
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize dataset manifests into unified schema")
    parser.add_argument("--input", type=str, required=True, help="Raw manifest (JSONL)")
    parser.add_argument("--output", type=str, required=True, help="Output manifest path")
    parser.add_argument("--lang", type=str, required=True, help="Language code")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_manifest(Path(args.input), Path(args.output), args.lang, args.dataset)


if __name__ == "__main__":
    main()
