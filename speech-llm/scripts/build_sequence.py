import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

SPECIAL_TOKENS = {
    "audio_start": "<AUDIO_IN>",
    "audio_end": "<AUDIO_IN_END>",
    "assistant_audio_start": "<ASSISTANT_AUDIO>",
    "assistant_audio_end": "</ASSISTANT_AUDIO>",
    "assistant_text_start": "<ASSISTANT_TEXT>",
    "assistant_text_end": "</ASSISTANT_TEXT>",
}


def load_codes(codes_dir: Path, uid: str) -> Dict:
    path = codes_dir / f"{uid}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing codebook file {path}")
    return {"codes": np.load(path)["codes"].tolist()}


def assemble_sequence(entry: Dict, codes: Dict) -> Dict:
    # TODO: integrate real tokenizer and attention mask construction.
    sequence = {
        "input_ids": [SPECIAL_TOKENS["audio_start"], SPECIAL_TOKENS["audio_end"]],
        "audio_codes": codes["codes"],
        "loss_mask": [0, 0],
        "meta": {
            "text": entry.get("text", ""),
            "lang": entry.get("lang", "unknown"),
            "dataset": entry.get("dataset", "unknown"),
        },
    }
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Build interleaved multimodal sequences")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--codes_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    codes_dir = Path(args.codes_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)
            uid = Path(entry["path"]).stem
            codes = load_codes(codes_dir, uid)
            sequence = assemble_sequence(entry, codes)
            fout.write(json.dumps(sequence, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
