import argparse
import json
from pathlib import Path

import librosa
import numpy as np


def mel_cepstral_distortion(ref_path: Path, gen_path: Path) -> float:
    ref, sr = librosa.load(ref_path, sr=None)
    gen, _ = librosa.load(gen_path, sr=sr)
    ref_mfcc = librosa.feature.mfcc(ref, sr=sr, n_mfcc=13)
    gen_mfcc = librosa.feature.mfcc(gen, sr=sr, n_mfcc=13)
    min_len = min(ref_mfcc.shape[1], gen_mfcc.shape[1])
    diff = ref_mfcc[:, :min_len] - gen_mfcc[:, :min_len]
    return np.mean(np.sqrt(2) * np.sqrt((diff ** 2).sum(axis=0)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate TTS quality proxies")
    parser.add_argument("--pairs", type=str, required=True, help="JSONL with ref/gen paths")
    args = parser.parse_args()

    total_mcd = 0.0
    count = 0
    with Path(args.pairs).open("r", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)
            mcd = mel_cepstral_distortion(Path(entry["reference"]), Path(entry["generated"]))
            total_mcd += mcd
            count += 1
    print(json.dumps({"mcd_avg": total_mcd / max(count, 1)}, indent=2))


if __name__ == "__main__":
    main()
