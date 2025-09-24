import argparse
import json
from pathlib import Path

import jiwer


def compute_metrics(references, hypotheses):
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveWhiteSpace(replace_with=" "),
    ])
    wer = jiwer.wer(references, hypotheses, truth_transform=transformation, hypothesis_transform=transformation)
    cer = jiwer.cer(references, hypotheses, truth_transform=transformation, hypothesis_transform=transformation)
    return wer, cer


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR WER/CER")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    args = parser.parse_args()

    manifest = [json.loads(line) for line in Path(args.manifest).open("r", encoding="utf-8")]
    preds = [json.loads(line) for line in Path(args.predictions).open("r", encoding="utf-8")]
    references = [item["text"] for item in manifest]
    hypotheses = [item["text"] for item in preds]
    wer, cer = compute_metrics(references, hypotheses)
    print(json.dumps({"wer": wer, "cer": cer}, indent=2))


if __name__ == "__main__":
    main()
