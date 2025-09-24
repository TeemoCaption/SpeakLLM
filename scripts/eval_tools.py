"""Evaluation utilities for Speech-LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def cmd_asr(args: argparse.Namespace) -> None:
    import jiwer

    manifest = [json.loads(line) for line in Path(args.manifest).open("r", encoding="utf-8")]
    preds = [json.loads(line) for line in Path(args.predictions).open("r", encoding="utf-8")]

    refs = [item["text"] for item in manifest]
    hyps = [item["text"] for item in preds]

    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveWhiteSpace(replace_with=" "),
    ])
    wer = jiwer.wer(refs, hyps, truth_transform=transformation, hypothesis_transform=transformation)
    cer = jiwer.cer(refs, hyps, truth_transform=transformation, hypothesis_transform=transformation)
    print(json.dumps({"wer": wer, "cer": cer}, indent=2))


def cmd_qa(args: argparse.Namespace) -> None:
    refs = [json.loads(line) for line in Path(args.references).open("r", encoding="utf-8")]
    preds = [json.loads(line) for line in Path(args.predictions).open("r", encoding="utf-8")]

    em = 0
    f1 = 0.0
    latency: List[float] = []

    for ref, pred in zip(refs, preds):
        em += int(ref["answer"].strip() == pred["answer"].strip())
        latency.append(float(pred.get("latency_ms", 0.0)))
        ref_tokens = ref["answer"].split()
        pred_tokens = pred["answer"].split()
        shared = len(set(ref_tokens) & set(pred_tokens))
        precision = shared / max(len(pred_tokens), 1)
        recall = shared / max(len(ref_tokens), 1)
        denom = max(precision + recall, 1e-8)
        f1 += 2 * precision * recall / denom

    total = max(len(refs), 1)
    metrics = {
        "exact_match": em / total,
        "f1": f1 / total,
        "latency_ms_avg": sum(latency) / max(len(latency), 1),
    }
    print(json.dumps(metrics, indent=2))


def cmd_tts(args: argparse.Namespace) -> None:
    import librosa
    import numpy as np

    total_mcd = 0.0
    count = 0
    with Path(args.pairs).open("r", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)
            ref, sr = librosa.load(entry["reference"], sr=None)
            gen, _ = librosa.load(entry["generated"], sr=sr)
            ref_mfcc = librosa.feature.mfcc(ref, sr=sr, n_mfcc=13)
            gen_mfcc = librosa.feature.mfcc(gen, sr=sr, n_mfcc=13)
            min_len = min(ref_mfcc.shape[1], gen_mfcc.shape[1])
            diff = ref_mfcc[:, :min_len] - gen_mfcc[:, :min_len]
            mcd = np.mean(np.sqrt(2.0) * np.sqrt((diff ** 2).sum(axis=0)))
            total_mcd += mcd
            count += 1
    print(json.dumps({"mcd_avg": total_mcd / max(count, 1)}, indent=2))


def cmd_barge(args: argparse.Namespace) -> None:
    success = over = false_stop = 0
    total = 0
    with Path(args.log).open("r", encoding="utf-8") as fin:
        for line in fin:
            event = json.loads(line)
            total += 1
            label = event.get("label")
            if label == "barge_in_success":
                success += 1
            elif label == "over_interrupt":
                over += 1
            elif label == "false_stop":
                false_stop += 1
    metrics = {
        "barge_in_success_rate": success / max(total, 1),
        "over_interrupt_rate": over / max(total, 1),
        "false_stop_rate": false_stop / max(total, 1),
    }
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speech-LLM evaluation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_asr = sub.add_parser("asr", help="Compute WER/CER")
    p_asr.add_argument("--manifest", type=str, required=True)
    p_asr.add_argument("--predictions", type=str, required=True)
    p_asr.set_defaults(func=cmd_asr)

    p_qa = sub.add_parser("qa", help="Evaluate Spoken QA exact match/F1")
    p_qa.add_argument("--references", type=str, required=True)
    p_qa.add_argument("--predictions", type=str, required=True)
    p_qa.set_defaults(func=cmd_qa)

    p_tts = sub.add_parser("tts", help="Compute TTS proxy metrics (MCD)")
    p_tts.add_argument("--pairs", type=str, required=True)
    p_tts.set_defaults(func=cmd_tts)

    p_barge = sub.add_parser("barge", help="Evaluate barge-in event logs")
    p_barge.add_argument("--log", type=str, required=True)
    p_barge.set_defaults(func=cmd_barge)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
