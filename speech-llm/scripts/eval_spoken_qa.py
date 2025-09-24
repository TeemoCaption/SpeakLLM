import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate spoken QA EM/F1 and latency")
    parser.add_argument("--references", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    args = parser.parse_args()

    refs = [json.loads(line) for line in Path(args.references).open("r", encoding="utf-8")]
    preds = [json.loads(line) for line in Path(args.predictions).open("r", encoding="utf-8")]

    em = 0
    f1 = 0.0
    latency = []
    for ref, pred in zip(refs, preds):
        em += int(ref["answer"].strip() == pred["answer"].strip())
        latency.append(pred.get("latency_ms", 0.0))
        ref_tokens = ref["answer"].split()
        pred_tokens = pred["answer"].split()
        shared = len(set(ref_tokens) & set(pred_tokens))
        precision = shared / max(len(pred_tokens), 1)
        recall = shared / max(len(ref_tokens), 1)
        f1 += 2 * precision * recall / max(precision + recall, 1e-8)

    total = max(len(refs), 1)
    metrics = {
        "exact_match": em / total,
        "f1": f1 / total,
        "latency_ms_avg": sum(latency) / max(len(latency), 1),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
