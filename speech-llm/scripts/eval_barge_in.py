import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate barge-in metrics")
    parser.add_argument("--log", type=str, required=True, help="Streaming event log JSONL")
    args = parser.parse_args()

    success = 0
    over_interrupt = 0
    false_stop = 0
    total = 0

    with Path(args.log).open("r", encoding="utf-8") as fin:
        for line in fin:
            event = json.loads(line)
            total += 1
            if event.get("label") == "barge_in_success":
                success += 1
            if event.get("label") == "over_interrupt":
                over_interrupt += 1
            if event.get("label") == "false_stop":
                false_stop += 1

    metrics = {
        "barge_in_success_rate": success / max(total, 1),
        "over_interrupt_rate": over_interrupt / max(total, 1),
        "false_stop_rate": false_stop / max(total, 1),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
