import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import requests


def worker(endpoint: str, payload: dict, timeout: float) -> float:
    start = time.time()
    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return -1.0
    return (time.time() - start) * 1000


def main():
    parser = argparse.ArgumentParser(description="Simple HTTP load tester for speech LLM service")
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    payload_template = {"text": "你好, 請介紹今天的天氣", "lang": "zh-TW"}
    latencies = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker, args.endpoint, payload_template, args.timeout) for _ in range(args.requests)]
        for fut in futures:
            lat = fut.result()
            if lat >= 0:
                latencies.append(lat)

    if latencies:
        latencies.sort()
        avg = sum(latencies) / len(latencies)
        p95 = latencies[int(len(latencies) * 0.95) - 1]
        print({"avg_ms": avg, "p95_ms": p95, "requests": len(latencies)})
    else:
        print("No successful requests recorded.")


if __name__ == "__main__":
    main()
