"""Build JSONL manifests from dataset mixture configs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from datasets.loaders import iter_weighted_samples, load_mixture  # noqa: E402

app = typer.Typer()


@app.command()
def main(config: Path = typer.Option(..., exists=True), output: Path = typer.Option(...)) -> None:
    mixture = load_mixture(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as sink:
        for item in iter_weighted_samples(mixture):
            dataset = item["dataset"]
            name = item["name"]
            for idx, sample in enumerate(dataset):
                record = {
                    "source": name,
                    "weight": item["weight"],
                    "utt_id": sample.get("id", f"{name}_{idx}"),
                    "text": sample.get("text") or sample.get("sentence"),
                    "audio": sample.get("audio"),
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    typer.echo(f"Manifest stored at {output}")


if __name__ == "__main__":
    app()
