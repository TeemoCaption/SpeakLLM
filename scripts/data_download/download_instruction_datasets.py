"""Download instruction-following speech datasets for SFT."""
from __future__ import annotations

from pathlib import Path

import typer

from datasets import load_dataset

from common.utils import load_yaml

app = typer.Typer()


@app.command()
def main(config: Path = typer.Option(..., exists=True), output: Path = typer.Option(Path("data/raw"))) -> None:
    cfg = load_yaml(config)
    sources = cfg.get("sources", [])
    output.mkdir(parents=True, exist_ok=True)
    for source in sources:
        name = source["name"]
        dataset = source["dataset"]
        split = source.get("split", "train")
        typer.echo(f"Preparing {dataset}:{split}")
        ds = load_dataset(dataset, split=split, streaming=source.get("streaming", False))
        if source.get("streaming", False):
            typer.echo(f"Streaming dataset {name}; skipping materialization")
            continue
        ds.to_json(output / f"{name}_{split}.jsonl")
    typer.echo("Instruction dataset download complete")


if __name__ == "__main__":
    app()
