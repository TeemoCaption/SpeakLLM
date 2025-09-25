"""Download TTS datasets required for CosyVoice2 fine-tuning."""
from __future__ import annotations

from pathlib import Path

import typer

from datasets import load_dataset

from common.utils import load_yaml

app = typer.Typer()


@app.command()
def main(config: Path = typer.Option(..., exists=True), output: Path = typer.Option(Path("data/raw"))) -> None:
    cfg = load_yaml(config)
    output.mkdir(parents=True, exist_ok=True)
    for source in cfg.get("sources", []):
        name = source["name"]
        dataset = source["dataset"]
        split = source.get("split", "train")
        if source.get("streaming", False):
            typer.echo(f"Skipping streaming-only dataset {name}")
            continue
        typer.echo(f"Downloading {dataset}:{split}")
        ds = load_dataset(dataset, split=split)
        out_path = output / f"{name}_{split}.jsonl"
        ds.to_json(out_path)
    typer.echo("TTS download complete")


if __name__ == "__main__":
    app()
