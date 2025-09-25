"""Download ASR datasets listed in the YAML config."""
from __future__ import annotations

from pathlib import Path

import typer

from datasets import load_dataset

from common.utils import load_yaml

app = typer.Typer()


@app.command()
def main(config: Path = typer.Option(..., exists=True, help="Path to data YAML config"), output: Path = typer.Option(Path("data/raw"), help="Directory to store downloaded shards")) -> None:
    cfg = load_yaml(config)
    sources = cfg.get("sources", [])
    output.mkdir(parents=True, exist_ok=True)
    for source in sources:
        name = source["name"]
        dataset = source["dataset"]
        split = source.get("split", "train")
        config_name = source.get("config")
        streaming = source.get("streaming", False)
        if streaming:
            typer.echo(f"Skipping streaming-only dataset {name}")
            continue
        kwargs = {"split": split}
        if config_name:
            kwargs["name"] = config_name
        typer.echo(f"Downloading {dataset}:{split}")
        ds = load_dataset(dataset, **kwargs)
        ds.to_json(output / f"{name}_{split}.jsonl")
    typer.echo("Download complete")


if __name__ == "__main__":
    app()
