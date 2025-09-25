"""Download TTS datasets required for CosyVoice2 fine-tuning."""
from __future__ import annotations

from pathlib import Path

import typer

from datasets import load_dataset

from common.utils import load_yaml

app = typer.Typer()


@app.command()
def main(config: Path = typer.Option(..., exists=True)) -> None:
    cfg = load_yaml(config)
    for source in cfg.get("sources", []):
        name = source["name"]
        dataset = source["dataset"]
        split = source.get("split", "train")
        streaming = source.get("streaming", False)
        kwargs = {
            "split": split,
            "streaming": streaming,
            "verification_mode": "no_checks",
            "trust_remote_code": True,
        }
        if source.get("config"):
            kwargs["name"] = source["config"]
        typer.echo(f"Prefetching {dataset}:{split} (streaming={streaming})")
        ds = load_dataset(dataset, **kwargs)
        if not streaming:
            _ = ds.shard(num_shards=1, index=0)
        typer.echo(f"Ready: {name}")
    typer.echo("Datasets checked/cached. Use scripts/prepare/build_manifests.py to create manifests.")


if __name__ == "__main__":
    app()
