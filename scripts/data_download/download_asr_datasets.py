"""Download ASR datasets listed in the YAML config."""
from __future__ import annotations

from pathlib import Path

import typer

import os

from datasets import load_dataset
from huggingface_hub import login

from common.utils import load_yaml

app = typer.Typer()


@app.command()
def main(config: Path = typer.Option(..., exists=True, help="Path to data YAML config")) -> None:
    token = os.getenv("HF_TOKEN")
    if token:
        try:
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass
    cfg = load_yaml(config)
    sources = cfg.get("sources", [])
    for source in sources:
        name = source["name"]
        dataset = source["dataset"]
        split = source.get("split", "train")
        config_name = source.get("config")
        streaming = source.get("streaming", False)
        kwargs = {
            "split": split,
            "streaming": streaming,
            "verification_mode": "no_checks",
            "trust_remote_code": True,
        }
        if config_name:
            kwargs["name"] = config_name
        if token:
            kwargs["token"] = token
        typer.echo(f"Prefetching {dataset}:{split} (streaming={streaming})")
        ds = load_dataset(dataset, **kwargs)
        if not streaming:
            _ = ds.shard(num_shards=1, index=0)
        typer.echo(f"Ready: {name}")
    typer.echo("Datasets checked/cached. Use scripts/prepare/build_manifests.py to create manifests.")


if __name__ == "__main__":
    app()
