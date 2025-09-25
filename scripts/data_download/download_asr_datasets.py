"""Download ASR datasets listed in the YAML config."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import typer

from datasets import load_dataset
from huggingface_hub import login

from common.utils import load_yaml

try:
    import download_wenetspeech
except ImportError:  # pragma: no cover
    download_wenetspeech = None

app = typer.Typer()

def _handle_offline_download(source: dict[str, Any]) -> None:
    offline = source.get("offline_download")
    if not offline:
        return
    script = offline.get("script")
    if script != "wenetspeech":
        typer.echo(f"Unknown offline download script: {script}")
        return
    if download_wenetspeech is None:
        typer.echo("WenetSpeech download helper unavailable; skipping offline download.")
        return
    typer.echo("Starting official WenetSpeech download (large download, ensure 500G free space)...")
    download_wenetspeech.main(
        password=offline.get("password"),
        repo_path=Path(offline.get("repo_path", "external/WenetSpeech")),
        download_dir=Path(offline.get("download_dir", "data/wenetspeech/download")),
        untar_dir=Path(offline.get("extract_dir", "data/wenetspeech/raw")),
        modelscope=bool(offline.get("modelscope", False)),
        stage=int(offline.get("stage", 0)),
    )


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
        _handle_offline_download(source)
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
