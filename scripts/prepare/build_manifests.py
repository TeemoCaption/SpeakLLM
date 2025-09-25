"""Build JSONL manifests from dataset mixture configs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import typer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from datasets.loaders import iter_weighted_samples, load_mixture  # noqa: E402

app = typer.Typer()


def _extract_audio_metadata(sample: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    audio = sample.get("audio")
    if isinstance(audio, dict):
        path = audio.get("path") or audio.get("file")
        if path:
            meta["audio_path"] = path
        url = audio.get("url")
        if url:
            meta["audio_url"] = url
    if "wav" in sample and isinstance(sample["wav"], dict):
        wav = sample["wav"]
        if wav.get("path"):
            meta["audio_path"] = wav["path"]
        if wav.get("bytes"):
            meta["has_bytes"] = True
    if "__url__" in sample:
        meta["archive_url"] = sample["__url__"]
    if "__key__" in sample:
        meta["archive_key"] = sample["__key__"]
    return meta


def _build_record(name: str, weight: float, idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "source": name,
        "weight": weight,
        "utt_id": sample.get("utt_id")
        or sample.get("id")
        or sample.get("uid")
        or f"{name}_{idx}",
        "text": sample.get("text")
        or sample.get("sentence")
        or sample.get("transcription")
        or sample.get("translation", {}).get("text")
        or "",
    }
    record.update(_extract_audio_metadata(sample))
    return record


@app.command()
def main(config: Path = typer.Option(..., exists=True), output: Path = typer.Option(...)) -> None:
    mixture = load_mixture(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output, "w", encoding="utf-8") as sink:
        for item in iter_weighted_samples(mixture):
            dataset = item["dataset"]
            name = item["name"]
            for idx, sample in enumerate(dataset):
                record = _build_record(name, item["weight"], idx, sample)
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
    typer.echo(f"Manifest stored at {output} with {total} entries")


if __name__ == "__main__":
    app()
