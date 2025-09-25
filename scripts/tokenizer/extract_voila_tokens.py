"""Generate Voila tokenizer codes for an audio manifest."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import typer

from tokenizer.voila_tokenizer import VoilaTokenizer, VoilaConfig

app = typer.Typer()


@app.command()
def main(
    manifest: Path = typer.Option(..., exists=True, help="JSONL manifest with audio paths"),
    output: Path = typer.Option(..., help="Output JSONL file with RVQ codes"),
    checkpoint: str = typer.Option("maitrix-org/Voila-Tokenizer", help="Model checkpoint"),
    device: str = typer.Option("cuda"),
) -> None:
    tokenizer = VoilaTokenizer(VoilaConfig(checkpoint=checkpoint))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "r", encoding="utf-8") as handle, open(output, "w", encoding="utf-8") as sink:
        for line in handle:
            record = json.loads(line)
            audio = torch.load(record["audio_tensor"])
            codes = tokenizer.encode(audio.to(device), record.get("sample_rate", 24000))
            sink.write(json.dumps({
                "utt_id": record.get("utt_id"),
                "semantic": codes["semantic"].int().tolist(),
                "acoustic": codes["acoustic"].int().tolist(),
            }) + "\n")
    typer.echo(f"Saved codes to {output}")


if __name__ == "__main__":
    app()
