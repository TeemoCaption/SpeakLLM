"""Normalize Traditional/Simplified Chinese text and attach tone tags."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import g2pc
import opencc
import typer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from llm.policy.tone_tags import TONE_TAGS  # noqa: E402

app = typer.Typer()


@app.command()
def main(
    manifest: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(...),
    conversion: str = typer.Option("t2s"),
) -> None:
    converter = opencc.OpenCC(f"{conversion}.json")
    phoneme = g2pc.G2p()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "r", encoding="utf-8") as src, open(output, "w", encoding="utf-8") as dst:
        for line in src:
            record = json.loads(line)
            text = converter.convert(record.get("text", ""))
            pinyin = phoneme(text)
            tones = [f"<tone{item[-1] if item[-1].isdigit() else 5}>" for item in pinyin]
            tones = [tone if tone in TONE_TAGS else "<tone0>" for tone in tones]
            record.update({"text": text, "pinyin": pinyin, "tones": tones})
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
    typer.echo(f"Normalized manifest written to {output}")


if __name__ == "__main__":
    app()
