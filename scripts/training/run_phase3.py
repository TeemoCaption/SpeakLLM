"""CLI wrapper for Phase 3 emotion/prosody training."""
from __future__ import annotations

import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.utils import load_yaml  # noqa: E402
from training.phase3_emotion_trainer import Phase3Config, Phase3Trainer  # noqa: E402

app = typer.Typer()


@app.command()
def main(config: Path = typer.Option(..., exists=True, help="YAML config for Phase 3")) -> None:
    cfg_dict = load_yaml(config)
    trainer = Phase3Trainer(Phase3Config())
    typer.echo(f"Loaded config from {config} (keys={list(cfg_dict.keys())})")
    typer.echo("Phase 3 trainer initialized (placeholder run)")


if __name__ == "__main__":
    app()
