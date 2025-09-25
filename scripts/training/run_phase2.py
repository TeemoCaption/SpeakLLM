"""CLI wrapper for Phase 2 full duplex training."""
from __future__ import annotations

import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.utils import load_yaml  # noqa: E402
from training.phase2_duplex_trainer import Phase2Config, Phase2Trainer  # noqa: E402

app = typer.Typer()


@app.command()
def main(config: Path = typer.Option(..., exists=True, help="YAML config for Phase 2")) -> None:
    cfg_dict = load_yaml(config)
    trainer = Phase2Trainer(Phase2Config())
    typer.echo(f"Loaded config from {config} (keys={list(cfg_dict.keys())})")
    typer.echo("Phase 2 trainer initialized (placeholder run)")


if __name__ == "__main__":
    app()
