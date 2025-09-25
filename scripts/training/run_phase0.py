"""CLI wrapper for Phase 0 connector training."""
from __future__ import annotations

import sys
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.utils import load_yaml  # noqa: E402
from training.phase0_connector_trainer import Phase0Config, Phase0Trainer  # noqa: E402

app = typer.Typer()


@app.command()
def main(config: Path = typer.Option(..., exists=True, help="YAML config for Phase 0")) -> None:
    cfg_dict = load_yaml(config)
    trainer = Phase0Trainer(Phase0Config())
    typer.echo(f"Loaded config from {config} (keys={list(cfg_dict.keys())})")
    typer.echo("Phase 0 trainer initialized (placeholder run)")


if __name__ == "__main__":
    app()
