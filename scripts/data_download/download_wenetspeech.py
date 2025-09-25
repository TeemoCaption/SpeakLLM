"""Wrapper for the official WenetSpeech download script."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer

REPO_URL = "https://github.com/wenet-e2e/WenetSpeech"

app = typer.Typer()


def _ensure_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        typer.echo(f"Updating existing repo at {repo_dir}...")
        subprocess.run(["git", "pull"], cwd=repo_dir, check=False)
    else:
        typer.echo(f"Cloning WenetSpeech repo into {repo_dir}...")
        subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)


@app.command()
def main(
    password: str = typer.Option(..., prompt=True, hide_input=True),
    repo_path: Path = typer.Option(Path("external/WenetSpeech"), help="Where to clone/update the official repo"),
    download_dir: Path = typer.Option(Path("data/wenetspeech/download"), help="Where to store encrypted archives"),
    untar_dir: Path = typer.Option(Path("data/wenetspeech/raw"), help="Where to extract the dataset"),
    modelscope: bool = typer.Option(False, help="Use ModelScope mirrors instead of Tencent"),
    stage: int = typer.Option(0, help="Forward to download script stage argument"),
) -> None:
    repo_path = repo_path.expanduser().resolve()
    download_dir = download_dir.expanduser().resolve()
    untar_dir = untar_dir.expanduser().resolve()

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    untar_dir.mkdir(parents=True, exist_ok=True)

    _ensure_repo(repo_path)

    safebox = repo_path / "SAFEBOX"
    safebox.mkdir(parents=True, exist_ok=True)
    (safebox / "password").write_text(password.strip(), encoding="utf-8")
    typer.echo(f"Password stored at {safebox / 'password'}")

    env = os.environ.copy()
    if modelscope:
        env["modelscope"] = "true"
    if stage:
        env["stage"] = str(stage)

    cmd = [
        "bash",
        "utils/download_wenetspeech.sh",
        str(download_dir),
        str(untar_dir),
    ]
    typer.echo(f"Running {' '.join(cmd)}")
    subprocess.run(cmd, cwd=repo_path, check=True, env=env)
    typer.echo("WenetSpeech download finished. Update configs/data/asr_zh.yaml to point manifests to the extracted path if needed.")


if __name__ == "__main__":
    app()
