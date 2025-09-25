"""Local CLI client to interact with the duplex server."""
from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import numpy as np
import typer
import websockets

app = typer.Typer()


async def _send_audio(uri: str, audio_path: Path) -> None:
    with wave.open(str(audio_path), "rb") as wav:
        sr = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    async with websockets.connect(uri) as ws:
        await ws.send(audio.tobytes())
        try:
            while True:
                chunk = await asyncio.wait_for(ws.recv(), timeout=1.0)
                typer.echo(f"Received {len(chunk)} bytes of audio")
        except asyncio.TimeoutError:
            typer.echo("No more chunks; closing connection")


@app.command()
def stream(uri: str = "ws://localhost:8080/ws", audio: Path = typer.Argument(...)) -> None:
    """Stream a local WAV file to the server and print chunk sizes."""
    asyncio.run(_send_audio(uri, audio))


if __name__ == "__main__":
    app()
