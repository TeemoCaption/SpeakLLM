"""Resample audio files and optionally perform VAD segmentation."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import torch
import typer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.audio_io import ResampleConfig, frame_audio, load_audio, save_audio  # noqa: E402
from common.vad import StreamingVAD, VADConfig  # noqa: E402

app = typer.Typer()


@app.command()
def main(
    manifest: Path = typer.Option(..., exists=True, help="JSONL manifest containing audio_path"),
    output_dir: Path = typer.Option(Path("data/resampled")),
    target_sr: int = typer.Option(16000),
    vad: bool = typer.Option(False, help="Enable VAD segmentation"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    resample_cfg = ResampleConfig(target_sr=target_sr)
    vad_processor = StreamingVAD(VADConfig(sample_rate=target_sr)) if vad else None
    segments: List[dict] = []
    with open(manifest, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            audio_path = Path(record["audio_path"])
            waveform, _ = load_audio(str(audio_path), target_sr)
            out_path = output_dir / audio_path.name
            save_audio(str(out_path), waveform, target_sr)
            if vad_processor:
                frame_len = vad_processor.frame_samples
                hop = frame_len
                frames = frame_audio(waveform, frame_len, hop)
                detected = vad_processor.detect_segments([frame for frame in frames])
                for start, end in detected:
                    segments.append({
                        "utt_id": record.get("utt_id"),
                        "start_frame": start,
                        "end_frame": end,
                        "path": str(out_path),
                    })
    if segments:
        seg_path = output_dir / "segments.jsonl"
        with open(seg_path, "w", encoding="utf-8") as sink:
            for seg in segments:
                sink.write(json.dumps(seg) + "\n")
        typer.echo(f"Saved VAD segments to {seg_path}")
    typer.echo("Resampling complete")


if __name__ == "__main__":
    app()
