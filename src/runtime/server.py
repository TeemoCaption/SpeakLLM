"""WebSocket/gRPC server entry point for duplex inference."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from common.utils import load_yaml
from pipelines.full_duplex import FullDuplexConfig, FullDuplexPipeline


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    max_clients: int = 16
    config_path: str = "configs/infer/server.yaml"


class InferenceState(BaseModel):
    sampling_rate: int = 16000


class DuplexServer:
    def __init__(self, config: ServerConfig | None = None) -> None:
        self.config = config or ServerConfig()
        runtime_cfg = load_yaml(self.config.config_path)
        pipeline_cfg = FullDuplexConfig()
        self.pipeline = FullDuplexPipeline(pipeline_cfg)
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.websocket("/ws")(self._handle_ws)

    async def _stream_audio(self, websocket: WebSocket) -> AsyncIterator[torch.Tensor]:
        try:
            while True:
                data = await websocket.receive_bytes()
                tensor = torch.frombuffer(data, dtype=torch.float32)
                yield tensor
        except WebSocketDisconnect:
            return

    async def _handle_ws(self, websocket: WebSocket) -> None:
        await websocket.accept()
        state = InferenceState()
        async for chunk in self._stream_audio(websocket):
            for tts_chunk in self.pipeline.run_stream([chunk], state.sampling_rate):
                await websocket.send_bytes(tts_chunk.numpy().tobytes())
        await websocket.close()

    def serve(self) -> None:
        uvicorn.run(self.app, host=self.config.host, port=self.config.port, log_level="info")


if __name__ == "__main__":
    DuplexServer().serve()
