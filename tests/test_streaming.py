from __future__ import annotations

import torch

from common.audio_io import chunk_stream


def test_chunk_stream_shapes() -> None:
    audio_iter = [torch.ones(3200), torch.ones(1600)]
    chunks = list(chunk_stream(audio_iter, chunk_size=1600))
    assert len(chunks) == 3
    assert all(chunk.shape[0] <= 1600 for chunk in chunks)
