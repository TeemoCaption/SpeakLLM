import argparse
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

STREAM_CHUNK_SECONDS = 0.32
STREAM_STRIDE_SECONDS = 0.16


class AudioStreamer:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.buffer: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)
        self.running = False

    def _callback(self, indata, frames, time_info, status):  # noqa: D401
        if status:
            print(status)
        if self.running:
            self.buffer.put_nowait(indata.copy())

    def start(self):
        self.running = True
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=int(self.sample_rate * STREAM_STRIDE_SECONDS),
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()

    def get_chunk(self) -> np.ndarray:
        data = [self.buffer.get()]
        while len(data) * STREAM_STRIDE_SECONDS < STREAM_CHUNK_SECONDS:
            data.append(self.buffer.get())
        return np.concatenate(data, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Streaming inference prototype")
    parser.add_argument("--checkpoint", type=str, required=False, help="Model checkpoint path")
    args = parser.parse_args()

    # TODO: Load model + tokenizer once training pipeline is ready.
    print("Streaming inference stub. Integrate with trained model later.")
    streamer = AudioStreamer()
    streamer.start()
    try:
        while True:
            chunk = streamer.get_chunk()
            # Placeholder: send chunk through Whisper encoder and Qwen decoder.
            print(f"Received chunk shape: {chunk.shape}")
            time.sleep(STREAM_CHUNK_SECONDS)
    except KeyboardInterrupt:
        print("Stopping stream...")
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()
