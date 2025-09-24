"""Utility helpers for SpeechLLM."""

from .device_utils import (
    describe_device,
    ensure_device,
    list_available_cuda_devices,
    resolve_device,
    resolve_device_with_info,
)
from .whisper_compat import get_sample_rate, load_audio

__all__ = [
    "describe_device",
    "ensure_device",
    "get_sample_rate",
    "list_available_cuda_devices",
    "load_audio",
    "resolve_device",
    "resolve_device_with_info",
]
