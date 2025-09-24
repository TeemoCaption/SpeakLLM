"""
Device utilities for consistent accelerator detection across the project.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch


def list_available_cuda_devices() -> list[str]:
    """Return a list of visible CUDA device identifiers."""
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]


def _normalize_preferred(preferred: Optional[str]) -> str:
    if preferred is None:
        return "auto"
    if isinstance(preferred, int):
        return f"cuda:{preferred}"
    return str(preferred).strip().lower()


def resolve_device(
    preferred: Optional[str] = None,
    allow_cpu_fallback: bool = True,
    logger: Optional[logging.Logger] = None,
) -> torch.device:
    """Resolve to a valid torch.device, falling back gracefully when needed."""

    normalized = _normalize_preferred(preferred)
    available_cuda = torch.cuda.is_available()
    available_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

    def _log(msg: str) -> None:
        if logger:
            logger.warning(msg)

    if normalized in {"cpu", "cpu:0"}:
        return torch.device("cpu")

    if normalized.startswith("cuda") or normalized.isdigit():
        if not available_cuda:
            if allow_cpu_fallback:
                _log("Requested CUDA device but CUDA is unavailable. Falling back to CPU.")
                return torch.device("cpu")
            raise RuntimeError("CUDA requested but no GPU is available.")

        # Parse index (cuda, cuda:0, 0, etc.)
        index = 0
        if normalized.isdigit():
            index = int(normalized)
        elif ":" in normalized:
            try:
                index = int(normalized.split(":", maxsplit=1)[1])
            except ValueError:
                index = 0

        if index < 0 or index >= torch.cuda.device_count():
            _log(
                f"CUDA device index {index} is invalid. Using cuda:0 instead."
            )
            index = 0
        return torch.device(f"cuda:{index}")

    if normalized.startswith("mps"):
        if available_mps:
            return torch.device("mps")
        if allow_cpu_fallback:
            _log("Requested MPS device but Metal backend is unavailable. Falling back to CPU.")
            return torch.device("cpu")
        raise RuntimeError("MPS requested but not available.")

    if normalized in {"auto", ""}:
        if available_cuda:
            return torch.device("cuda:0")
        if available_mps:
            return torch.device("mps")
        return torch.device("cpu")

    if allow_cpu_fallback:
        _log(f"Unknown device '{preferred}'. Falling back to CPU.")
        return torch.device("cpu")

    raise ValueError(f"Unsupported device specification: {preferred}")


def describe_device(device: torch.device) -> str:
    """Human readable device description including GPU name when possible."""
    if device.type == "cuda":
        index = device.index or 0
        try:
            name = torch.cuda.get_device_name(index)
        except RuntimeError:
            name = "Unknown CUDA device"
        return f"cuda:{index} ({name})"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"


def ensure_device(device: torch.device, logger: Optional[logging.Logger] = None) -> torch.device:
    """Ensure the resolved device is usable, otherwise fall back to CPU."""
    if device.type == "cuda" and not torch.cuda.is_available():
        if logger:
            logger.warning("CUDA device became unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        if logger:
            logger.warning("MPS device became unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return device


def resolve_device_with_info(
    preferred: Optional[str] = None,
    allow_cpu_fallback: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[torch.device, str]:
    """Resolve a device and provide a descriptive string for logging."""
    device = resolve_device(preferred=preferred, allow_cpu_fallback=allow_cpu_fallback, logger=logger)
    device = ensure_device(device, logger=logger)
    return device, describe_device(device)
