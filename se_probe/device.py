"""Device autodetection helpers (CUDA -> MPS -> CPU).

Lets the same code run on A100 nodes (research), Apple Silicon (demo), and
CPU-only machines without per-platform conditionals. Some PyTorch ops have
no native MPS kernel; ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is set automatically
when MPS is selected so those ops fall back to CPU. The speech-enhancement
magnitude pathways used here have been verified safe under that fallback.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

__all__ = ["get_device", "device_info"]


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Return a ``torch.device`` autodetected as CUDA -> MPS -> CPU.

    ``prefer`` may be one of ``"cuda"``, ``"mps"``, ``"cpu"`` to override the
    preference order. ``None`` (default) uses the autodetect chain. If a
    preferred backend is unavailable, falls back through the remainder of the
    chain rather than raising.
    """
    if prefer == "cpu":
        return torch.device("cpu")

    want_cuda = prefer in (None, "cuda")
    if want_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    want_mps = prefer in (None, "mps")
    if want_mps and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")

    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    """Pretty-printable summary of a torch.device for notebook bootstrap."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        return f"Detected device: cuda ({name}, {total_gb:.0f} GB)"
    if device.type == "mps":
        return "Detected device: mps (Apple Silicon)"
    return "Detected device: cpu"
