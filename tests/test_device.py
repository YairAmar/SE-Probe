"""Tests for ``se_probe.device.get_device`` / ``device_info``.

Test level: **UNIT**.
Rationale: ``get_device`` is pure logic over ``torch.cuda.is_available`` /
``torch.backends.mps.is_available`` plus an env-var side effect. There is
no I/O, no fixture file, no DB. A unit test is the highest-fidelity test
that meaningfully exercises the function — going to integration would mean
faking torch, which is exactly what we should not do. We use the *real*
torch state of the runner (CPU on CI, MPS on Apple Silicon, CUDA on a
research box) and assert against that observed truth. The MPS-fallback
env-var assertion is gated on actually hitting the MPS branch so it
remains deterministic across runners.
"""
from __future__ import annotations

import os

import pytest
import torch

from se_probe.device import device_info, get_device


class TestGetDevice:
    """Functional behaviour of ``get_device``."""

    def test_returns_torch_device(self):
        d = get_device()
        assert isinstance(d, torch.device), (
            f"get_device() must return a torch.device, got {type(d).__name__}"
        )
        assert d.type in {"cuda", "mps", "cpu"}, (
            f"get_device() returned unexpected backend {d.type!r}; "
            "expected one of cuda/mps/cpu"
        )

    def test_prefer_cpu_returns_cpu_unconditionally(self):
        d = get_device(prefer="cpu")
        assert d == torch.device("cpu"), (
            f"prefer='cpu' must return CPU device regardless of available "
            f"accelerators; got {d}"
        )

    def test_prefer_cpu_overrides_cuda_and_mps(self):
        # Even when CUDA or MPS is available on the runner, prefer='cpu'
        # must short-circuit to CPU.
        d = get_device(prefer="cpu")
        assert d.type == "cpu"

    def test_default_autodetect_priority(self):
        # When prefer is None, the documented priority is CUDA -> MPS -> CPU.
        d = get_device()
        if torch.cuda.is_available():
            assert d.type == "cuda", (
                "CUDA is available; autodetect must prefer it over MPS/CPU"
            )
        elif torch.backends.mps.is_available():
            assert d.type == "mps", (
                "MPS is available and CUDA is not; autodetect must prefer MPS over CPU"
            )
        else:
            assert d.type == "cpu"

    def test_prefer_cuda_falls_back_when_unavailable(self):
        # prefer='cuda' should not raise on a machine without CUDA — it
        # should fall back through the chain rather than blow up.
        d = get_device(prefer="cuda")
        if torch.cuda.is_available():
            assert d.type == "cuda"
        else:
            assert d.type in {"mps", "cpu"}

    def test_prefer_mps_falls_back_when_unavailable(self):
        d = get_device(prefer="mps")
        if torch.backends.mps.is_available():
            assert d.type == "mps"
        else:
            assert d.type in {"cuda", "cpu"}

    def test_mps_path_sets_pytorch_enable_mps_fallback(self, monkeypatch):
        # Run only on machines where the MPS branch is actually reached.
        if not torch.backends.mps.is_available():
            pytest.skip("MPS unavailable on this runner; cannot exercise MPS branch")
        if torch.cuda.is_available():
            pytest.skip(
                "CUDA shadows MPS in autodetect; force the MPS branch via "
                "prefer='mps' instead"
            )
        monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
        d = get_device(prefer="mps")
        assert d.type == "mps"
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1", (
            "Selecting MPS must set PYTORCH_ENABLE_MPS_FALLBACK=1 so unsupported "
            "ops (e.g. linalg.eigh) fall back to CPU rather than raising"
        )

    def test_mps_path_does_not_clobber_existing_env(self, monkeypatch):
        if not torch.backends.mps.is_available():
            pytest.skip("MPS unavailable")
        if torch.cuda.is_available():
            pytest.skip("CUDA shadows MPS in autodetect")
        monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
        get_device(prefer="mps")
        # Implementation uses os.environ.setdefault — pre-existing value wins.
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0", (
            "get_device must not overwrite a user-set PYTORCH_ENABLE_MPS_FALLBACK; "
            "it should only set the default when unset"
        )


class TestDeviceInfo:
    """``device_info`` returns a human-readable summary for notebook bootstrap."""

    def test_cpu_string(self):
        info = device_info(torch.device("cpu"))
        assert info == "Detected device: cpu"

    def test_mps_string(self):
        info = device_info(torch.device("mps"))
        assert info == "Detected device: mps (Apple Silicon)"

    def test_cuda_string_shape(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable on this runner")
        info = device_info(get_device(prefer="cuda"))
        assert info.startswith("Detected device: cuda ("), (
            f"CUDA device_info should start with 'Detected device: cuda (', "
            f"got: {info!r}"
        )
        assert "GB" in info, (
            f"CUDA device_info should embed total memory in GB; got: {info!r}"
        )

    def test_returns_str(self):
        # Regardless of backend, output must be a plain str (no f-string objects, etc.).
        for d in (torch.device("cpu"), torch.device("mps")):
            assert isinstance(device_info(d), str)
