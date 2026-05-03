"""Tests pinning the lazy-import contract of ``se_probe.metrics``.

Test level: **INTEGRATION** — runs against the real installed package.
Rationale: the contract under test is *what happens at import time*.
Mocking would defeat the purpose. The whole point of these tests is to
catch a regression where someone reintroduces a top-level
``import torch_stoi`` (or any other optional GPU-evaluator dep) and
silently breaks ``from se_probe.metrics import c50`` on a fresh
``pip install -e .`` venv that does not include the optional extras.

These tests would have caught reviewer-flagged Critical #2 before it
reached CI.
"""
from __future__ import annotations

import importlib
import sys

import pytest


class TestMetricsTopLevelImport:
    """The ``metrics`` module must import using only core pyproject deps."""

    def test_module_imports(self):
        # Force re-import to catch caching from prior tests.
        sys.modules.pop("se_probe.metrics", None)
        m = importlib.import_module("se_probe.metrics")
        assert hasattr(m, "c50"), "metrics module must expose c50() at top level"
        assert hasattr(m, "compute_audio_metrics"), (
            "metrics module must expose compute_audio_metrics() at top level"
        )

    def test_top_level_names_importable(self):
        # Each of these is in metrics.__all__ — they must be importable
        # without any optional GPU-evaluator dep installed.
        from se_probe.metrics import (  # noqa: F401
            c50,
            compute_audio_metrics,
            drr,
            sisdr,
        )

    def test_no_torch_stoi_dependency_at_top_level(self):
        # We cannot trivially assert "torch_stoi was not imported" because
        # an earlier test may have imported it. Instead, assert that
        # se_probe.metrics is reachable in a subinterpreter-equivalent
        # state by clearing modules and re-importing without torch_stoi
        # available.
        sys.modules.pop("se_probe.metrics", None)
        # Inject a None sentinel for torch_stoi so any top-level
        # `import torch_stoi` would raise immediately. If the module
        # imports without raising, the dep is correctly deferred.
        prev = sys.modules.get("torch_stoi", "_unset")
        sys.modules["torch_stoi"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError):
                # Sanity: confirm our None-injection blocks a direct import.
                importlib.import_module("torch_stoi")
            # The real assertion: metrics imports cleanly anyway.
            m = importlib.import_module("se_probe.metrics")
            assert hasattr(m, "c50")
        finally:
            if prev == "_unset":
                sys.modules.pop("torch_stoi", None)
            else:
                sys.modules["torch_stoi"] = prev


class TestPackageImport:
    """The umbrella ``import se_probe`` must succeed on core deps only."""

    def test_se_probe_imports(self):
        sys.modules.pop("se_probe", None)
        pkg = importlib.import_module("se_probe")
        assert pkg.__version__ == "0.1.2"

    def test_public_surface_resolves(self):
        import se_probe

        # A handful of names that have to resolve without the optional GPU
        # metric extras installed.
        for name in (
            "cka",
            "diffusion_map_torch",
            "ActivationsExtractor",
            "load_clean_wavs",
            "SAMPLE_RATE",
            "TEST_NOISES",
        ):
            assert hasattr(se_probe, name), (
                f"se_probe public surface should expose {name!r}; "
                "check se_probe/__init__.py __all__"
            )
