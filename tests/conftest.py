"""Pytest collection hook for SE-Probe tests.

This module loads BEFORE any test file is imported, so it is the only
hook that can set ``PYTORCH_ENABLE_MPS_FALLBACK=1`` early enough to
take effect in pytest collection on Apple Silicon.

Why this is needed (background):
- Several pre-existing test modules (``test_diffusion_maps.py``,
  ``test_integration.py``) do ``import torch`` at the top, before
  ``from se_probe...``. Pytest imports the module to collect tests,
  which triggers ``import torch`` first.
- PyTorch caches the fallback flag at first MPS dispatch / interpreter
  init. Setting it after that point has no effect.
- ``se_probe/__init__.py`` already calls ``setdefault`` for end users
  (whose notebooks import ``se_probe`` first), but that's too late
  here.

In production code paths (notebooks, ``scripts/setup.py``), the env
var is set early enough through ``se_probe/__init__.py`` because the
notebook bootstrap imports ``se_probe.device`` *before* any torch
import. This conftest is purely a pytest-collection-order safety net.
``setdefault`` preserves an explicit user override (e.g. ``export
PYTORCH_ENABLE_MPS_FALLBACK=0`` to assert hard failures during MPS
debugging).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
