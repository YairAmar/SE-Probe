"""Tests for ``se_probe.plotting``.

Test level: **UNIT**.
Rationale: ``apply_paper_rcparams`` mutates a shared global
(``matplotlib.rcParams``) and ``MODEL_COLORS`` / ``MODEL_LABELS`` are
plain dicts. There is no I/O, no fixture, no rendering — the function's
contract is "specific keys land with specific values". A unit test is
the highest-fidelity assertion possible. We restore rcParams in a
fixture so test order does not leak global style state into the rest of
the suite.
"""
from __future__ import annotations

import matplotlib as mpl
import pytest

from se_probe.plotting import (
    MODEL_COLORS,
    MODEL_LABELS,
    apply_paper_rcparams,
)


@pytest.fixture
def restore_rcparams():
    """Snapshot rcParams before each test and restore after, so notebooks
    or other tests that didn't call ``apply_paper_rcparams`` see the
    matplotlib defaults they expect."""
    saved = mpl.rcParams.copy()
    yield
    mpl.rcParams.update(saved)


class TestApplyPaperRcparams:
    """``apply_paper_rcparams`` writes specific keys into matplotlib.rcParams."""

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("font.family", ["serif"]),
            ("mathtext.fontset", "cm"),
            ("font.size", 10),
            ("axes.titlesize", 14),
            ("axes.labelsize", 12),
            ("xtick.labelsize", 10),
            ("ytick.labelsize", 10),
            ("legend.fontsize", 10),
            ("figure.dpi", 72),
            ("savefig.dpi", 300),
            ("pdf.fonttype", 42),
        ],
    )
    def test_rcparam_key_lands(self, key, expected, restore_rcparams):
        apply_paper_rcparams()
        actual = mpl.rcParams[key]
        # font.family is normalised to a list by matplotlib; numeric sizes
        # may come back as float on some matplotlib versions. Assert either
        # equal-as-stored or as-numeric.
        if isinstance(expected, list):
            assert actual == expected, (
                f"rcParams[{key!r}] should be {expected!r} after "
                f"apply_paper_rcparams; got {actual!r}"
            )
        elif isinstance(expected, (int, float)):
            assert float(actual) == float(expected), (
                f"rcParams[{key!r}] should be {expected!r}; got {actual!r}"
            )
        else:
            assert actual == expected, (
                f"rcParams[{key!r}] should be {expected!r}; got {actual!r}"
            )

    def test_idempotent(self, restore_rcparams):
        apply_paper_rcparams()
        snapshot = {k: mpl.rcParams[k] for k in (
            "font.family", "font.size", "savefig.dpi", "pdf.fonttype",
        )}
        apply_paper_rcparams()
        for k, v in snapshot.items():
            assert mpl.rcParams[k] == v, (
                f"apply_paper_rcparams must be idempotent; rcParams[{k!r}] "
                f"changed across two calls: {v!r} -> {mpl.rcParams[k]!r}"
            )

    def test_returns_none(self, restore_rcparams):
        # Notebooks rely on the implicit None return when the call is the
        # last line of a cell to avoid printing the rcParams object.
        assert apply_paper_rcparams() is None


class TestModelStyleMaps:
    """``MODEL_COLORS`` / ``MODEL_LABELS`` cover the three demo models."""

    def test_colors_keys(self):
        assert set(MODEL_COLORS) == {"muse", "mpsenet", "demucs"}, (
            f"MODEL_COLORS must cover the three demo models exactly; "
            f"got keys={set(MODEL_COLORS)}"
        )

    def test_labels_keys(self):
        assert set(MODEL_LABELS) == {"muse", "mpsenet", "demucs"}

    def test_color_values_are_hex(self):
        for model, color in MODEL_COLORS.items():
            assert isinstance(color, str), f"{model} colour must be a string"
            assert color.startswith("#"), (
                f"{model} colour {color!r} should be a hex string"
            )
            assert len(color) == 7, (
                f"{model} colour {color!r} should be a 7-char hex string (#RRGGBB)"
            )

    def test_labels_are_strings(self):
        for model, label in MODEL_LABELS.items():
            assert isinstance(label, str)
            assert label, f"{model} label must be non-empty"
