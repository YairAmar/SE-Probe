"""Unbiased HSIC and sample-convention variants of linear CKA."""
import importlib

import numpy as np
import pytest
import torch

C = importlib.import_module("se_probe.cka")


def test_biased_variant_equals_the_batched_cka():
    a = torch.randn(16, 40, 65)
    b = a + 0.1 * torch.randn_like(a)
    assert C.linear_cka_biased(a.mean(1), b.mean(1)) == pytest.approx(C.cka(a.mean(1)[None], b.mean(1)[None])[0], abs=1e-6)


def test_unbiased_cka_is_one_for_identical_inputs_and_near_biased_for_large_n():
    x = torch.randn(400, 32)
    assert C.linear_cka_unbiased(x, x) == pytest.approx(1.0, abs=1e-6)
    y = x @ torch.randn(32, 32) + 0.3 * torch.randn(400, 32)
    assert abs(C.linear_cka_unbiased(x, y) - C.linear_cka_biased(x, y)) < 0.02


def test_unbiased_cka_reduces_the_small_sample_bias():
    # two independent random matrices: the biased estimator is positive, the unbiased one near zero
    torch.manual_seed(0)
    vals_b, vals_u = [], []
    for _ in range(50):
        x, y = torch.randn(12, 64), torch.randn(12, 64)
        vals_b.append(C.linear_cka_biased(x, y))
        vals_u.append(C.linear_cka_unbiased(x, y))
    assert np.mean(vals_b) > 0.2 and np.mean(vals_u) < np.mean(vals_b) / 2


def test_hsic_unbiased_rejects_tiny_n():
    with pytest.raises(ValueError):
        C.hsic_unbiased(torch.eye(3), torch.eye(3))


def test_representations_have_the_documented_shapes():
    act = torch.randn(16, 20, 64)  # (C, T, F)
    r = C.representations_from_activation(act)
    assert r["pooled"].shape == (16, 64)
    assert r["frames_TFC"].shape == (20, 16 * 64)
    assert r["frames_TC"].shape == (20, 16)
    v = C.cka_variants(act, act + 0.05 * torch.randn_like(act))
    assert set(v) == set(C.CKA_VARIANTS) and all(0.0 <= x <= 1.0 for x in v.values())
