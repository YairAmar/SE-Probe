"""Checkpoint plumbing of the model adapters (no weights downloaded)."""
import os
from pathlib import Path

import pytest
import torch

from se_probe.mpsenet.model import remap_trainer_keys

REPO = Path(__file__).resolve().parents[1]


def test_trainer_keys_are_remapped_to_the_pretrained_names():
    sd = {"encoder.a": 1, "enhancer.0.b": 2, "decoder.mask_decoder.c": 3, "decoder.phase_decoder.d": 4}
    ref = ["dense_encoder.a", "TSTransformer.0.b", "mask_decoder.c", "phase_decoder.d"]
    assert remap_trainer_keys(sd, ref) == dict(zip(ref, [1, 2, 3, 4]))


def test_pretrained_keys_pass_through_untouched():
    sd = {"dense_encoder.a": 1, "TSTransformer.0.b": 2}
    assert remap_trainer_keys(sd, list(sd)) is sd


def test_random_init_muse_is_seed_reproducible_and_untrained():
    cfg = REPO / "training" / "paper_result" / "config.json"
    if not cfg.exists():
        pytest.skip("MUSE config not in tree")
    os.environ["MUSE_CONFIG"] = str(cfg)
    from se_probe.muse.model import load_random_init_muse_model

    a = load_random_init_muse_model(0, device="cpu")
    b = load_random_init_muse_model(0, device="cpu")
    c = load_random_init_muse_model(1, device="cpu")
    pa, pb, pc = (next(iter(m.parameters())) for m in (a, b, c))
    assert torch.equal(pa, pb) and not torch.equal(pa, pc)
    assert not a.training
