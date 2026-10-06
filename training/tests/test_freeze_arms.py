"""Selective freezing in training/train.py, on a stand-in module.

No MUSE model is built: a plain nn.Module whose parameter names mimic the MUSE
generator's is enough to pin the key rules of ``apply_freeze_arm`` (the Job A
arms) and ``apply_unfreeze`` (the per-block / per-layer study).
"""
import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train as train_module  # noqa: E402
from train import (  # noqa: E402
    apply_freeze_arm,
    apply_unfreeze,
    validate_unfreeze,
)

STAGES = ["encoder_level1", "encoder_level2", "latent",
          "decoder_level2", "decoder_level1", "mag_refinement", "pha_refinement"]


class FakeMUSE(nn.Module):
    """Parameters named like models.generator.MUSE's state_dict keys."""

    def __init__(self):
        super().__init__()
        names = ["dense_encoder.conv.weight", "mask_decoder.conv.weight",
                 "phase_decoder.conv.weight",
                 "TCFTransformer.patch_embed_encoder_level1.w",
                 "TCFTransformer.down1_2.w", "TCFTransformer.up3_2.w",
                 "TCFTransformer.mag_output.w"]
        for st in STAGES:
            for layer in range(4):
                names.append(f"TCFTransformer.{st}.mhca_blks.0.transformer_layers.{layer}.norm1.weight")
        self._names = names
        for n in names:
            self.register_parameter(n.replace(".", "__"), nn.Parameter(torch.zeros(3)))

    def named_parameters(self, *a, **k):  # restore the dotted names
        for n in self._names:
            yield n, getattr(self, n.replace(".", "__"))

    def parameters(self, *a, **k):
        for _, p in self.named_parameters():
            yield p


def _trainable(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_full_ft_arm_freezes_nothing():
    m = FakeMUSE()
    trainable, frozen = apply_freeze_arm(m, "full_ft")
    assert frozen == 0 and trainable == 3 * len(m._names)


def test_freeze_encoder_arm_rules():
    m = FakeMUSE()
    apply_freeze_arm(m, "freeze_encoder")
    t = _trainable(m)
    assert "dense_encoder.conv.weight" not in t
    assert not any("TCFTransformer.encoder_level" in n for n in t)
    # decoder, latent, refinement, heads and the non-stage TCF parameters stay trainable
    assert "mask_decoder.conv.weight" in t and "phase_decoder.conv.weight" in t
    assert any("decoder_level2" in n for n in t) and any(".latent." in n for n in t)
    assert any("mag_refinement" in n for n in t)
    # the patch-embed of encoder_level1 does NOT contain the key substring and stays trainable
    assert "TCFTransformer.patch_embed_encoder_level1.w" in t


def test_freeze_decoder_arm_rules():
    m = FakeMUSE()
    apply_freeze_arm(m, "freeze_decoder")
    t = _trainable(m)
    assert not any("TCFTransformer.decoder_level" in n for n in t)
    assert "mask_decoder.conv.weight" not in t and "phase_decoder.conv.weight" not in t
    assert "dense_encoder.conv.weight" in t
    assert any("encoder_level1" in n for n in t)
    # documented quirk: refinement and latent are not frozen by this arm
    assert any("mag_refinement" in n for n in t) and any(".latent." in n for n in t)


def test_unknown_arm_rejected():
    with pytest.raises(ValueError):
        apply_freeze_arm(FakeMUSE(), "freeze_everything")


def test_unfreeze_block_and_boundary():
    m = FakeMUSE()
    trainable, total = apply_unfreeze(m, "encoder_l1")
    t = _trainable(m)
    assert t == {"TCFTransformer.patch_embed_encoder_level1.w", "TCFTransformer.down1_2.w"} | {
        f"TCFTransformer.encoder_level1.mhca_blks.0.transformer_layers.{i}.norm1.weight" for i in range(4)}
    assert trainable == 3 * len(t) and total == 3 * len(m._names)

    apply_unfreeze(m, "refinement", unfreeze_boundary=True)
    t = _trainable(m)
    assert "dense_encoder.conv.weight" in t and "mask_decoder.conv.weight" in t
    assert "TCFTransformer.mag_output.w" in t
    assert all(("refinement" in n) or n.startswith(("dense_encoder.", "mask_decoder.", "phase_decoder."))
               or n == "TCFTransformer.mag_output.w" for n in t)


def test_unfreeze_single_layer():
    m = FakeMUSE()
    apply_unfreeze(m, "decoder_l2.3")
    assert _trainable(m) == {
        "TCFTransformer.decoder_level2.mhca_blks.0.transformer_layers.3.norm1.weight"}

    m = FakeMUSE()
    apply_unfreeze(m, "refinement.0")
    assert _trainable(m) == {
        "TCFTransformer.mag_refinement.mhca_blks.0.transformer_layers.0.norm1.weight",
        "TCFTransformer.pha_refinement.mhca_blks.0.transformer_layers.0.norm1.weight"}


def test_unfreeze_all_restores_everything():
    m = FakeMUSE()
    apply_unfreeze(m, "latent")
    apply_unfreeze(m, "all")
    assert _trainable(m) == set(m._names)


@pytest.mark.parametrize("bad", ["encoder", "encoder_l1.4", "encoder_l1.x", "bogus.0"])
def test_validate_unfreeze_rejects(bad):
    with pytest.raises(ValueError):
        validate_unfreeze(bad)


def test_module_level_maps_are_the_documented_ones():
    assert set(train_module.FREEZE_ARM_KEYS) == {"full_ft", "freeze_encoder", "freeze_decoder"}
    assert train_module.FREEZE_ARM_KEYS["freeze_encoder"] == ("dense_encoder", "TCFTransformer.encoder_level")
    assert train_module.FREEZE_ARM_KEYS["freeze_decoder"] == (
        "TCFTransformer.decoder_level", "mask_decoder", "phase_decoder")
    assert set(train_module.BLOCK_TO_PREFIXES) == set(train_module.BLOCK_TO_LAYER_PREFIXES)
