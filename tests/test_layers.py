"""Probed-layer sets, depth order and labels (se_probe.layers)."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from se_probe import layers as L

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results_tables"


def _load_consts(rel):
    spec = importlib.util.spec_from_file_location(rel, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_muse_probed_layers_are_the_24_norm1_hooks_in_depth_order():
    mc = _load_consts("se_probe/muse/consts.py")
    got = L.probed_layers("muse", mc.LAYERS)
    assert got == mc.NORM1_LAYERS
    assert len(got) == L.N_PROBED_LAYERS["muse"]
    assert L.short_labels("muse", got)[:5] == ["Enc-L1.0", "Enc-L1.1", "Enc-L1.2", "Enc-L1.3", "Enc-L2.0"]
    assert L.short_label("muse", got[-1]) == "Refine.3"


def test_mpsenet_probed_layers_time_before_freq():
    pc = _load_consts("se_probe/mpsenet/consts.py")
    got = L.probed_layers("mpsenet", pc.LAYERS)
    assert len(got) == 8
    assert L.short_labels("mpsenet", got) == L.MPSENET_TICKS


def test_demucs_probed_layers_follow_the_block_list_not_the_alphabet():
    dc = _load_consts("se_probe/demucs/consts.py")
    got = L.probed_layers("demucs", dc.LAYERS)
    assert got == L.DEMUCS_BLOCKS
    # alphabetical order would put the decoder first
    assert sorted(got) != got


def test_depth_order_is_insensitive_to_input_order():
    mc = _load_consts("se_probe/muse/consts.py")
    shuffled = list(reversed(mc.NORM1_LAYERS))
    assert L.depth_order("muse", shuffled) == mc.NORM1_LAYERS


@pytest.mark.parametrize("model", ["muse", "mpsenet", "demucs"])
def test_shipped_fits_tables_are_in_depth_order(model):
    p = TABLES / "snr" / f"fits_{model}_snr_jobB.csv"
    if not p.exists():
        pytest.skip("results_tables not present")
    t = pd.read_csv(p)
    assert list(t["layer"]) == L.depth_order(model, t["layer"])
    assert list(t["depth"]) == list(range(len(t)))


def test_noise_split_partition_covers_all_eighteen():
    all_ = set(L.VOICEBANK_DEMAND_TEST_NOISES) | set(L.VOICEBANK_DEMAND_TRAIN_NOISES) | set(L.DEMAND_NEITHER_SPLIT)
    assert all_ == set(L.DEMAND_NOISES_18)
    assert len(L.VOICEBANK_DEMAND_TEST_NOISES) == 5 and len(L.VOICEBANK_DEMAND_TRAIN_NOISES) == 8


def test_layouts_have_consistent_sizes():
    for m in L.MODELS:
        lay = L.layout(m)
        assert lay["n"] == L.N_PROBED_LAYERS[m]
        assert len(lay["ticks"]) == len(lay["ticklabels"])
    assert L.layout("muse")["skip_in"] == [12, 16, 20]


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        L.probed_layers("gtcrn", ["a"])
