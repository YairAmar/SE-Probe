"""824-scale diffusion-map helpers against the shipped per-block tables."""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

D = importlib.import_module("se_probe.diffusion_analysis")
REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results_tables" / "diffusion"


def test_arc_length_and_reference_distance():
    pts = np.array([[0, 0, 9], [3, 4, 9], [6, 8, 9]], float)
    assert D.arc_length(pts) == pytest.approx(10.0)
    np.testing.assert_allclose(D.distances_from_reference(pts[:, :2], 0), [0, 5, 10])


def test_group_distances_normalises_per_panel():
    Dm = np.array([[0, 1, 4, 4], [1, 0, 4, 4], [4, 4, 0, 1], [4, 4, 1, 0]], float)
    g = D.group_distances(Dm, [0, 1], [2, 3])
    assert g["between"] == pytest.approx(1.0) and g["within_a"] == pytest.approx(0.25)


@pytest.mark.skipif(not TABLES.exists(), reason="results_tables not present")
def test_per_block_statistics_reproduce_the_shipped_arc_lengths_and_ratio():
    psi = pd.read_parquet(TABLES / "diffusion_maps_per_layer_t0.5_824_new5.parquet")
    rows = []
    for layer, name in zip(D.REPRESENTATIVE_LAYERS, D.BLOCK_NAMES):
        snrs, M = D.block_trajectory_from_psi(psi, layer)
        st = D.block_statistics(snrs, M)
        st["block"] = name
        rows.append(st)
    pb = pd.DataFrame(rows).set_index("block")
    ref = pd.read_csv(TABLES / "IIIG2_per_block_new5.csv").set_index("block")
    np.testing.assert_allclose(pb.loc[ref.index, "arc_len_2d"], ref["arc_len_2d"], rtol=1e-6)
    assert (pb["abs_rho"] > 0.995).all()
    ratio = D.arc_length_ratio(pb.reset_index())
    assert abs(ratio["ratio"] - 3.03) < 5e-3
    assert abs(ratio["decoder_mean"] - 0.193) < 5e-4 and abs(ratio["encoder_mean"] - 0.064) < 5e-4
