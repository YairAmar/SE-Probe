"""Quality-association estimators (se_probe.perceptual)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from se_probe import perceptual as Q

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results_tables"
DEMO = REPO / "results_demo"


def test_t_ci_reproduces_the_published_speaker_level_interval():
    r = [-0.3195455220247532, -0.2152412701106129, 0.0217480512006277, -0.3094216153150330]
    mean, sd, lo, hi, n = Q.t_ci(r)
    assert n == 4
    assert lo == pytest.approx(-0.458118, abs=1e-6)
    assert hi == pytest.approx(0.046888, abs=1e-6)
    assert (hi - lo) == pytest.approx(0.505, abs=5e-4)


@pytest.mark.skipif(not TABLES.exists(), reason="results_tables not present")
def test_pooled_four_speaker_summary_matches_the_shipped_table():
    ps = pd.read_csv(TABLES / "perceptual" / "spk_v2__per_speaker_all.csv")
    ps = ps[ps.model == "muse"]
    mine = Q.speaker_summary_from_published(ps).set_index(["metric", "layer_idx"])
    ref = pd.read_csv(TABLES / "perceptual" / "spk_v2__speaker_level_all.csv")
    ref = ref[(ref.scope == "pooled4") & (ref.model == "muse")].set_index(["metric", "layer_idx"])
    for col in ("mean_r", "spk_lo", "spk_hi"):
        assert np.abs(mine.loc[ref.index, col].to_numpy() - ref[col].to_numpy()).max() < 1e-9, col
    assert bool(ref.loc[("SI-SDR", 23), "excludes_zero"])  # Dec-L1.3 SI-SDR
    assert not bool(ref.loc[("PESQ", 15), "excludes_zero"])  # Dec-L2.3 PESQ spans zero


def test_cluster_bootstrap_ci_brackets_pearson_r():
    rng = np.random.default_rng(3)
    n_utt, k = 60, 5
    utt = np.repeat(np.arange(n_utt), k)
    x = rng.normal(size=n_utt * k)
    y = 0.5 * x + rng.normal(size=n_utt * k)
    lo, hi = Q.cluster_bootstrap_pearson_ci(x, y, utt, n_boot=400, seed=0)
    r = np.corrcoef(x, y)[0, 1]
    assert lo < r < hi and hi - lo < 0.6


def test_per_layer_correlations_and_speaker_analysis_on_demo_rows():
    df = pd.read_parquet(DEMO / "cka_snr_muse_demo.parquet")
    layers = sorted(L for L in df.layer.unique() if L.endswith(".norm1"))[:3]
    t = Q.per_layer_correlations(df, layers, metrics=["PESQ", "STOI"], model="muse",
                                 unit_col="clean_idx", n_boot=100)
    assert set(t.metric) == {"PESQ", "STOI"} and len(t) == 6
    assert (t.utt_lo <= t.pearson).all() and (t.pearson <= t.utt_hi).all()
    out = Q.speaker_level_analysis(df, "muse", sweep="780", metrics=["PESQ"], layers=layers, n_boot=50)
    assert set(out["per_speaker"].speaker) == {"p226", "p287"}
    assert len(out["speaker"]) == 3 and (out["speaker"].n_spk == 2).all()


def test_sisdr_sign_convention_detects_a_negated_table():
    df = pd.read_parquet(DEMO / "cka_snr_muse_demo.parquet")
    assert not Q.sisdr_sign_convention(df)["needs_negation"]
    neg = df.assign(noisy_sisdr=-df["noisy_sisdr"])
    assert Q.sisdr_sign_convention(neg)["needs_negation"]


def test_ceiling_free_gain_is_a_fraction_of_headroom():
    g = Q.ceiling_free_gain([3.0, 4.5], [2.0, 4.0])
    np.testing.assert_allclose(g, [0.4, 1.0])
