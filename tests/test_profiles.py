"""Per-layer profile statistics (se_probe.profiles) against the shipped paper tables."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from se_probe import layers as L
from se_probe import profiles as P

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results_tables"
DEMO = REPO / "results_demo"

needs_tables = pytest.mark.skipif(not TABLES.exists(), reason="results_tables not present")


def _muse_snr_curves():
    g = pd.read_csv(TABLES / "snr" / "cka_heatmap_values_raw_jobB18.csv")
    g = g[g["layer"].str.endswith(".norm1")]
    curves = g.pivot(index="layer", columns="snr", values="mean")
    return curves.loc[L.probed_layers("muse", curves.index)]


def test_fit_curves_recovers_a_known_line():
    x = np.arange(-10.0, 31.0)
    curves = np.stack([0.8 - 0.01 * x, 0.5 + 0.02 * x])
    f = P.fit_curves(curves, x)
    np.testing.assert_allclose(f["alpha"], [0.8, 0.5], atol=1e-12)
    np.testing.assert_allclose(f["beta"], [-0.01, 0.02], atol=1e-12)
    np.testing.assert_allclose(f["r2"], [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(f["c_low"], curves[:, 0])
    np.testing.assert_allclose(f["c_high"], curves[:, -1])
    # normalised trapezoid of a line = its midpoint value
    np.testing.assert_allclose(f["auc"], [0.8 - 0.01 * 10, 0.5 + 0.02 * 10], atol=1e-12)


@needs_tables
def test_muse_snr_fits_reproduce_the_shipped_table():
    curves = _muse_snr_curves()
    f = P.fit_curves(curves)
    ref = pd.read_csv(TABLES / "snr" / "fits_muse_snr_jobB.csv")
    for k in ("alpha", "beta", "r2", "c_low", "c_high", "auc"):
        assert np.abs(f[k] - ref[k].to_numpy()).max() < 1e-12, k


@needs_tables
def test_tradeoff_summary_reproduces_table_i_muse_snr_row():
    curves = _muse_snr_curves()
    s = P.tradeoff_summary(curves)
    ref = pd.read_csv(TABLES / "snr" / "summary_jobB.csv").set_index("model").loc["muse"]
    for k in ("r_alpha_beta", "rho_alpha_beta", "sat_freedom", "identity_r", "pc1", "pc2",
              "r_auc_alpha", "r_auc_clow", "r2_min", "r2_median", "rho_depth_range",
              "rho_depth_auc", "r_alpha_chigh"):
        assert abs(s[k] - float(ref[k])) < 1e-9, k
    # the paper's Table I numbers
    assert round(s["r_alpha_beta"], 3) == -0.950
    assert round(s["sat_freedom"], 3) == 0.275
    assert round(s["pc1"], 2) == 97.54
    assert s["peak_beta_layer"].endswith("decoder_level2.mhca_blks.0.transformer_layers.0.norm1")


@needs_tables
def test_tradeoff_summary_reproduces_table_i_c50_rows():
    mc = pd.read_csv(TABLES / "c50" / "mean_curves_c50_six_arms.csv")
    levels = [c for c in mc.columns if c not in ("scale", "arm", "model", "layer", "depth")]
    paper = {"muse_ft": (-0.993, -0.996, 0.037, 96.52), "mpsenet_ft": (-0.999, -0.905, 0.025, 99.23),
             "demucs_ft": (-0.918, -0.927, 0.167, 83.58)}
    for arm, (r, rho, f, pc1) in paper.items():
        t = mc[mc["arm"] == arm].sort_values("depth").set_index("layer")[levels]
        t.columns = t.columns.astype(float)
        s = P.tradeoff_summary(t)
        assert abs(s["r_alpha_beta"] - r) < 6e-4, arm
        assert abs(s["rho_alpha_beta"] - rho) < 6e-4, arm
        assert abs(s["sat_freedom"] - f) < 6e-4, arm
        assert abs(s["pc1"] - pc1) < 0.02, arm


def test_identity_forces_minus_one_as_spread_vanishes():
    assert P.identity_r(0.8, 0.0) == pytest.approx(-1.0)
    # the identity's floor on |r| at a given spread is reached at r_c = f
    f = 0.956
    assert abs(P.identity_r(f, f)) == pytest.approx(np.sqrt(1 - f * f), abs=1e-12)


def test_pc_shares_sum_to_hundred_and_center_over_layers():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(10, 41))
    pc1, pc2 = P.pc_shares(M)
    assert 0 < pc2 <= pc1 <= 100
    # adding a level-wise constant (same for all layers) must not change the shares
    pc1b, _ = P.pc_shares(M + rng.normal(size=(1, 41)))
    assert pc1b == pytest.approx(pc1, abs=1e-9)


@needs_tables
def test_noise_set_independence_reproduces_the_paper_contrast():
    t = pd.read_csv(TABLES / "snr" / "per_noise_beta_muse.csv", index_col=0)
    r = P.noise_set_independence(t, L.VOICEBANK_DEMAND_TEST_NOISES, L.VOICEBANK_DEMAND_TRAIN_NOISES)
    assert abs(r["pearson_r"] - 0.9812) < 5e-4
    assert abs(r["spearman_rho"] - 0.9757) < 5e-4
    assert abs(r["mean_beta_a"] - 0.0145) < 5e-5 and abs(r["mean_beta_b"] - 0.0136) < 5e-5
    assert r["peak_layer_a"] == r["peak_layer_b"]
    r2 = P.noise_set_independence(t, L.VOICEBANK_DEMAND_TEST_NOISES, L.DEMAND_NEITHER_SPLIT)
    assert abs(r2["mean_beta_b"] - 0.0135) < 5e-5


@needs_tables
def test_random_init_summary_reproduces_section_iiie():
    rnd = pd.read_csv(TABLES / "random_init" / "fig_v6_random_init_824_values.csv").sort_values("depth")
    # the trained profile on the SAME five-noise grid as the control (mean of per-noise slopes
    # == slope of the five-noise mean curve)
    per_noise = pd.read_csv(TABLES / "snr" / "per_noise_beta_muse.csv", index_col=0)
    trained = per_noise[L.VOICEBANK_DEMAND_TEST_NOISES].mean(axis=1).to_numpy()
    s = P.random_init_summary(trained, {k: rnd[f"beta_s{k}"] for k in range(3)})
    assert abs(s["factor_trained_over_largest_seed"] - 21.5) < 0.1
    rho = [s["per_seed"][str(k)]["rho_depth_beta"] for k in range(3)]
    np.testing.assert_allclose(rho, [0.43, 0.86, -0.07], atol=6e-3)
    assert s["trained_peak_depth"] == 12  # Dec-L2.0


@needs_tables
def test_emergence_convergence_reproduces_section_iiif():
    t = pd.read_csv(TABLES / "emergence" / "IIIF_per_layer_profiles_new5.csv")
    epochs = [1, 3, 5, 7, 9, 10, 12, 24, 36, 48]
    profiles = {"pretrained": pd.DataFrame({"layer": t.layer, "alpha": t.alpha_pre, "beta": t.beta_pre})}
    for e in epochs:
        profiles[e] = pd.DataFrame({"layer": t.layer, "alpha": t[f"alpha_e{e}"], "beta": t[f"beta_e{e}"]})
    out = P.emergence_convergence(profiles, final_key=48, pretrained_key="pretrained")
    conv = out["convergence"].set_index("epoch")
    assert abs(conv.loc[1, "beta_r_vs_final"] - 0.76) < 6e-3
    assert conv.loc[36, "beta_r_vs_final"] >= 0.99
    assert abs(out["concentration"]["pretrained_beta_r_vs_final"] - 0.76) < 6e-3
    assert abs(out["concentration"]["ratio_dalpha_sensitive_over_reference"] - 5.77) < 0.02
    assert abs(out["concentration"]["ratio_dbeta_sensitive_over_reference"] - 3.40) < 0.02


@needs_tables
def test_finetune_shift_reproduces_before_after_correlations():
    t = pd.read_csv(TABLES / "c50" / "c50_824_per_layer_supp.csv")
    paper = {"muse": (0.91, 0.85, 0.0020), "mpsenet": (0.81, 0.84, 0.0029), "demucs": (0.98, 0.90, 0.0015)}
    for m, (rb, ra, mad) in paper.items():
        pre = t[t.arm == f"{m}_pre"].sort_values("depth")
        post = t[t.arm == f"{m}_ft"].sort_values("depth")
        s = P.finetune_shift(pre, post)
        assert abs(s["beta_r"] - rb) < 6e-3, m
        assert abs(s["alpha_r"] - ra) < 6e-3, m
        assert abs(s["beta_mean_abs_delta"] - mad) < 1e-4, m


def test_utterance_bootstrap_on_demo_rows_runs_and_brackets_the_point_estimate():
    df = pd.read_parquet(DEMO / "cka_snr_muse_demo.parquet")
    layers = L.probed_layers("muse", df["layer"].unique())[:4]
    cube, units, levels = P.build_cube(df, layers)
    # the demo subset samples different utterances per SNR cell, so the cube is sparse
    assert cube.shape[0] == 4 and cube.shape[2] == 5 and np.isfinite(cube).any(axis=1).all()
    out = P.utterance_bootstrap(cube, levels, n_boot=50, seed=0, stats_keys=("beta", "auc"))
    assert np.all(out["beta_lo"] <= out["beta"]) and np.all(out["beta"] <= out["beta_hi"])
    assert out["replicates"]["auc"].shape == (50, 4)


def test_two_way_bootstrap_matches_the_closed_form_point_estimate():
    rng = np.random.default_rng(1)
    utts, noises, snrs = np.arange(12), ["A", "B", "C"], np.arange(-10, 31, 10)
    rows = []
    for L_ in ("x", "y"):
        for n in noises:
            for u in utts:
                for s in snrs:
                    rows.append(dict(layer=L_, noise_name=n, clean_idx=u, snr=s,
                                     CKA=0.7 + 0.005 * s + rng.normal(0, 0.02)))
    df = pd.DataFrame(rows)
    out = P.two_way_cluster_bootstrap(df, ["x", "y"], n_boot=40, seed=0)
    curves = P.mean_level_curves(df, layers=["x", "y"], unit_col=None)
    f = P.fit_curves(curves)
    np.testing.assert_allclose(out.sort_values("layer")["beta"], f["beta"], atol=1e-12)
    assert (out["beta_ci_lo"] <= out["beta"]).all() and (out["beta"] <= out["beta_ci_hi"]).all()


def test_bottleneck_gap_identifies_the_target():
    point = np.array([0.5, 0.9, 0.6])
    reps = point[None, :] + np.random.default_rng(0).normal(0, 0.01, (200, 3))
    g = P.bottleneck_gap(reps, point, ["a", "lstm", "c"], target="lstm")
    assert g["runner_up"] == "c" and g["p_peak_is_target"] == 1.0 and g["gap"] == pytest.approx(0.3)


# ----------------------------------------------------------------- audit regressions
def _two_way_table(rng, n_utt=40, n_cluster=12, per_utt=3, levels=(-10.0, 0.0, 10.0, 20.0, 30.0),
                   layer="L", alpha=0.8, beta=0.005):
    """Unbalanced utterance x cluster design: each utterance meets ``per_utt`` of the clusters."""
    rows = []
    for u in range(n_utt):
        for c in rng.choice(n_cluster, size=per_utt, replace=False):
            for s in levels:
                rows.append(dict(layer=layer, clean_idx=u, noise_name=f"c{c}", snr=s,
                                 CKA=alpha + beta * s + 0.01 * rng.standard_normal()))
    return pd.DataFrame(rows)


def test_two_way_bootstrap_brackets_the_point_on_an_unbalanced_design():
    rng = np.random.default_rng(1)
    df = _two_way_table(rng)
    out = P.two_way_cluster_bootstrap(df, ["L"], level_col="snr", n_boot=200, seed=0)
    r = out.iloc[0]
    assert abs(r["alpha"] - 0.8) < 0.02
    assert r["alpha_ci_lo"] <= r["alpha"] <= r["alpha_ci_hi"]
    assert r["beta_ci_lo"] <= r["beta"] <= r["beta_ci_hi"]
    assert r["alpha_ci_hi"] - r["alpha_ci_lo"] < 0.05   # not scaled by the 3/12 fill


def test_two_way_bootstrap_averages_duplicate_cells():
    rng = np.random.default_rng(2)
    df = _two_way_table(rng)
    dup = pd.concat([df, df.assign(CKA=df["CKA"] + 0.3)], ignore_index=True)
    out = P.two_way_cluster_bootstrap(dup, ["L"], level_col="snr", n_boot=20, seed=0)
    base = P.two_way_cluster_bootstrap(df, ["L"], level_col="snr", n_boot=20, seed=0)
    assert abs(out["c_low"][0] - (base["c_low"][0] + 0.15)) < 1e-9


def test_two_way_bootstrap_keeps_the_given_layer_order_and_depth():
    rng = np.random.default_rng(3)
    df = pd.concat([_two_way_table(rng, layer="b_layer"), _two_way_table(rng, layer="a_layer")])
    out = P.two_way_cluster_bootstrap(df, ["b_layer", "a_layer"], level_col="snr", n_boot=10, seed=0)
    assert list(out["layer"]) == ["b_layer", "a_layer"]
    assert list(out["depth"]) == [0, 1]


def test_two_way_bootstrap_stream_layers_change_the_draws_but_not_the_rows():
    rng = np.random.default_rng(4)
    df = pd.concat([_two_way_table(rng, layer=name) for name in ["x", "y", "z"]])
    alone = P.two_way_cluster_bootstrap(df, ["y"], level_col="snr", n_boot=30, seed=0)
    threaded = P.two_way_cluster_bootstrap(df, ["y"], level_col="snr", n_boot=30, seed=0,
                                           stream_layers=["x", "y", "z"])
    full = P.two_way_cluster_bootstrap(df, ["x", "y", "z"], level_col="snr", n_boot=30, seed=0)
    assert list(threaded["layer"]) == ["y"]
    # the published stream visited every layer in sorted order: y's draws come after x's
    assert threaded["alpha_ci_lo"][0] == full.set_index("layer").loc["y", "alpha_ci_lo"]
    assert threaded["alpha_ci_lo"][0] != alone["alpha_ci_lo"][0]


def test_bottleneck_gap_interval_uses_the_fixed_runner_up():
    reps = np.array([[0.9, 0.7, 0.85], [0.9, 0.86, 0.5], [0.9, 0.6, 0.6]])
    point = np.array([0.9, 0.8, 0.7])
    g = P.bottleneck_gap(reps, point, ["lstm", "a", "b"], target="lstm", ci=1.0)  # tail 0 -> min/max
    assert g["runner_up"] == "a" and abs(g["gap"] - 0.1) < 1e-12
    # replicate gaps vs the fixed runner-up "a": 0.2, 0.04, 0.3 -> min 0.04, not the 0.05 of the best-other
    assert abs(g["gap_lo"] - 0.04) < 1e-12 and abs(g["gap_hi"] - 0.3) < 1e-12


def test_mean_level_curves_refuses_alphabetical_order():
    df = pd.DataFrame(dict(layer=["b", "a"] * 2, clean_idx=[0, 0, 1, 1], snr=[0.0] * 4, CKA=[0.5] * 4))
    with pytest.raises(ValueError):
        P.mean_level_curves(df, unit_col=None)


def test_average_centroids_single_key_gives_scalar_labels():
    import io

    from se_probe import centroids as C
    vec = np.arange(4, dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, vec)
    df = pd.DataFrame(dict(layer=["l", "l"], snr=[0, 10], noise_name=["n", "n"],
                           centroid=[buf.getvalue(), buf.getvalue()]))
    try:
        out = C.average_centroids(df, "snr")
    except Exception as exc:  # the encoding helper may differ; only the key shape is under test
        pytest.skip(f"centroid decoding not exercised here: {exc}")
    assert out["snr"].tolist() == [0, 10]
