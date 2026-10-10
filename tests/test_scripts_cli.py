"""Smoke tests for the scripts/ drivers: every module imports without side effects,
exposes an argparse ``--help``, and the analysis entry points run on tiny synthetic
tables."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

DRIVERS = [
    "run_snr_grid", "run_reverb_grid", "run_random_init_grid", "run_emergence_grid",
    "run_perceptual_grid", "join_perceptual", "run_diffusion_centroids", "build_diffusion_psi",
    "run_hsic_ablation", "analyze_snr_profiles", "analyze_c50_profiles", "analyze_random_init",
    "analyze_emergence", "analyze_perceptual", "analyze_diffusion", "analyze_freeze_arms",
    "hf_upload_artifacts",
]

MUSE_LAYERS = [
    "TCFTransformer.encoder_level1.mhca_blks.0.transformer_layers.0.norm1",
    "TCFTransformer.latent.mhca_blks.0.transformer_layers.2.norm1",
    "TCFTransformer.decoder_level1.mhca_blks.0.transformer_layers.3.norm1",
]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", DRIVERS)
def test_module_imports_and_has_parser(name):
    mod = _load(name)
    assert callable(mod.build_parser)
    assert callable(mod.main)


@pytest.mark.parametrize("name", ["run_snr_grid", "analyze_snr_profiles", "analyze_diffusion"])
def test_help_runs(name):
    r = subprocess.run([sys.executable, str(SCRIPTS / f"{name}.py"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout.lower()


def _synthetic_snr_cells(noises, snrs, layers=MUSE_LAYERS, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for n in noises:
        for s in snrs:
            for i, L in enumerate(layers):
                rows.append(dict(model_name="muse", noise_name=n, snr=s, layer=L,
                                 CKA=float(np.clip(0.3 + 0.2 * i + 0.004 * (i + 1) * s + rng.normal(0, 0.01), 0, 1))))
    return pd.DataFrame(rows)


def test_analyze_snr_profiles_on_synthetic_cells():
    mod = _load("analyze_snr_profiles")
    cells = _synthetic_snr_cells(["TBUS", "SCAFE", "DKITCHEN", "PRESTO", "NFIELD", "NPARK"], [-10, 0, 10, 20, 30])
    res = mod.analyze_model(cells, "muse")
    assert list(res["fits"]["layer"]) == MUSE_LAYERS
    assert res["fits"]["beta"].is_monotonic_increasing
    assert {"r_alpha_beta", "sat_freedom", "pc1"} <= set(res["summary"])
    assert set(res["contrasts"]["contrast"]) == {"test5_vs_train8", "test5_vs_neither5", "train8_vs_neither5"}


def test_analyze_c50_profiles_on_synthetic_arm():
    mod = _load("analyze_c50_profiles")
    rng = np.random.default_rng(1)
    levels = np.arange(-5.0, 25.1, 2.5)
    rows = []
    for u in range(12):
        for r in range(2):
            for lv in levels:
                for i, L in enumerate(MUSE_LAYERS):
                    rows.append(dict(clean_idx=u, utt_id=f"p23{u % 2}_{u:03d}", rir_name=f"rir{r}", target_c50=lv,
                                     layer=L, CKA=float(np.clip(0.5 + 0.1 * i + 0.005 * lv + rng.normal(0, 0.01), 0, 1)),
                                     model_name="muse_pre"))
    res = mod.analyze_arm(pd.DataFrame(rows), "muse_pre", n_boot=20, seed=0)
    assert len(res["per_layer"]) == len(MUSE_LAYERS)
    assert {"A", "A_lo", "A_hi", "alpha", "beta", "r2"} <= set(res["per_layer"].columns)
    assert "per_speaker" in res and set(res["per_speaker"]["speaker"]) == {"p230", "p231"}


def test_analyze_random_init_and_emergence_on_synthetic():
    ri = _load("analyze_random_init")
    layers = MUSE_LAYERS
    trained = _synthetic_snr_cells(["TBUS"], [-10, 0, 10, 20, 30], layers)
    rand = pd.concat([_synthetic_snr_cells(["TBUS"], [-10, 0, 10, 20, 30], layers, seed=s).assign(seed=s) for s in (0, 1)])
    rand["CKA"] = 0.99 + 0.001 * rand["snr"] / 30
    res = ri.analyze(rand, ri.curves_from_cells(trained, layers), layers)
    assert res["summary"]["factor_trained_over_largest_seed"] > 1
    assert set(res["per_layer"].columns) >= {"beta_s0", "beta_s1", "beta_mean", "cka_min"}
    em = _load("analyze_emergence")
    profiles = {"pretrained": em.profile_from_cells(trained, layers)}
    for e in (1, 5):
        profiles[e] = em.profile_from_cells(_synthetic_snr_cells(["TBUS"], [-10, 0, 10, 20, 30], layers, seed=e), layers)
    out = em.analyze(profiles, 5)
    assert len(out["convergence"]) == 3 and "concentration" in out


def test_analyze_diffusion_on_shipped_tables():
    mod = _load("analyze_diffusion")
    pl = REPO / "results_tables/diffusion/diffusion_maps_per_layer_t0.5_824_new5.parquet"
    if not pl.exists():
        pytest.skip("results_tables not present")
    pb = mod.per_block_table(pd.read_parquet(pl))
    from se_probe.diffusion_analysis import arc_length_ratio
    assert abs(arc_length_ratio(pb)["ratio"] - 3.03) < 0.01
    assert (pb["abs_rho"] > 0.995).all()


def test_analyze_perceptual_on_demo_table():
    mod = _load("analyze_perceptual")
    demo = REPO / "results_demo/cka_snr_muse_demo.parquet"
    if not demo.exists():
        pytest.skip("results_demo not present")
    res = mod.analyze_sweep(pd.read_parquet(demo), "780", n_boot=50, seed=0)
    assert {"PESQ", "STOI", "SI-SDR"} <= set(res["per_layer"]["metric"])
    assert set(res["speaker"]["n_spk"]) == {2}


def test_reverb_pool_gate_rejects_wrong_pool():
    mod = _load("run_reverb_grid")
    df = pd.DataFrame(dict(clean_idx=[0, 0], rir_name=["air_binaural_office_0", "air_binaural_booth_1"]))
    with pytest.raises(AssertionError):
        mod.verify_pool(df, n_utts=1)


def test_hf_upload_is_dry_run_by_default(tmp_path, capsys):
    mod = _load("hf_upload_artifacts")
    agg = tmp_path / "agg"
    agg.mkdir()
    (agg / "a.csv").write_text("x\n")
    assert mod.main(["--aggregated-dir", str(agg)]) == 0
    assert "dry run" in capsys.readouterr().out
