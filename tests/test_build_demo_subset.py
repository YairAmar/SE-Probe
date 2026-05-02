"""Tests for ``scripts/build_demo_subset.py``.

Test level: **INTEGRATION**.
Rationale: the script reads four parquets, filters them, and writes
five output parquets, then aborts non-zero on a budget overshoot. The
file-shuffling logic is the entire contract — mocking pandas would
defeat the purpose. We synthesise a tiny ``results_df`` tree on disk
in ``tmp_path`` with the exact column shapes the script expects,
invoke the script as a subprocess against the real interpreter (so we
exercise the actual CLI parser, the same code path users hit), and
assert on the produced files.

Schemas were derived from the script source:
- SNR parquets need: ``snr``, ``noise_name``, ``clean_idx`` and any
  payload columns.
- Reverb parquet needs: ``target_c50``, optionally ``rir_name``,
  ``clean_idx`` and payload.
- Diffusion-maps per-layer parquet needs: ``layer``, ``snr``,
  ``noise_name`` (and the script subsets to MUSE NORM1 layers).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_demo_subset.py"


def _make_snr_parquet(path: Path, model: str) -> None:
    """Write a tiny SNR parquet with the columns build_demo_subset expects."""
    rows = []
    snrs = [-10, -5, 0, 5, 10, 15, 20, 25, 30]  # superset of DEFAULT_SNRS
    noises = ["TBUS", "DKITCHEN", "OOFFICE"]   # superset including TBUS default
    for snr in snrs:
        for noise in noises:
            for clean_idx in range(20):  # > 10 utts so subsampling actually runs
                rows.append({
                    "snr": snr,
                    "noise_name": noise,
                    "clean_idx": clean_idx,
                    "layer": "encoder.norm1",
                    "CKA": float(
                        np.random.default_rng(abs(snr * 17 + clean_idx)).random()
                    ),
                })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _make_reverb_parquet(path: Path) -> None:
    rows = []
    c50s = list(np.arange(-5, 27.5, 2.5))     # 13 levels in real data
    rirs = ["air_binaural_office_0_0_1.mat", "air_binaural_meeting_0_0_1.mat"]
    for c50 in c50s:
        for rir in rirs:
            for clean_idx in range(20):
                rows.append({
                    "target_c50": float(c50),
                    "rir_name": rir,
                    "clean_idx": clean_idx,
                    "layer": "encoder.norm1",
                    "CKA": float(
                        np.random.default_rng(abs(int(c50 * 100)) + clean_idx).random()
                    ),
                })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _make_diffusion_parquet(path: Path) -> None:
    """The script filters by MUSE NORM1_LAYERS — load real layer names so
    the subset is non-empty."""
    import importlib.util
    consts_path = REPO_ROOT / "se_probe" / "muse" / "consts.py"
    spec = importlib.util.spec_from_file_location("_muse_c", consts_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    norm1_layers = list(mod.NORM1_LAYERS)

    rows = []
    snrs = [-10, 0, 10, 20, 30]
    noises = ["TBUS", "OOFFICE"]
    for layer in norm1_layers:
        for snr in snrs:
            for noise in noises:
                rows.append({
                    "layer": layer,
                    "snr": snr,
                    "noise_name": noise,
                    "psi_0": float(
                        np.random.default_rng(abs(hash((layer, snr))) & 0xFFFF).random()
                    ),
                    "psi_1": 0.5,
                    "n_components": 5,
                })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


@pytest.fixture
def synth_source_dir(tmp_path: Path) -> Path:
    """Build a complete synthetic ``results_df``-shaped directory."""
    src = tmp_path / "results_df"
    _make_snr_parquet(src / "snr" / "cka_snr_muse.parquet", "muse")
    _make_snr_parquet(src / "snr" / "cka_snr_mpsenet.parquet", "mpsenet")
    _make_snr_parquet(src / "snr" / "cka_snr_demucs.parquet", "demucs")
    _make_reverb_parquet(src / "reverb" / "cka_reverb_muse_epoch_48.parquet")
    _make_diffusion_parquet(
        src / "diffusion_maps" / "diffusion_maps_per_layer_t0.5.parquet"
    )
    return src


def _run(source_dir: Path, target_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--source-dir", str(source_dir),
            "--target-dir", str(target_dir),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestBuildDemoSubsetGoldenPath:
    """Happy path: synthetic results_df -> demo parquets, exit 0."""

    def test_script_exists(self):
        assert SCRIPT.exists(), f"build_demo_subset.py must exist at {SCRIPT}"

    def test_run_against_synth_source_succeeds(self, synth_source_dir, tmp_path):
        target = tmp_path / "results_demo"
        proc = _run(synth_source_dir, target)
        assert proc.returncode == 0, (
            f"build_demo_subset.py against a complete synthetic source should "
            f"exit 0; got {proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )

    def test_writes_five_demo_parquets(self, synth_source_dir, tmp_path):
        target = tmp_path / "results_demo"
        proc = _run(synth_source_dir, target)
        assert proc.returncode == 0, proc.stderr
        expected = {
            "cka_snr_muse_demo.parquet",
            "cka_snr_mpsenet_demo.parquet",
            "cka_snr_demucs_demo.parquet",
            "cka_reverb_muse_demo.parquet",
            "diffusion_maps_per_layer_demo.parquet",
        }
        actual = {p.name for p in target.iterdir() if p.suffix == ".parquet"}
        missing = expected - actual
        assert not missing, (
            f"build_demo_subset.py should write all 5 demo parquets; "
            f"missing: {missing}; got: {actual}"
        )

    def test_snr_parquet_filtered_to_requested_snrs_and_noise(
        self, synth_source_dir, tmp_path
    ):
        target = tmp_path / "results_demo"
        _run(synth_source_dir, target)
        df = pd.read_parquet(target / "cka_snr_muse_demo.parquet")
        assert set(df["snr"].unique()) == {-10, 0, 10, 20, 30}, (
            f"SNR demo subset should keep exactly the 5 default SNRs; "
            f"got {set(df['snr'].unique())}"
        )
        assert set(df["noise_name"].unique()) == {"TBUS"}, (
            f"SNR demo subset should keep only the default noise (TBUS); "
            f"got {set(df['noise_name'].unique())}"
        )

    def test_snr_parquet_subsamples_clean_idx(self, synth_source_dir, tmp_path):
        target = tmp_path / "results_demo"
        _run(synth_source_dir, target)
        df = pd.read_parquet(target / "cka_snr_muse_demo.parquet")
        # Per-(snr, noise) cell should have at most utts-per-cell unique clean_idx
        for (snr, noise), grp in df.groupby(["snr", "noise_name"]):
            n = grp["clean_idx"].nunique()
            assert n <= 10, (
                f"per-cell utterance count must respect --utts-per-cell=10; "
                f"snr={snr} noise={noise} got {n}"
            )

    def test_reverb_parquet_subsets_c50(self, synth_source_dir, tmp_path):
        target = tmp_path / "results_demo"
        _run(synth_source_dir, target)
        df = pd.read_parquet(target / "cka_reverb_muse_demo.parquet")
        # Script keeps at most 5 c50 levels and a single RIR
        assert df["target_c50"].nunique() <= 5, (
            f"reverb demo should keep <=5 c50 levels; "
            f"got {df['target_c50'].nunique()}"
        )
        if "rir_name" in df.columns:
            assert df["rir_name"].nunique() == 1, (
                f"reverb demo should pin a single RIR for controlled "
                f"comparison; got {df['rir_name'].nunique()}"
            )

    def test_diffusion_parquet_filtered_to_norm1_layers(
        self, synth_source_dir, tmp_path
    ):
        target = tmp_path / "results_demo"
        _run(synth_source_dir, target)
        df = pd.read_parquet(target / "diffusion_maps_per_layer_demo.parquet")
        # Every kept layer must end with `.norm1` per build_demo_subset's
        # _filter_diffusion -> NORM1_LAYERS contract.
        assert (df["layer"].str.endswith(".norm1")).all(), (
            f"diffusion demo subset should keep only MUSE NORM1 layers; "
            f"saw layers: {df['layer'].unique()[:5]}"
        )

    def test_total_size_under_budget(self, synth_source_dir, tmp_path):
        target = tmp_path / "results_demo"
        proc = _run(synth_source_dir, target)
        assert proc.returncode == 0
        total_bytes = sum(
            p.stat().st_size for p in target.iterdir() if p.suffix == ".parquet"
        )
        assert total_bytes < 50 * 1024 * 1024, (
            f"demo parquets should fit in 50 MB budget; got "
            f"{total_bytes / 1024 / 1024:.2f} MB"
        )


class TestBuildDemoSubsetErrors:
    """Failure modes: friendly errors when inputs are missing."""

    def test_missing_source_dir_exits_nonzero(self, tmp_path):
        proc = _run(tmp_path / "nonexistent", tmp_path / "out")
        assert proc.returncode != 0, (
            "build_demo_subset.py must abort with non-zero when --source-dir "
            "does not exist"
        )
        # Combined message: "ERROR: source dir not found: <path>"
        combined = proc.stdout + proc.stderr
        assert "source dir not found" in combined, (
            f"missing-source error should be human-readable, got "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )

    def test_missing_input_parquet_exits_nonzero(self, tmp_path):
        # Source dir exists but is empty.
        src = tmp_path / "results_df"
        src.mkdir()
        proc = _run(src, tmp_path / "out")
        assert proc.returncode != 0
        combined = proc.stdout + proc.stderr
        assert "missing input parquet" in combined or "missing" in combined.lower(), (
            f"missing-parquet error should be human-readable, got "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )

    def test_budget_overshoot_aborts(self, synth_source_dir, tmp_path):
        # Setting --budget-mb to 0 should always overshoot.
        target = tmp_path / "results_demo"
        proc = _run(synth_source_dir, target, "--budget-mb", "0.0")
        assert proc.returncode != 0, (
            "--budget-mb=0 must trigger the budget guard and abort non-zero "
            "so accidental size growth never lands in main"
        )
        combined = proc.stdout + proc.stderr
        assert "budget" in combined.lower(), (
            f"budget-overshoot error should mention 'budget'; "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
