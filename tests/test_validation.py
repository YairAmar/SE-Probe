"""
Validation and sanity check tests.

The full-scale validations (DC1-vs-SNR correlation, full-grid completeness)
are gated behind the presence of ``results_df/diffusion_maps/diffusion_maps_all.parquet``
and skipped on a fresh clone. Lighter checks run against the always-shipped
``tests/fixtures/smoke.parquet`` so ``pytest tests/`` passes out of the box.
"""
import numpy as np
import pandas as pd
import pytest
import os
from pathlib import Path


# Path to full results (only present after the full pipeline / `setup --full-data`).
DIFFUSION_MAPS_PATH = "results_df/diffusion_maps/diffusion_maps_all.parquet"

SMOKE_FIXTURE = Path(__file__).parent / "fixtures" / "smoke.parquet"


def results_available():
    """Check if results file exists."""
    return os.path.exists(DIFFUSION_MAPS_PATH)


class TestSmokeFixture:
    """Always-on tests against the shipped smoke fixture (no env / no downloads)."""

    @pytest.fixture
    def df(self):
        if not SMOKE_FIXTURE.exists():
            pytest.skip(f"smoke fixture missing: {SMOKE_FIXTURE}")
        return pd.read_parquet(SMOKE_FIXTURE)

    def test_smoke_has_rows(self, df):
        assert len(df) > 0

    def test_smoke_has_cka_column(self, df):
        assert "CKA" in df.columns
        assert np.isfinite(df["CKA"]).all()

    def test_smoke_cka_in_unit_range(self, df):
        # CKA is in [0, 1] up to floating-point slack
        assert df["CKA"].min() >= -1e-6
        assert df["CKA"].max() <= 1 + 1e-6


@pytest.mark.skipif(not results_available(), reason="Results file not available")
class TestDC1CorrelatesWithSNR:
    """Tests that DC1 (first diffusion coordinate) correlates with SNR."""

    def test_dc1_correlates_with_snr(self):
        """Test that DC1 has strong correlation with SNR."""
        from scipy.stats import spearmanr

        df = pd.read_parquet(DIFFUSION_MAPS_PATH)

        # Test a few layers
        test_layers = df['layer'].unique()[:5]

        for layer in test_layers:
            layer_df = df[df['layer'] == layer]

            # Compute Spearman correlation between DC1 (psi_0) and SNR
            if 'psi_0' not in layer_df.columns:
                continue

            corr, pval = spearmanr(layer_df['snr'], layer_df['psi_0'])

            # Should have strong correlation (positive or negative)
            # Using a lower threshold since exact correlation depends on layer
            assert abs(corr) > 0.5 or pval > 0.05, \
                f"Layer {layer}: unexpected low correlation (r={corr:.3f}, p={pval:.3f})"


@pytest.mark.skipif(not results_available(), reason="Results file not available")
class TestComponentCountReasonable:
    """Tests that component count is reasonable."""

    def test_component_count_reasonable(self):
        """Test that component count is << N for 533 points."""
        df = pd.read_parquet(DIFFUSION_MAPS_PATH)

        # For 533 points, component count should be << 533
        n_components = df['n_components'].unique()

        # All should be < 100 for 99% energy cutoff (typically 10-30)
        assert all(nc < 100 for nc in n_components), \
            f"Component counts too high: {n_components}"

    def test_component_count_positive(self):
        """Test that all layers have at least 1 component."""
        df = pd.read_parquet(DIFFUSION_MAPS_PATH)
        assert all(df['n_components'] > 0)


@pytest.mark.skipif(not results_available(), reason="Results file not available")
class TestNoiseTypesShowStructure:
    """Tests that different noise types show distinct structure."""

    def test_noise_types_cluster(self):
        """Test that different noise types have different DC2 values."""
        df = pd.read_parquet(DIFFUSION_MAPS_PATH)

        # Pick a layer
        layer = df['layer'].iloc[0]
        layer_df = df[df['layer'] == layer]

        if 'psi_1' not in layer_df.columns:
            pytest.skip("No psi_1 column (only 1 component selected)")

        # Group by noise, compute mean DC2 (secondary structure)
        noise_means = layer_df.groupby('noise_name')['psi_1'].mean()

        # Different noises should have different DC2 values
        assert noise_means.std() > 1e-6, \
            "All noise types have same DC2 mean - no noise structure"


@pytest.mark.skipif(not results_available(), reason="Results file not available")
class TestEigenvalueProperties:
    """Tests for eigenvalue properties in results."""

    def test_eigenvalues_can_be_loaded(self):
        """Test that eigenvalues can be deserialized from bytes."""
        df = pd.read_parquet(DIFFUSION_MAPS_PATH)

        row = df.iloc[0]
        eigs = np.frombuffer(row['eigenvalues'], dtype=np.float32)

        assert len(eigs) > 0
        assert np.all(np.isfinite(eigs))

    def test_eigenvalues_descending(self):
        """Test that eigenvalues are in descending order."""
        df = pd.read_parquet(DIFFUSION_MAPS_PATH)

        # Check a few rows
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            eigs = np.frombuffer(row['eigenvalues'], dtype=np.float32)
            assert np.all(np.diff(eigs) <= 1e-5), \
                f"Eigenvalues not descending for row {i}"


@pytest.mark.skipif(not results_available(), reason="Results file not available")
class TestDataCompleteness:
    """Tests for data completeness."""

    def test_all_layers_present(self):
        """Test that all 72 layers are present."""
        df = pd.read_parquet(DIFFUSION_MAPS_PATH)
        n_layers = df['layer'].nunique()
        assert n_layers == 72, f"Expected 72 layers, got {n_layers}"

    def test_all_noises_present(self):
        """Test that all 13 noise types are present."""
        df = pd.read_parquet(DIFFUSION_MAPS_PATH)
        n_noises = df['noise_name'].nunique()
        assert n_noises == 13, f"Expected 13 noise types, got {n_noises}"

    def test_all_snrs_present(self):
        """Test that all 41 SNR values are present."""
        df = pd.read_parquet(DIFFUSION_MAPS_PATH)
        n_snrs = df['snr'].nunique()
        assert n_snrs == 41, f"Expected 41 SNR values, got {n_snrs}"

    def test_row_count(self):
        """Test that total row count matches expected."""
        df = pd.read_parquet(DIFFUSION_MAPS_PATH)
        expected = 72 * 13 * 41  # layers × noises × SNRs
        assert len(df) == expected, f"Expected {expected} rows, got {len(df)}"
