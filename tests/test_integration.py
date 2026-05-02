"""
Integration tests for the diffusion maps pipeline.
"""
import os
import tempfile

import numpy as np
import pandas as pd

from se_probe.diffusion_maps import diffusion_map_torch


class TestDiffusionMapsOnMockCentroids:
    """Tests for diffusion maps on mock centroid data."""

    def test_diffusion_maps_subset(self):
        """Test diffusion maps on subset of mock centroids."""
        # Create mock centroids for 3 noises × 5 SNRs = 15 points
        mock_centroids = []
        for noise in ['A', 'B', 'C']:
            for snr in [-5, 0, 5, 10, 15]:
                mock_centroids.append({
                    'noise_name': noise,
                    'snr': float(snr),
                    'layer': 'test_layer',
                    'centroid': np.random.randn(1000).astype(np.float32).tobytes(),
                    'n_samples': 10
                })

        df_centroids = pd.DataFrame(mock_centroids)

        # Extract centroids and run diffusion maps
        X = np.vstack([np.frombuffer(c, dtype=np.float32) for c in df_centroids['centroid']])
        psi = diffusion_map_torch(X, cutoff=0.99)

        assert psi.shape[0] == 15  # 15 centroids
        assert psi.shape[1] <= 14  # At most N-1 components

    def test_diffusion_maps_full_scale(self):
        """Test diffusion maps at full scale (13 noises × 41 SNRs = 533 points)."""
        # Create mock centroids matching production scale
        noises = [f'NOISE_{i}' for i in range(13)]
        snrs = list(range(-10, 31))  # -10 to 30 dB
        feature_dim = 1000  # Smaller than real for test speed

        mock_centroids = []
        for noise in noises:
            for snr in snrs:
                mock_centroids.append({
                    'noise_name': noise,
                    'snr': float(snr),
                    'layer': 'test_layer',
                    'centroid': np.random.randn(feature_dim).astype(np.float32).tobytes(),
                    'n_samples': 100
                })

        df_centroids = pd.DataFrame(mock_centroids)
        assert len(df_centroids) == 533

        # Extract centroids and run diffusion maps
        X = np.vstack([np.frombuffer(c, dtype=np.float32) for c in df_centroids['centroid']])
        psi, eigs = diffusion_map_torch(X, cutoff=0.99, return_eigs=True)

        assert psi.shape[0] == 533
        assert psi.shape[1] < 533  # Components selected by cutoff

    def test_multiple_layers(self):
        """Test diffusion maps for multiple layers."""
        layers = ['layer1', 'layer2', 'layer3']
        noises = ['A', 'B']
        snrs = [0, 5, 10]

        mock_centroids = []
        for layer in layers:
            for noise in noises:
                for snr in snrs:
                    mock_centroids.append({
                        'noise_name': noise,
                        'snr': float(snr),
                        'layer': layer,
                        'centroid': np.random.randn(500).astype(np.float32).tobytes(),
                        'n_samples': 10
                    })

        df_centroids = pd.DataFrame(mock_centroids)

        # Process each layer separately
        results = []
        for layer in layers:
            layer_df = df_centroids[df_centroids['layer'] == layer]
            X = np.vstack([np.frombuffer(c, dtype=np.float32) for c in layer_df['centroid']])
            psi, eigs = diffusion_map_torch(X, cutoff=0.99, return_eigs=True)

            for i, (_, row) in enumerate(layer_df.iterrows()):
                result = {
                    'layer': layer,
                    'noise_name': row['noise_name'],
                    'snr': row['snr'],
                    'n_components': psi.shape[1],
                }
                for j in range(psi.shape[1]):
                    result[f'psi_{j}'] = psi[i, j]
                results.append(result)

        result_df = pd.DataFrame(results)

        assert len(result_df) == len(layers) * len(noises) * len(snrs)
        assert result_df['layer'].nunique() == 3


class TestCentroidFileLoading:
    """Tests for loading centroid files."""

    def test_load_single_file(self):
        """Test loading a single centroid parquet file."""
        centroid = np.random.randn(1000).astype(np.float32)
        df = pd.DataFrame([{
            'noise_name': 'TEST',
            'snr': 0.0,
            'layer': 'layer1',
            'centroid': centroid.tobytes(),
            'n_samples': 50
        }])

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "centroid_0dB_muse_TEST.parquet")
            df.to_parquet(path)

            df_loaded = pd.read_parquet(path)
            assert len(df_loaded) == 1
            assert df_loaded['noise_name'].iloc[0] == 'TEST'

    def test_load_multiple_files(self):
        """Test loading multiple centroid parquet files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create multiple files
            for snr in [-5, 0, 5]:
                for noise in ['A', 'B']:
                    df = pd.DataFrame([{
                        'noise_name': noise,
                        'snr': float(snr),
                        'layer': 'layer1',
                        'centroid': np.random.randn(100).astype(np.float32).tobytes(),
                        'n_samples': 10
                    }])
                    path = os.path.join(tmp_dir, f"centroid_{snr}dB_muse_{noise}.parquet")
                    df.to_parquet(path)

            # Load all files
            from glob import glob
            files = sorted(glob(os.path.join(tmp_dir, "centroid_*.parquet")))
            dfs = [pd.read_parquet(f) for f in files]
            combined_df = pd.concat(dfs, ignore_index=True)

            assert len(combined_df) == 6  # 3 SNRs × 2 noises


class TestOutputSchema:
    """Tests for diffusion maps output schema."""

    def test_output_has_required_columns(self):
        """Test that output DataFrame has required columns."""
        # Create minimal mock data
        X = np.random.randn(10, 50).astype(np.float32)
        psi, eigs = diffusion_map_torch(X, cutoff=0.99, return_eigs=True)

        # Build result row
        result = {
            'layer': 'test_layer',
            'noise_name': 'TEST',
            'snr': 0.0,
            'n_components': psi.shape[1],
            'eigenvalues': eigs.tobytes(),
        }
        for j in range(psi.shape[1]):
            result[f'psi_{j}'] = psi[0, j]

        df = pd.DataFrame([result])

        required_cols = {'layer', 'noise_name', 'snr', 'n_components', 'eigenvalues'}
        assert required_cols.issubset(set(df.columns))

        # Check psi columns exist
        psi_cols = [c for c in df.columns if c.startswith('psi_')]
        assert len(psi_cols) == psi.shape[1]
