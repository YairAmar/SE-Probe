"""
Unit tests for diffusion_map_torch() function.
"""
import numpy as np
import pytest
import torch

from se_probe.diffusion_maps import diffusion_map_torch


class TestDiffusionMapBasic:
    """Basic functionality tests."""

    def test_output_shape(self):
        """Test that output has correct shape (N, L) where L <= N-1."""
        X = np.random.randn(100, 50).astype(np.float32)
        psi = diffusion_map_torch(X)
        assert psi.shape[0] == 100  # N points
        assert psi.shape[1] <= 99  # At most N-1 components

    def test_return_eigs(self):
        """Test that return_eigs returns eigenvalues."""
        X = np.random.randn(50, 20).astype(np.float32)
        psi, eigs = diffusion_map_torch(X, return_eigs=True)
        assert isinstance(psi, np.ndarray)
        assert isinstance(eigs, np.ndarray)
        assert len(eigs) == psi.shape[0] - 1  # N-1 non-trivial eigenvalues

    def test_deterministic(self):
        """Test that same input produces same output."""
        X = np.random.randn(50, 20).astype(np.float32)
        psi1 = diffusion_map_torch(X)
        psi2 = diffusion_map_torch(X)
        np.testing.assert_allclose(psi1, psi2)


class TestDiffusionMapEdgeCases:
    """Edge case tests."""

    def test_small_matrix(self):
        """Test with small matrix (10 points, 5 features)."""
        X = np.random.randn(10, 5).astype(np.float32)
        psi = diffusion_map_torch(X)
        assert psi.shape[0] == 10
        assert psi.shape[1] <= 9  # At most N-1 components

    def test_identical_points(self):
        """Test with all identical points (degenerate kernel)."""
        X = np.ones((20, 10), dtype=np.float32)
        # Should handle gracefully due to eps clamping
        psi = diffusion_map_torch(X)
        assert psi.shape[0] == 20

    def test_single_feature(self):
        """Test with single feature dimension."""
        X = np.random.randn(30, 1).astype(np.float32)
        psi = diffusion_map_torch(X)
        assert psi.shape[0] == 30


class TestDiffusionMapEigenvalueProperties:
    """Tests for eigenvalue properties."""

    def test_eigenvalues_descending(self):
        """Test that eigenvalues are in descending order."""
        X = np.random.randn(100, 50).astype(np.float32)
        _, eigs = diffusion_map_torch(X, return_eigs=True)
        assert np.all(np.diff(eigs) <= 1e-6)  # Allow small numerical error

    def test_eigenvalues_bounded(self):
        """Test that eigenvalues are in [0, 1] (stochastic matrix property)."""
        X = np.random.randn(100, 50).astype(np.float32)
        _, eigs = diffusion_map_torch(X, return_eigs=True)
        assert np.all(eigs >= -1e-6)  # Non-negative (allow small numerical error)
        assert np.all(eigs <= 1 + 1e-6)  # Bounded by 1


class TestDiffusionMapParameters:
    """Tests for parameter variations."""

    def test_cutoff_affects_components(self):
        """Test that lower cutoff produces fewer components."""
        X = np.random.randn(100, 50).astype(np.float32)
        psi_99 = diffusion_map_torch(X, cutoff=0.99)
        psi_90 = diffusion_map_torch(X, cutoff=0.90)
        psi_50 = diffusion_map_torch(X, cutoff=0.50)
        assert psi_50.shape[1] <= psi_90.shape[1] <= psi_99.shape[1]

    def test_diffusion_time_scaling(self):
        """Test that diffusion time affects output."""
        X = np.random.randn(50, 20).astype(np.float32)
        psi_t1 = diffusion_map_torch(X, diffusion_time=1)
        psi_t10 = diffusion_map_torch(X, diffusion_time=10)
        # Different diffusion times should produce different embeddings
        assert not np.allclose(psi_t1, psi_t10)

    def test_alpha_normalization(self):
        """Test that alpha parameter runs without error."""
        X = np.random.randn(50, 20).astype(np.float32)
        psi_a0 = diffusion_map_torch(X, alpha=0.0)
        psi_a1 = diffusion_map_torch(X, alpha=1.0)
        # Both should produce valid output shapes
        assert psi_a0.shape[0] == 50
        assert psi_a1.shape[0] == 50


class TestDiffusionMapDevice:
    """Tests for device handling."""

    def test_cpu_execution(self):
        """Test execution on CPU."""
        X = np.random.randn(50, 20).astype(np.float32)
        psi = diffusion_map_torch(X, device='cpu')
        assert isinstance(psi, np.ndarray)
        assert psi.shape[0] == 50

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self):
        """Test execution on CUDA."""
        X = np.random.randn(50, 20).astype(np.float32)
        psi = diffusion_map_torch(X, device='cuda')
        assert isinstance(psi, np.ndarray)
        assert psi.shape[0] == 50

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_gpu_match(self):
        """Test that CPU and GPU produce same results."""
        X = np.random.randn(100, 50).astype(np.float32)
        psi_cpu = diffusion_map_torch(X, device='cpu')
        psi_gpu = diffusion_map_torch(X, device='cuda')
        np.testing.assert_allclose(psi_cpu, psi_gpu, rtol=1e-4, atol=1e-5)


class TestDiffusionMapReturns:
    """Tests for return options."""

    def test_return_complement(self):
        """Test return_complement option."""
        X = np.random.randn(50, 20).astype(np.float32)
        psi, psi_rest = diffusion_map_torch(X, return_complement=True)
        assert isinstance(psi, np.ndarray)
        assert isinstance(psi_rest, np.ndarray)
        # Sum of selected and complement should be N-1
        assert psi.shape[1] + psi_rest.shape[1] == 49

    def test_return_all(self):
        """Test returning all optional outputs."""
        X = np.random.randn(50, 20).astype(np.float32)
        psi, psi_rest, eigs, c_val = diffusion_map_torch(
            X, return_complement=True, return_eigs=True, return_cval=True
        )
        assert isinstance(psi, np.ndarray)
        assert isinstance(psi_rest, np.ndarray)
        assert isinstance(eigs, np.ndarray)
        assert isinstance(c_val, float)


class TestDiffusionMapLarge:
    """Tests for larger matrices (chunked distance computation)."""

    @pytest.mark.slow
    def test_large_matrix(self):
        """Test with matrix larger than chunk threshold."""
        X = np.random.randn(1500, 100).astype(np.float32)
        psi = diffusion_map_torch(X, device='cpu')
        assert psi.shape[0] == 1500
        assert psi.shape[1] <= 1499


class TestDiffusionMapEigSolver:
    """Tests for eigenvalue solver options."""

    def test_full_solver(self):
        """Test full eigenvalue solver."""
        X = np.random.randn(50, 20).astype(np.float32)
        psi = diffusion_map_torch(X, eig_solver='full')
        assert psi.shape[0] == 50

    def test_lobpcg_solver(self):
        """Test LOBPCG solver."""
        X = np.random.randn(50, 20).astype(np.float32)
        psi = diffusion_map_torch(X, eig_solver='lobpcg', k=10)
        assert psi.shape[0] == 50
        assert psi.shape[1] <= 10

    def test_invalid_solver_raises(self):
        """Test that invalid solver raises ValueError."""
        X = np.random.randn(50, 20).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown eig_solver"):
            diffusion_map_torch(X, eig_solver='invalid')
