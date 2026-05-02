"""
Diffusion maps implementation using PyTorch.

Ported from the old repo's activation_research_utils.py.
"""
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch

from se_probe.device import get_device

__all__ = ["diffusion_map_torch"]


def diffusion_map_torch(
    X_np: np.ndarray,
    *,
    cutoff: float = 0.99,
    tol: float = 1e-3,
    diffusion_time: int = 10,
    alpha: float = 0.0,
    eig_solver: str = "full",
    k: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    return_eigs: bool = False,
    return_complement: bool = False,
    return_cval: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Compute diffusion maps embedding using PyTorch.

    Args:
        X_np: Input data array of shape (N, D) where N is number of points
              and D is the feature dimension.
        cutoff: Cumulative eigenvalue energy cutoff for selecting components (default: 0.99).
        tol: Tolerance for LOBPCG solver (default: 1e-3).
        diffusion_time: Diffusion time parameter t, scales eigenvalues as lambda^t (default: 10).
        alpha: Normalization parameter for the kernel (default: 0.0).
        eig_solver: Eigenvalue solver to use, "full" or "lobpcg" (default: "full").
        k: Number of eigenpairs to compute (default: None, uses cutoff).
        device: Torch device for the computation. ``None`` autodetects via
                :func:`se_probe.device.get_device` (CUDA -> MPS -> CPU).
        return_eigs: If True, also return eigenvalues (default: False).
        return_complement: If True, also return complement of selected components (default: False).
        return_cval: If True, also return c-value for mixing time estimation (default: False).

    Returns:
        Psi: Diffusion coordinates array of shape (N, L) where L is selected by cutoff.
        If return_eigs: also returns eigenvalues array.
        If return_complement: also returns complement coordinates.
        If return_cval: also returns c-value.

    Algorithm:
        1. Compute pairwise squared distances
        2. Build Gaussian kernel: K = exp(-D² / 2ε) where ε = median(D²)
        3. Apply symmetric normalization: K_sym = D^(-1/2) K D^(-1/2)
        4. Eigendecomposition of K_sym
        5. Select components by cumulative energy cutoff
        6. Scale by diffusion time: Ψ = ψ × λ^t
    """
    # Device handling — autodetect CUDA -> MPS -> CPU when not specified
    if not isinstance(device, torch.device):
        device = get_device(device)

    # Convert to tensor on the requested device
    X = torch.as_tensor(X_np, dtype=torch.float32, device=device)
    N = X.shape[0]

    # Pairwise squared distances (chunked for memory safety)
    if N > 1000:
        chunk_size = min(500, N)
        D2 = torch.zeros((N, N), dtype=X.dtype, device=X.device)
        for i0 in range(0, N, chunk_size):
            i1 = min(i0 + chunk_size, N)
            Xi = X[i0:i1]
            for j0 in range(0, N, chunk_size):
                j1 = min(j0 + chunk_size, N)
                Xj = X[j0:j1]
                D2[i0:i1, j0:j1] = torch.cdist(Xi, Xj).pow_(2)
    else:
        D2 = torch.cdist(X, X).pow_(2)

    # Kernel construction
    iu, ju = torch.triu_indices(N, N, offset=1, device=X.device)
    eps = torch.median(D2[iu, ju])
    # Guard against degenerate eps
    eps = torch.clamp(eps, min=torch.finfo(D2.dtype).eps)
    K = torch.exp(-D2 / (2 * eps))

    d = K.sum(dim=1)
    if alpha != 0.0:
        d_alpha_inv = d.pow(-alpha)
        K = d_alpha_inv[:, None] * K * d_alpha_inv[None, :]
        d = K.sum(dim=1)

    if return_cval:
        pi_min = d.min() / d.sum()  # min stationary prob
        c_val = (pi_min.rsqrt() / math.log(2.0)).item()

    # Symmetric normalization
    D_half_inv = torch.diag(torch.rsqrt(torch.clamp(d, min=torch.finfo(d.dtype).eps)))
    K_sym = D_half_inv @ K @ D_half_inv

    # Eigen decomposition
    if eig_solver == "lobpcg":
        m = k if k is not None else min(N - 1, 50)
        # LOBPCG expects symmetric; provide init
        init = torch.randn(N, m, dtype=K_sym.dtype, device=K_sym.device)
        vals, vecs = torch.lobpcg(K_sym, k=m, X=init, niter=200, tol=tol, largest=True)
    elif eig_solver == "full":
        vals, vecs = torch.linalg.eigh(K_sym)     # ascending
        vals, vecs = vals.flip(0), vecs.flip(1)   # make descending
        if k is not None:
            vecs = vecs[:, :k + 1]
            vals = vals[:k + 1]
    else:
        raise ValueError(f"Unknown eig_solver '{eig_solver}'")

    # Discard the trivial first eigenpair
    psi = vecs[:, 1:]
    lam = vals[1:]

    # Choose number of components by cumulative energy (cutoff)
    cum = torch.cumsum(lam, dim=0)
    L = int((cum / torch.clamp(cum[-1], min=torch.finfo(cum.dtype).eps) < cutoff).sum().item()) + 1

    # Diffusion time scaling
    lam_pow = lam.pow(diffusion_time)
    psi_all = psi * lam_pow

    Psi = psi_all[:, :L]
    Psi_rest = psi_all[:, L:]

    # Returns
    Psi_np = Psi.cpu().numpy()
    if return_complement and return_eigs and return_cval:
        return Psi_np, Psi_rest.cpu().numpy(), lam.cpu().numpy(), c_val
    if return_complement and return_eigs:
        return Psi_np, Psi_rest.cpu().numpy(), lam.cpu().numpy()
    if return_complement:
        return Psi_np, Psi_rest.cpu().numpy()
    if return_eigs:
        return Psi_np, lam.cpu().numpy()
    return Psi_np
