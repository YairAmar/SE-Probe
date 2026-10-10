import torch

__all__ = ["cka"]


def _center_columns(M: torch.Tensor) -> torch.Tensor:
    """
    Center each feature (column) by subtracting its mean over rows (tokens).
    M: (B, N, H)
    """
    return M - M.mean(dim=1, keepdim=True)

def cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Linear CKA between two time-averaged activation sets.

    Inputs:
      X, Y: activations for the *same utterance+noise*, different SNRs.
            Each can be (B, T_p, F_p, H) or already time-averaged (B, F_p, H).

    Steps:
      1) Time-average to (B, F_p, H).
      2) Treat F_p as "tokens": X, Y in R^{B x N x H} with N=F_p.
      3) Compute linear CKA with feature centering.
    """
    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch after time-avg: {tuple(X.shape)} vs {tuple(Y.shape)}")

    # Center features (columns)
    Xc = _center_columns(X)
    Yc = _center_columns(Y)

    # Cross-covariance in feature space
    # Matrix multiplication for batched (B, N, H): want (B, H, H) output
    XtY = torch.matmul(Xc.transpose(1, 2), Yc)    # (B, H, H)
    XtX = torch.matmul(Xc.transpose(1, 2), Xc)    # (B, H, H)
    YtY = torch.matmul(Yc.transpose(1, 2), Yc)    # (B, H, H)

    # Now compute per-batch CKA, then average
    num = torch.sum(XtY * XtY, dim=[1,2])  # (B,) sum over last two dims
    denom = torch.norm(XtX, p='fro', dim=[1,2]) * torch.norm(YtY, p='fro', dim=[1,2]) + eps  # (B,)
    cka = num / denom  # (B,)
    cka = cka.tolist()  # return as a list, one value per sample in the batch
    # numerical guard
    # Apply numerical guards to each value in cka list
    cka = [max(0.0, min(1.0, float(val))) for val in cka]
    return cka



# --------------------------------------------------------------------------- #
# Estimator and sample-convention variants (ablation of Sec. "Centered Kernel
# Alignment"). The paper's statistic is the biased linear estimator above on the
# time-pooled [C, F] representation; these let the same comparison be re-run with
# the unbiased HSIC estimator and with frames rather than channels as samples.
# --------------------------------------------------------------------------- #
_EPS = 1e-12


def linear_cka_biased(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Biased linear CKA between two ``(n_samples, n_features)`` matrices.

    Feature-space form ``||X'Y||_F^2 / (||X'X||_F ||Y'Y||_F)`` on column-centred
    matrices; equals :func:`cka` on a single pooled representation.
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    XtY = Xc.t() @ Yc
    XtX = Xc.t() @ Xc
    YtY = Yc.t() @ Yc
    num = XtY.pow(2).sum()
    den = torch.linalg.norm(XtX) * torch.linalg.norm(YtY) + _EPS
    return float(torch.clamp(num / den, 0.0, 1.0))


def hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator (Song et al. 2012) for ``n x n`` Gram matrices.

    Requires ``n > 3``. This is the estimator used for CKA-based network
    comparison by Nguyen, Raghu and Kornblith (2021).
    """
    n = K.shape[0]
    if n <= 3:
        raise ValueError(f"unbiased HSIC needs more than 3 samples, got {n}")
    K = K - torch.diag(torch.diag(K))
    L = L - torch.diag(torch.diag(L))
    Ks1 = K.sum(dim=1)
    Ls1 = L.sum(dim=1)
    trKL = (K * L).sum()
    term2 = (K.sum() * L.sum()) / ((n - 1) * (n - 2))
    term3 = (2.0 / (n - 2)) * (Ks1 @ Ls1)
    return (trKL + term2 - term3) / (n * (n - 3))


def linear_cka_unbiased(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Unbiased linear CKA via :func:`hsic_unbiased` on ``K = XX'``, ``L = YY'``."""
    K = X @ X.t()
    L = Y @ Y.t()
    hkl = hsic_unbiased(K, L)
    hkk = hsic_unbiased(K, K)
    hll = hsic_unbiased(L, L)
    den = torch.sqrt(torch.clamp(hkk * hll, min=_EPS))
    return float(torch.clamp(hkl / (den + _EPS), 0.0, 1.0))


def representations_from_activation(act: torch.Tensor) -> dict:
    """The sample/feature conventions compared in the ablation, from one non-pooled
    first-segment activation ``(C, T, F)``.

    Returns ``{"pooled": (C, F), "frames_TFC": (T, C*F), "frames_TC": (T, C)}``:

    * ``pooled``: time-averaged, channels as samples and frequency bins as features
      (the paper's convention for MUSE and MP-SENet);
    * ``frames_TFC``: time frames as samples with the full ``C*F`` vector per frame;
    * ``frames_TC``: time frames as samples, channels as features after averaging
      over frequency (the Demucs-style ``[T, C]`` convention).
    """
    if act.dim() != 3:
        raise ValueError(f"expected a (C, T, F) activation, got shape {tuple(act.shape)}")
    C, T, F = act.shape
    return {
        "pooled": act.mean(dim=1),
        "frames_TFC": act.permute(1, 0, 2).reshape(T, C * F),
        "frames_TC": act.mean(dim=2).t(),
    }


CKA_VARIANTS = ("pooled_linear", "hsic_unbiased", "frames_TFC", "frames_TC")


def cka_variants(act_clean: torch.Tensor, act_degraded: torch.Tensor) -> dict:
    """All four estimator/convention variants for one clean/degraded activation pair."""
    rc = representations_from_activation(act_clean)
    rn = representations_from_activation(act_degraded)
    return {
        "pooled_linear": linear_cka_biased(rc["pooled"], rn["pooled"]),
        "hsic_unbiased": linear_cka_unbiased(rc["pooled"], rn["pooled"]),
        "frames_TFC": linear_cka_biased(rc["frames_TFC"], rn["frames_TFC"]),
        "frames_TC": linear_cka_biased(rc["frames_TC"], rn["frames_TC"]),
    }


__all__ += ["linear_cka_biased", "hsic_unbiased", "linear_cka_unbiased",
            "representations_from_activation", "cka_variants", "CKA_VARIANTS"]
