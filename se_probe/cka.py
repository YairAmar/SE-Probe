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

