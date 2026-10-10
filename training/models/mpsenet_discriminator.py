"""MPSENet discriminator + batch_pesq utility.

Copied from Meta-Interface-Asteroid:
  - MetricDiscriminator + LearnableSigmoid1d from asteroid/losses/mpsenet_metric_discriminator.py
  - batch_pesq from asteroid/losses/mpsenet_loss.py

Key difference from MUSE discriminator: default dim=64 (vs MUSE dim=16).
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from pesq import pesq
from joblib import Parallel, delayed


class MetricDiscriminator(nn.Module):
    """Metric discriminator that compares two spectral inputs.

    Args:
        dim (int): Base dimension for the network layers. Defaults to 64.
        in_channel (int): Number of input channels. Defaults to 2.
    """
    def __init__(self, dim: int = 64, in_channel: int = 2) -> None:
        super(MetricDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, dim, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True),
            nn.PReLU(dim),
            nn.utils.spectral_norm(nn.Conv2d(dim, dim * 2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 2, affine=True),
            nn.PReLU(dim * 2),
            nn.utils.spectral_norm(nn.Conv2d(dim * 2, dim * 4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 4, affine=True),
            nn.PReLU(dim * 4),
            nn.utils.spectral_norm(nn.Conv2d(dim * 4, dim * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 8, affine=True),
            nn.PReLU(dim * 8),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(dim * 8, dim * 4)),
            nn.Dropout(0.3),
            nn.PReLU(dim * 4),
            nn.utils.spectral_norm(nn.Linear(dim * 4, 1)),
            LearnableSigmoid1d(1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = torch.stack((x, y), dim=1)
        return self.layers(xy)


class LearnableSigmoid1d(nn.Module):
    def __init__(self, in_features: int, beta: float = 1) -> None:
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * torch.sigmoid(self.slope * x)


# ---------- batch_pesq (from asteroid/losses/mpsenet_loss.py) ----------

def _cal_pesq(clean: np.ndarray, noisy: np.ndarray, sr: int = 16000) -> float:
    try:
        return pesq(sr, clean, noisy, 'wb')
    except Exception:
        return -1


def batch_pesq(clean: List[np.ndarray], noisy: List[np.ndarray]) -> Optional[torch.Tensor]:
    pesq_score = Parallel(n_jobs=15)(delayed(_cal_pesq)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score)
