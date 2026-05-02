"""
MUSE model architectures.
"""

from se_probe.muse.models.generator import MUSE
from se_probe.muse.models.pooling import pool_muse_activations, select_first_segment

__all__ = [
    "MUSE",
    "pool_muse_activations",
    "select_first_segment",
]
