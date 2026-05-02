"""
MUSE model support for SE-Probe.

MUSE (Magnitude and phase speech enhancement) is a U-Net transformer
architecture for speech enhancement.
"""

from se_probe.muse.consts import LAYERS as MUSE_LAYERS
from se_probe.muse.model import load_muse_activation_extractor, load_muse_model

__all__ = [
    "load_muse_model",
    "load_muse_activation_extractor",
    "MUSE_LAYERS",
]
