"""
Demucs DNS64 model support for SE-Probe.

Demucs is a 1D convolutional U-Net with 5 encoder stages, a BLSTM bottleneck,
and 5 decoder stages, designed for speech denoising.
"""

from se_probe.demucs.model import load_demucs_model, load_demucs_activation_extractor
from se_probe.demucs.consts import LAYERS as DEMUCS_LAYERS
from se_probe.demucs.consts import BLOCK_OUTPUT_LAYERS as DEMUCS_BLOCK_OUTPUT_LAYERS

__all__ = [
    "load_demucs_model",
    "load_demucs_activation_extractor",
    "DEMUCS_LAYERS",
    "DEMUCS_BLOCK_OUTPUT_LAYERS",
]
