"""
SE-Probe — probing representations of deep speech enhancement models.

Provides CKA, diffusion maps, and activation-extraction utilities for
analysing MUSE, MP-SENet, and Demucs.
"""

# Apple Silicon (MPS) backend doesn't implement every PyTorch op (e.g.
# aten::_linalg_eigh). PyTorch reads PYTORCH_ENABLE_MPS_FALLBACK during
# interpreter / first-MPS-dispatch initialisation, so it must be set BEFORE
# any torch import — including any transitively pulled in by submodules
# below. ``setdefault`` preserves a user override (export=0 to force hard
# failures).
import os as _os

_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from se_probe.activation_extraction import (
    ActivationsExtractor,
    extract_activations_on_audios,
    get_activations,
    load_demucs_activation_extractor,
    load_mpsenet_activation_extractor,
    load_muse_activation_extractor,
)
from se_probe.cka import cka
from se_probe.consts import (
    DEFAULT_SNRS,
    SAMPLE_RATE,
    TEST_NOISES,
    VOICEBANK_DEMAND_NOISES,
)
from se_probe.data_generation import add_noise_at_snr, load_demand_noise
from se_probe.diffusion_analysis import (
    BLOCK_NAMES,
    BLOCK_ORDER,
    REPRESENTATIVE_LAYERS,
    compute_distance_from_ref_snr,
    compute_layer_distance_matrix,
    create_psi_column,
    get_layer_order_key,
)
from se_probe.diffusion_maps import diffusion_map_torch
from se_probe.io import load_clean_wavs

__version__ = "0.1.2"

__all__ = [
    # CKA
    "cka",
    # Diffusion maps
    "diffusion_map_torch",
    # Activation extraction
    "ActivationsExtractor",
    "get_activations",
    "extract_activations_on_audios",
    "load_muse_activation_extractor",
    "load_mpsenet_activation_extractor",
    "load_demucs_activation_extractor",
    # Data generation
    "add_noise_at_snr",
    "load_demand_noise",
    # I/O
    "load_clean_wavs",
    # Constants
    "SAMPLE_RATE",
    "DEFAULT_SNRS",
    "VOICEBANK_DEMAND_NOISES",
    "TEST_NOISES",
    # Diffusion analysis
    "create_psi_column",
    "compute_distance_from_ref_snr",
    "compute_layer_distance_matrix",
    "get_layer_order_key",
    "REPRESENTATIVE_LAYERS",
    "BLOCK_NAMES",
    "BLOCK_ORDER",
]
