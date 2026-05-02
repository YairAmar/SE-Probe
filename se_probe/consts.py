"""
Global constants for the SE-Probe package.

Data paths can be configured via environment variables (or
:func:`set_paths` for programmatic configuration in tests):

- ``SEPROBE_DEMAND_DIR``  — DEMAND noise dataset (16 kHz)
- ``SEPROBE_VCTK_DIR``    — VCTK corpus (wav48 directory)
- ``SEPROBE_AIR_RIR_DIR`` — AIR room impulse response dataset

The path constants ``DEMAND_NOISE_DIR``, ``VCTK_DIR`` and ``AIR_RIR_DIR``
are resolved lazily on first access; importing :mod:`se_probe.consts`
does not require any of them to be set. Accessing one without a
corresponding env var (or :func:`set_paths` call) raises
:class:`EnvironmentError`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_SNRS",
    "SAMPLE_RATE",
    "DEFAULT_NOISE_NAME",
    "DEMAND_NOISE_DIR",
    "VCTK_DIR",
    "TEST_SPEAKERS",
    "VOICEBANK_DEMAND_NOISES",
    "SNR_RANGE_DIFFUSION",
    "TEST_NOISES",
    "set_paths",
    # Reverb constants
    "EARLY_REFLECTION_IDX",
    "DEFAULT_TARGET_C50S",
    "REFERENCE_C50",
    "AIR_RIR_DIR",
    "AIR_TEST_ROOMS",
    "AIR_TRAIN_ROOMS",
    "CKA_RIRS",
    "N_RIRS_PER_UTTERANCE",
]

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# SNR values for additive noise experiments (None = clean, no noise)
DEFAULT_SNRS = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, None]

# Audio configuration
SAMPLE_RATE = 16000

# DEMAND noise dataset configuration
DEFAULT_NOISE_NAME = "SPSQUARE"

TEST_SPEAKERS = ['p226', 'p287']

# VoiceBank-DEMAND test set noise types
VOICEBANK_DEMAND_NOISES = [
    'DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', 'NRIVER',
    'OHALLWAY', 'PRESTO', 'SPSQUARE', 'STRAFFIC', 'TBUS', 'TCAR', 'TMETRO', 'PCAFETER', 'OOFFICE'
]

# SNR range for diffusion maps extraction (-10 to 30 dB in 1 dB steps)
SNR_RANGE_DIFFUSION = list(range(-10, 31))  # 41 values: -10, -9, ..., 29, 30

# Test noises for diffusion maps analysis (subset of VoiceBank-DEMAND)
TEST_NOISES = ['TBUS', 'PCAFETER', 'DLIVING', 'OOFFICE', 'SPSQUARE']

# =============================================================================
# Reverb constants
# =============================================================================

# Early reflection boundary: 50ms at 16kHz = 800 samples
EARLY_REFLECTION_IDX = 800

# Target C50 values for reverb experiments (13 values from -5 to 25 dB, 2.5 dB steps)
DEFAULT_TARGET_C50S = np.arange(-5, 27.5, 2.5).tolist()

# Reference C50 for CKA comparison (lightly reverberant)
REFERENCE_C50 = 50.0

# Train/test split by room type
AIR_TEST_ROOMS = ["office", "meeting", "lecture", "bathroom", "kitchen", "corridor"]
AIR_TRAIN_ROOMS = ["booth", "aula_carolina", "stairway"]

# Curated 10 test RIRs for controlled CKA vs C50 analysis
# Selected for room diversity: 5 binaural + 5 phone, native C50 range 6.5-22.8 dB
CKA_RIRS = [
    "air_binaural_office_0_0_1.mat",               # office, C50=14.6
    "air_binaural_meeting_0_0_1.mat",              # meeting, C50=19.3
    "air_binaural_lecture_0_0_1.mat",              # lecture, C50=8.3
    "air_binaural_aula_carolina_0_1_1_90_3.mat",   # large hall, C50=14.6
    "air_binaural_stairway_0_1_1_0.mat",           # stairway, C50=6.5
    "air_phone_bathroom_hfrp_0.mat",               # bathroom, C50=12.1
    "air_phone_kitchen_hfrp_0.mat",                # kitchen, C50=11.5
    "air_phone_corridor_hfrp_0.mat",               # corridor, C50=11.1
    "air_phone_office_hfrp_0.mat",                 # office, C50=12.7
    "air_phone_meeting_hfrp_0.mat",                # meeting, C50=22.8
]

# Number of RIRs to randomly sample per utterance (for full test-set experiments)
N_RIRS_PER_UTTERANCE = 5

# =============================================================================
# Lazy data-path resolution
# =============================================================================

_PATH_ENV = {
    "DEMAND_NOISE_DIR": "SEPROBE_DEMAND_DIR",
    "VCTK_DIR": "SEPROBE_VCTK_DIR",
    "AIR_RIR_DIR": "SEPROBE_AIR_RIR_DIR",
}

_path_overrides: dict[str, str] = {}


def set_paths(
    *,
    vctk: Optional[Union[str, Path]] = None,
    demand: Optional[Union[str, Path]] = None,
    air_rir: Optional[Union[str, Path]] = None,
) -> None:
    """Programmatically set dataset paths (overrides env vars).

    Useful for tests with fixture data. Pass only the paths you want to set;
    leave others as ``None`` to keep their current resolution.
    """
    if vctk is not None:
        _path_overrides["VCTK_DIR"] = str(vctk)
    if demand is not None:
        _path_overrides["DEMAND_NOISE_DIR"] = str(demand)
    if air_rir is not None:
        _path_overrides["AIR_RIR_DIR"] = str(air_rir)


def _resolve_path(name: str) -> str:
    if name in _path_overrides:
        return _path_overrides[name]
    env_var = _PATH_ENV[name]
    value = os.environ.get(env_var)
    if value is None:
        raise EnvironmentError(
            f"Set ${env_var} or call se_probe.consts.set_paths(...) "
            f"to configure {name}."
        )
    return value


def __getattr__(name: str):  # PEP 562: lazy module attribute access
    if name in _PATH_ENV:
        return _resolve_path(name)
    raise AttributeError(f"module 'se_probe.consts' has no attribute {name!r}")
