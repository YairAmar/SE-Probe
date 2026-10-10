"""Probed-layer sets, architectural depth order and labels for the three models.

Every per-layer statistic in the paper is computed over a fixed *probed* subset of
the hooked layers, listed in **architectural depth order**:

* MUSE: the 24 ``.norm1`` hooks (first LayerNorm of each transformer layer), ordered
  by stage ``encoder_level1, encoder_level2, latent, decoder_level2, decoder_level1,
  mag_refinement`` and then by transformer-layer index.
* MP-SENet: the 8 ``.norm1`` hooks of ``TSTransformer.{0..3}.{time,freq}_transformer``,
  ordered by block index with the time sub-block before the frequency sub-block.
* Demucs: the 11 block outputs ``encoder.0-4, lstm, decoder.0-4``.

Sorting layer names alphabetically is **not** depth order (it ranks Demucs decoder
blocks before its encoder blocks and scrambles MUSE's stages); that mistake once
produced wrong depth correlations, so the ordering lives here and nowhere else.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np

__all__ = [
    "MODELS",
    "MUSE_STAGES",
    "MUSE_STAGE_LABELS",
    "MUSE_BLOCK_LABELS",
    "MUSE_SKIP_OUT",
    "MUSE_SKIP_IN",
    "MPSENET_TICKS",
    "DEMUCS_BLOCKS",
    "DEMUCS_TICKS",
    "N_PROBED_LAYERS",
    "SNR_GRID",
    "C50_GRID",
    "C50_WIDE_GRID",
    "DEMAND_NOISES_18",
    "VOICEBANK_DEMAND_TEST_NOISES",
    "VOICEBANK_DEMAND_TRAIN_NOISES",
    "DEMAND_NEITHER_SPLIT",
    "HELD_OUT_NOISES_OLD",
    "AIR_TEST_POOL_ROOM_COUNTS",
    "AIR_TEST_POOL_SIZE",
    "SPEAKER_RANGES",
    "probed_layers",
    "depth_order",
    "depth_index",
    "short_label",
    "short_labels",
    "muse_stage",
    "muse_stage_of",
    "layout",
]

MODELS = ("muse", "mpsenet", "demucs")

# --------------------------------------------------------------------------- MUSE
MUSE_STAGES = [
    "encoder_level1", "encoder_level2", "latent",
    "decoder_level2", "decoder_level1", "mag_refinement",
]
MUSE_STAGE_LABELS = {
    "encoder_level1": "Enc-L1", "encoder_level2": "Enc-L2", "latent": "Latent",
    "decoder_level2": "Dec-L2", "decoder_level1": "Dec-L1", "mag_refinement": "Refine",
}
MUSE_BLOCK_LABELS = [MUSE_STAGE_LABELS[s] for s in MUSE_STAGES]
#: Depth indices of the encoder skip *sources* (Enc-L1.3, Enc-L2.3) and the decoder
#: skip *targets* (Dec-L2.0, Dec-L1.0, Refine.0) among the 24 probed MUSE layers.
MUSE_SKIP_OUT = [3, 7]
MUSE_SKIP_IN = [12, 16, 20]

# ------------------------------------------------------------------------ MP-SENet
MPSENET_TICKS = ["T0", "F0", "T1", "F1", "T2", "F2", "T3", "F3"]

# -------------------------------------------------------------------------- Demucs
DEMUCS_BLOCKS = [f"encoder.{i}" for i in range(5)] + ["lstm"] + [f"decoder.{i}" for i in range(5)]
DEMUCS_TICKS = ["Enc-0", "Enc-1", "Enc-2", "Enc-3", "Enc-4", "LSTM",
                "Dec-0", "Dec-1", "Dec-2", "Dec-3", "Dec-4"]

N_PROBED_LAYERS = {"muse": 24, "mpsenet": 8, "demucs": 11}

# --------------------------------------------------------------------------- grids
#: 41 integer SNR levels of the noise sweep (dB).
SNR_GRID = np.arange(-10, 31)
#: 13 nominal target C50 levels of the reverberation sweep (dB).
C50_GRID = np.arange(-5.0, 25.1, 2.5)
#: 14-level wide window used only by the window-stability ablation.
C50_WIDE_GRID = np.linspace(-20.0, 50.0, 14)

# -------------------------------------------------------------------------- noises
#: The eighteen DEMAND recordings of the full-scale noise sweep.
DEMAND_NOISES_18 = [
    "DKITCHEN", "DLIVING", "DWASHING", "NFIELD", "NPARK", "NRIVER",
    "OHALLWAY", "OMEETING", "OOFFICE", "PCAFETER", "PRESTO", "PSTATION",
    "SCAFE", "SPSQUARE", "STRAFFIC", "TBUS", "TCAR", "TMETRO",
]
#: The five recordings that constitute the VoiceBank-DEMAND *test* split (held out
#: from the pretrained MUSE).
VOICEBANK_DEMAND_TEST_NOISES = ["TBUS", "SCAFE", "DLIVING", "OOFFICE", "SPSQUARE"]
#: The eight recordings of the VoiceBank-DEMAND *training* split.
VOICEBANK_DEMAND_TRAIN_NOISES = ["DKITCHEN", "OMEETING", "PCAFETER", "PRESTO",
                                 "PSTATION", "TCAR", "TMETRO", "STRAFFIC"]
#: The five DEMAND recordings that enter neither split.
DEMAND_NEITHER_SPLIT = ["DWASHING", "NFIELD", "NPARK", "NRIVER", "OHALLWAY"]
#: The earlier "held-out five" with PCAFETER in place of SCAFE (superseded).
HELD_OUT_NOISES_OLD = ["TBUS", "PCAFETER", "DLIVING", "OOFFICE", "SPSQUARE"]

# ---------------------------------------------------------------------------- RIRs
#: Room histogram of the 88-RIR six-room AIR probing pool (56 binaural + 32 phone).
AIR_TEST_POOL_ROOM_COUNTS = {"lecture": 32, "meeting": 24, "office": 18,
                             "corridor": 6, "bathroom": 4, "kitchen": 4}
AIR_TEST_POOL_SIZE = 88

# ------------------------------------------------------------------------ speakers
#: Inclusive ``clean_idx`` ranges of the probing speakers in each utterance set.
SPEAKER_RANGES = {
    "780": {"p226": (0, 355), "p287": (356, 779)},
    "824": {"p232": (0, 392), "p257": (393, 823)},
}


def _check_model(model: str) -> str:
    m = model.lower().replace("-", "").replace("_", "")
    if m == "mpsenet":
        return "mpsenet"
    if m not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")
    return m


def muse_stage_of(layer: str) -> str:
    """Stage name (``encoder_level1`` ...) of a MUSE module path."""
    for s in MUSE_STAGES:
        if f".{s}." in layer or layer.startswith(f"{s}."):
            return s
    raise ValueError(f"not a MUSE transformer layer: {layer!r}")


def muse_stage(layer: str) -> str:
    """Short stage label (``Enc-L1`` ...) of a MUSE module path."""
    return MUSE_STAGE_LABELS[muse_stage_of(layer)]


def probed_layers(model: str, all_layers: Iterable[str]) -> List[str]:
    """The paper's probed subset of ``all_layers``, in architectural depth order."""
    model = _check_model(model)
    all_layers = list(all_layers)
    if model in ("muse", "mpsenet"):
        keep = [L for L in all_layers if L.endswith(".norm1")]
    else:
        present = set(all_layers)
        keep = [L for L in DEMUCS_BLOCKS if L in present]
    return depth_order(model, keep)


def _muse_key(layer: str):
    p = layer.split(".")
    stage = MUSE_STAGES.index(muse_stage_of(layer))
    blk = int(p[p.index("mhca_blks") + 1]) if "mhca_blks" in p else 0
    sub = int(p[p.index("transformer_layers") + 1]) if "transformer_layers" in p else 0
    return (stage, blk, sub, layer)


def _mpsenet_key(layer: str):
    p = layer.split(".")
    try:
        blk = int(p[1])
    except (IndexError, ValueError):
        blk = 99
    return (blk, 0 if "time_" in layer else 1, layer)


def _demucs_key(layer: str):
    return (DEMUCS_BLOCKS.index(layer) if layer in DEMUCS_BLOCKS else 99, layer)


def depth_order(model: str, layers: Iterable[str]) -> List[str]:
    """Sort ``layers`` by architectural depth for ``model`` (never alphabetically)."""
    model = _check_model(model)
    key = {"muse": _muse_key, "mpsenet": _mpsenet_key, "demucs": _demucs_key}[model]
    return sorted(set(layers), key=key)


def depth_index(model: str, layers: Sequence[str]) -> Dict[str, int]:
    """Map each layer to its 0-based depth index."""
    return {L: i for i, L in enumerate(depth_order(model, layers))}


def short_label(model: str, layer: str) -> str:
    """Human-readable label: ``Enc-L1.0``, ``T0`` / ``F0``, ``Enc-0`` / ``LSTM``."""
    model = _check_model(model)
    if model == "muse":
        p = layer.split(".")
        sub = p[p.index("transformer_layers") + 1] if "transformer_layers" in p else "?"
        return f"{muse_stage(layer)}.{sub}"
    if model == "mpsenet":
        p = layer.split(".")
        return f"{'T' if 'time_' in layer else 'F'}{p[1]}"
    return DEMUCS_TICKS[DEMUCS_BLOCKS.index(layer)] if layer in DEMUCS_BLOCKS else layer


def short_labels(model: str, layers: Sequence[str]) -> List[str]:
    return [short_label(model, L) for L in layers]


def layout(model: str) -> dict:
    """Block bands, separators, ticks and skip markers for depth-axis figures."""
    model = _check_model(model)
    if model == "muse":
        return dict(
            n=24,
            bands=[(b * 4 - 0.5, b * 4 + 3.5) for b in (0, 2, 4)],
            seps=[3.5, 7.5, 11.5, 15.5, 19.5],
            ticks=[i * 4 + 1.5 for i in range(6)],
            ticklabels=list(MUSE_BLOCK_LABELS), rotation=30,
            skip_out=list(MUSE_SKIP_OUT), skip_in=list(MUSE_SKIP_IN),
        )
    if model == "mpsenet":
        return dict(
            n=8, bands=[(b * 2 - 0.5, b * 2 + 1.5) for b in (0, 2)],
            seps=[1.5, 3.5, 5.5], ticks=list(range(8)),
            ticklabels=list(MPSENET_TICKS), rotation=0, skip_out=[], skip_in=[],
        )
    return dict(
        n=11, bands=[(-0.5, 4.5), (5.5, 10.5)], seps=[4.5, 5.5], ticks=list(range(11)),
        ticklabels=list(DEMUCS_TICKS), rotation=30, skip_out=[], skip_in=[],
    )
