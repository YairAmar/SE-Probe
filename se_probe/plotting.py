"""Paper-quality matplotlib styling and shared model colour/label maps."""
from __future__ import annotations

from typing import Dict

import matplotlib as mpl

__all__ = ["apply_paper_rcparams", "MODEL_COLORS", "MODEL_LABELS"]


MODEL_COLORS: Dict[str, str] = {
    "muse": "#2176AE",
    "mpsenet": "#D95319",
    "demucs": "#77AC30",
}

MODEL_LABELS: Dict[str, str] = {
    "muse": "MUSE",
    "mpsenet": "MP-SENet",
    "demucs": "Demucs",
}


def apply_paper_rcparams() -> None:
    """Apply the paper/poster-quality matplotlib rcParams used by the notebooks."""
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 72,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })
