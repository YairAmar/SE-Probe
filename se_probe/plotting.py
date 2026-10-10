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


# ----------------------------------------------------------------------------- #
# Paper figure helpers (TASLP revision). All take a matplotlib Axes and draw in
# the published styling; nothing here saves a file.
# ----------------------------------------------------------------------------- #
#: Okabe-Ito palette used by the column-width figures.
OKABE_ITO = {
    "blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "orange": "#E69F00",
    "purple": "#CC79A7", "sky": "#56B4E9", "gray": "#555555",
}
#: Degradation-axis colours.
AXIS_COLORS = {"snr": OKABE_ITO["blue"], "c50": OKABE_ITO["vermillion"]}
#: Marker per model, so figures survive greyscale.
MODEL_MARKERS = {"muse": "o", "mpsenet": "s", "demucs": "^"}
MODEL_DASHES = {"muse": (0, (3.6, 1.5)), "mpsenet": (0, (1.1, 1.1)), "demucs": (0, (5.0, 1.4, 0.9, 1.4))}
PROFILE_SLOPE_COLOR = "#2176AE"
PROFILE_INTERCEPT_COLOR = "#E8890C"
BAND_GRAY = "0.93"


def apply_column_rcparams() -> None:
    """rcParams for figures authored at a 3.5 in IEEE column width (no downscaling)."""
    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "figure.dpi": 100, "savefig.dpi": 300, "pdf.fonttype": 42, "axes.grid": False,
    })


def depth_axis(ax, layout: dict, show_labels: bool = True) -> None:
    """Block bands, dashed separators and block ticks of a depth axis (see
    :func:`se_probe.layers.layout`)."""
    for lo, hi in layout["bands"]:
        ax.axvspan(lo, hi, color=BAND_GRAY, zorder=0, lw=0)
    for sep in layout["seps"]:
        ax.axvline(sep, color="k", ls="--", lw=0.6, zorder=1)
    ax.set_xlim(-0.5, layout["n"] - 0.5)
    ax.set_xticks(layout["ticks"])
    ax.set_xticklabels(layout["ticklabels"] if show_labels else [],
                       rotation=layout.get("rotation", 0),
                       ha="right" if layout.get("rotation", 0) else "center")


def mark_skip_junctions(ax, y, layout: dict) -> None:
    """Star the encoder skip sources and triangle the decoder skip targets."""
    import numpy as np

    y = np.asarray(y, float)
    if layout["skip_out"]:
        ax.scatter(layout["skip_out"], y[layout["skip_out"]], marker="*", s=110, color="orange",
                   edgecolors="k", linewidths=0.4, zorder=4, label="skip source")
    if layout["skip_in"]:
        ax.scatter(layout["skip_in"], y[layout["skip_in"]], marker="^", s=55, color="green",
                   edgecolors="k", linewidths=0.4, zorder=4, label="skip target")


def profile_panel_pair(ax_slope, ax_intercept, table, layout: dict, axis_tag: str = "SNR",
                       beta_ci=("beta_lo", "beta_hi"), alpha_ci=("alpha_lo", "alpha_hi"),
                       title_prefix: str = "") -> None:
    """The paper's per-layer profile figure: slope ``beta`` above, intercept ``alpha``
    below, both against depth, with shaded CI bands and the block layout.

    ``table`` is a DataFrame in depth order with ``alpha``, ``beta`` and (optionally)
    the CI columns named by ``beta_ci`` / ``alpha_ci``.
    """
    import numpy as np

    x = np.arange(len(table))
    for ax, key, ci, color, marker, ylabel in (
            (ax_slope, "beta", beta_ci, PROFILE_SLOPE_COLOR, "o", r"$\beta_\ell$  (CKA/dB)"),
            (ax_intercept, "alpha", alpha_ci, PROFILE_INTERCEPT_COLOR, "s", rf"$\alpha_\ell$  (CKA at 0 dB {axis_tag})")):
        depth_axis(ax, layout, show_labels=ax is ax_intercept)
        if ci[0] in table.columns and ci[1] in table.columns:
            ax.fill_between(x, table[ci[0]].to_numpy(), table[ci[1]].to_numpy(),
                            color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(x, table[key].to_numpy(), marker=marker, ms=4, color=color, lw=1.4, zorder=3)
        mark_skip_junctions(ax, table[key].to_numpy(), layout)
        ax.set_ylabel(ylabel)
    ax_slope.set_title(f"{title_prefix}Sensitivity per layer ({axis_tag})")
    ax_intercept.set_title(f"{title_prefix}Robustness per layer ({axis_tag})")


def tradeoff_scatter(ax, tables: dict, unit: str = "SNR", annotate_stats: dict = None) -> None:
    """Robustness/sensitivity scatter, one cloud and dashed OLS line per model.

    ``tables`` maps a model key to a DataFrame with ``alpha`` and ``beta``;
    ``annotate_stats`` optionally maps the same keys to a ``tradeoff_summary`` dict
    whose ``r_alpha_beta`` and ``rho_alpha_beta`` go into the legend.
    """
    import numpy as np

    for model, t in tables.items():
        a, b = t["alpha"].to_numpy(float), t["beta"].to_numpy(float)
        label = MODEL_LABELS.get(model, model)
        if annotate_stats and model in annotate_stats:
            s = annotate_stats[model]
            label += rf"  ($r$ = {s['r_alpha_beta']:.3f}, $\rho$ = {s['rho_alpha_beta']:.3f})"
        ax.scatter(a, b, s=16, marker=MODEL_MARKERS.get(model, "o"), color=MODEL_COLORS.get(model, "k"),
                   alpha=0.7, linewidths=0, zorder=3, label=label)
        if len(a) > 1:
            sl, ic = np.polyfit(a, b, 1)
            xs = np.linspace(a.min(), a.max(), 20)
            ax.plot(xs, ic + sl * xs, ls=MODEL_DASHES.get(model, "--"), lw=0.9,
                    color=MODEL_COLORS.get(model, "k"), alpha=0.7, zorder=2)
    ax.set_xlabel(rf"Intercept $\alpha_\ell$  (CKA at 0 dB {unit})")
    ax.set_ylabel(rf"Slope $\beta_\ell$  ($\Delta$CKA per dB {unit})")
    ax.grid(True, color="#b0b0b0", lw=0.6, alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="0.8")


__all__ += ["OKABE_ITO", "AXIS_COLORS", "MODEL_MARKERS", "MODEL_DASHES", "apply_column_rcparams",
            "depth_axis", "mark_skip_junctions", "profile_panel_pair", "tradeoff_scatter"]
