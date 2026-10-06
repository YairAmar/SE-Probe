"""Hierarchical (cluster) bootstrap confidence intervals.

Every confidence interval reported in the accompanying TASLP paper is produced by
a hierarchical cluster bootstrap over the *sampling units* of the probing design,
not by resampling individual rows of the results table. This module is the
reference implementation of that procedure.

Why clusters
------------
A probing sweep is deeply nested and crossed. One row of a results table is a
single ``(utterance, degradation-level, layer)`` measurement, but utterances are
nested in speakers, and each utterance is crossed with every noise type (noise
axis) or every room impulse response (reverberation axis). Rows are therefore
very far from independent. Resampling rows -- or resampling the mean
degradation-level curve -- propagates only the residual scatter about the fit and
silently discards the dominant variance components. In the paper's own sweeps
that understates interval widths by one to two orders of magnitude.

The estimand
------------
**Read this before quoting any interval from this module.** A bootstrap interval
here describes *sampling variation over the units that were resampled,
conditional on everything held fixed*. It is not a generic "95% CI".

* ``unit="utterance"`` gives the sampling variation you would see if the same
  speakers, the same noise types and the same RIRs were re-recorded with a fresh
  draw of utterances. It says nothing about generalization to new speakers.
* ``unit="speaker"`` gives sampling variation over speakers, and is the only
  choice that supports a claim about a speaker population.
* ``unit="noise_type"`` gives sampling variation over the noise corpus.

Everything *above* ``unit`` in the nesting is held fixed and contributes no
uncertainty. Choosing a unit that is finer than the level your scientific claim
is about is pseudo-replication, and it is the single most common way to
manufacture a spuriously tight interval.

The paper measures this directly. For the association between a layer's CKA and
the utterance-level PESQ gain at the deepest second-decoder layer of MUSE, an
utterance-cluster interval has width 0.041 while a speaker-level interval at the
same layer has width 0.505 -- roughly twelve times wider, and spanning zero. The
two disagree because the first resamples the wrong unit. Section III-B of the
paper consequently reports speaker-level uncertainty throughout and confines its
conclusions to associations that survive it.

For that reason ``unit`` is a **required keyword argument with no default**
everywhere in this module. There is no "safe" default to fall back on; the
correct unit is a property of the question being asked, and the caller has to
state it.

Design specification
--------------------
``levels`` is an ordered mapping, **outermost first**, describing the nesting of
one observation per entry, e.g.::

    levels = {"speaker": speaker_id, "utterance": utterance_id}

``unit`` names the primary sampling unit and must be one of those keys.
Resampling starts at ``unit``:

* levels *above* ``unit`` are held fixed (they contribute no variance, and a
  :class:`PseudoReplicationWarning` is issued so this is never silent);
* the ``unit`` level is resampled with replacement to its original size;
* levels *below* ``unit`` are **carried along whole**: drawing a speaker brings
  all of that speaker's utterances, and all of their rows, with it.

Carrying the lower levels whole is what the paper's scripts do, and it is the
statistically correct choice. For a balanced two-level design it estimates
``sigma_between**2 / n_units + sigma_within**2 / n_obs`` exactly. The tempting
alternative -- redrawing utterances within each drawn speaker -- adds a second,
spurious ``sigma_within**2 / n_obs`` term and inflates the interval by up to a
factor of sqrt(2) even when there is no between-speaker variance at all. It is
available as ``resample_within=True`` for the rare case where the inner level is
genuinely a fresh sample from a per-cluster superpopulation, but it is never the
default and the docstring says why.

Labels within one level must be globally unique. If the same utterance label
appears under two different speakers the design is crossed, not nested, and it is
rejected rather than silently reinterpreted -- see ``crossed`` and ``strata``
below.

Two further mechanisms mirror the paper's two axes:

* ``strata`` -- a per-observation label whose levels are *held fixed* while units
  are resampled independently within each. The paper's headline noise-axis
  interval is a noise-stratified utterance bootstrap: only five noise types exist,
  too few to bootstrap reliably, so they are fixed strata and the utterances are
  resampled within each one.
* ``crossed`` -- factors that are crossed with, rather than nested in, the unit
  and are resampled on their own margin. Observation weights multiply across
  margins, which is the standard multiplicative-weight cluster bootstrap. The
  paper's two-way sensitivity checks (utterance x noise type, utterance x RIR)
  use this.

Both mechanisms are reported in ``BootstrapResult`` so an interval is always
traceable to the design that produced it.

Implementation notes
--------------------
Cluster resampling is carried out as observation *multiplicities* rather than by
materializing duplicated index arrays: drawing a cluster twice is exactly the
same as giving each of its observations weight two. That is both far cheaper and
numerically identical for any weight-respecting statistic. Statistics that cannot
accept weights (medians, rank statistics) can request an explicit expanded index
array via ``expand_indices=True``.

Only NumPy is required. pandas is never imported; pass NumPy arrays, or
``df["col"].to_numpy()``. SciPy is imported lazily by :func:`t_interval` only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "PseudoReplicationWarning",
    "BootstrapResult",
    "NestedDesign",
    "hierarchical_resample_indices",
    "hierarchical_resample_weights",
    "hierarchical_bootstrap",
    "percentile_ci",
    "interval_width",
    "half_width",
    "t_interval",
    "weighted_mean",
    "weighted_pearson_r",
    "ols_alpha_beta",
    "PAPER_SEED",
    "PAPER_N_BOOT",
]

#: Seed used for the confidence intervals published in the paper.
PAPER_SEED = 0

#: Number of bootstrap replicates used for the intervals published in the paper.
PAPER_N_BOOT = 1000


class PseudoReplicationWarning(UserWarning):
    """The chosen sampling unit is nested inside a coarser level of the design.

    Variance from the coarser level is not propagated, so the resulting interval
    is narrower than the uncertainty of any claim about that coarser level. Pass
    ``acknowledge_pseudoreplication=True`` once you have confirmed that the finer
    unit really is the estimand you want.
    """


# --------------------------------------------------------------------------- #
# Interval helpers
# --------------------------------------------------------------------------- #
def percentile_ci(samples, ci_level: float = 0.95, axis: int = 0):
    """Equal-tailed percentile confidence interval of bootstrap replicates.

    Parameters
    ----------
    samples : array_like
        Bootstrap replicates. Leading axis (or ``axis``) indexes replicates; any
        remaining axes are carried through, so a statistic returning one value
        per layer yields one interval per layer.
    ci_level : float
        Coverage, e.g. ``0.95`` for the 2.5/97.5 percentiles.
    axis : int
        Axis along which replicates are stored.

    Returns
    -------
    (low, high) : tuple of ndarray
        The interval describes sampling variation over whatever units were
        resampled, conditional on the corpus. It is not a population interval
        unless the resampled unit is the population unit.
    """
    samples = np.asarray(samples, dtype=float)
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must lie strictly in (0, 1), got {ci_level!r}")
    tail = 100.0 * (1.0 - ci_level) / 2.0
    low, high = np.percentile(samples, [tail, 100.0 - tail], axis=axis)
    return low, high


def interval_width(samples, ci_level: float = 0.95, axis: int = 0):
    """Width ``high - low`` of the percentile interval. See :func:`percentile_ci`."""
    low, high = percentile_ci(samples, ci_level=ci_level, axis=axis)
    return high - low


def half_width(samples, ci_level: float = 0.95, axis: int = 0):
    """Half-width ``(high - low) / 2`` of the percentile interval."""
    return interval_width(samples, ci_level=ci_level, axis=axis) / 2.0


def t_interval(values, ci_level: float = 0.95):
    """Student-*t* interval on the mean of a handful of per-unit estimates.

    This is the estimator the paper uses for speaker-level uncertainty. When the
    number of units is tiny -- the VoiceBank-DEMAND test split contains two
    speakers, four across all probing material -- a percentile bootstrap is
    degenerate: there are only a few distinct resamples, so the empirical
    percentiles cannot reach past the observed extremes and the interval is
    biased *inward*, exactly the wrong direction. A *t* interval on the per-unit
    values is the appropriate small-sample alternative.

    Parameters
    ----------
    values : array_like
        One value per sampling unit, e.g. one correlation per speaker.
    ci_level : float
        Coverage.

    Returns
    -------
    (mean, low, high, n) : tuple
        ``mean`` is the unweighted mean over units; the interval is sampling
        variation over those units.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n < 2:
        raise ValueError(f"t_interval needs at least 2 finite values, got {n}")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must lie strictly in (0, 1), got {ci_level!r}")
    try:
        from scipy.stats import t as _t
    except ImportError as exc:  # pragma: no cover - scipy is a hard dependency
        raise ImportError("t_interval requires SciPy (scipy.stats.t)") from exc
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    h = float(_t.ppf(0.5 + ci_level / 2.0, n - 1)) * sd / np.sqrt(n)
    return mean, mean - h, mean + h, n


# --------------------------------------------------------------------------- #
# Statistic helpers
# --------------------------------------------------------------------------- #
def weighted_mean(values, weights, axis: int = 0):
    """Weighted mean, safe for the zero-weight rows a cluster resample produces."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights sum to zero; cannot form a weighted mean")
    shape = [1] * values.ndim
    shape[axis] = weights.size
    return (values * weights.reshape(shape)).sum(axis=axis) / total


def weighted_pearson_r(x, y, weights):
    """Weighted Pearson correlation.

    Written in sufficient-statistic form so that a cluster resample is just a
    reweighting: this is what makes an utterance-cluster bootstrap of a
    CKA-versus-quality association cheap enough to run over every layer.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = w.sum()
    if n <= 0:
        raise ValueError("weights sum to zero; cannot form a correlation")
    sx = (w * x).sum()
    sy = (w * y).sum()
    sxx = (w * x * x).sum() - sx * sx / n
    syy = (w * y * y).sum() - sy * sy / n
    sxy = (w * x * y).sum() - sx * sy / n
    den = np.sqrt(max(sxx, 0.0) * max(syy, 0.0))
    return float(sxy / den) if den > 0 else float("nan")


def ols_alpha_beta(curve, x):
    """OLS intercept and slope of ``curve`` on a fixed grid ``x``.

    The paper's degradation-axis statistic: a layer's mean CKA curve is regressed
    on the degradation level (input SNR in dB, or C50 in dB) and the pair
    ``(alpha, beta)`` is bootstrapped. ``curve`` may carry leading axes (for
    example one row per layer); the regression runs along the last axis.
    """
    curve = np.asarray(curve, dtype=float)
    x = np.asarray(x, dtype=float)
    if curve.shape[-1] != x.shape[-1]:
        raise ValueError(f"curve last axis {curve.shape[-1]} != len(x) {x.shape[-1]}")
    xc = x - x.mean()
    sxx = (xc**2).sum()
    if sxx <= 0:
        raise ValueError("x has zero variance; slope is undefined")
    beta = (curve * xc).sum(axis=-1) / sxx
    alpha = curve.mean(axis=-1) - beta * x.mean()
    return alpha, beta


# --------------------------------------------------------------------------- #
# Grouping utilities
# --------------------------------------------------------------------------- #
def _dense_codes(labels, name: str) -> np.ndarray:
    """Map arbitrary labels to dense integer codes ``0 .. k-1``."""
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError(f"level {name!r} must be a 1-D array of labels, got shape {arr.shape}")
    _, inv = np.unique(arr, return_inverse=True)
    return np.ascontiguousarray(inv.ravel().astype(np.intp))


def _composite_codes(code_arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Dense codes for the tuple of several code arrays (a nesting path)."""
    if len(code_arrays) == 1:
        return code_arrays[0]
    stacked = np.stack(code_arrays, axis=1)
    _, inv = np.unique(stacked, axis=0, return_inverse=True)
    return np.ascontiguousarray(inv.ravel().astype(np.intp))


def _group_csr(keys: np.ndarray, n_keys: int):
    """CSR-style grouping: elements ``order[bounds[k]:bounds[k + 1]]`` have key ``k``."""
    order = np.argsort(keys, kind="stable").astype(np.intp)
    counts = np.bincount(keys, minlength=n_keys)
    bounds = np.concatenate([[0], np.cumsum(counts)]).astype(np.intp)
    return order, bounds


# --------------------------------------------------------------------------- #
# The design
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class BootstrapResult:
    """A hierarchical bootstrap interval, together with the design that made it.

    The interval is sampling variation over ``unit`` (and any nested levels in
    ``resampled_levels``), conditional on ``fixed_levels`` and ``strata``. It is
    a statement about a population only if ``unit`` is the population's unit.
    """

    estimate: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    replicates: np.ndarray
    unit: str
    n_boot: int
    ci_level: float
    seed: int | None
    resampled_levels: tuple[str, ...] = ()
    carried_levels: tuple[str, ...] = ()
    fixed_levels: tuple[str, ...] = ()
    strata: str | None = None
    crossed: tuple[str, ...] = ()
    n_units: int = 0

    @property
    def width(self):
        """Interval width ``ci_high - ci_low``."""
        return self.ci_high - self.ci_low

    @property
    def half_width(self):
        """Half the interval width."""
        return self.width / 2.0

    def excludes(self, value: float = 0.0) -> np.ndarray:
        """Whether the interval excludes ``value`` (default: zero)."""
        return (self.ci_low > value) | (self.ci_high < value)

    def describe(self) -> str:
        """One-line human-readable statement of the estimand."""
        fixed = ", ".join(self.fixed_levels) if self.fixed_levels else "none"
        crossed = ", ".join(self.crossed) if self.crossed else "none"
        carried = ", ".join(self.carried_levels) if self.carried_levels else "none"
        return (
            f"{self.ci_level:.0%} percentile interval from {self.n_boot} replicates; "
            f"sampling unit={self.unit} (n={self.n_units}); "
            f"nested levels resampled: {', '.join(self.resampled_levels)}; "
            f"nested levels carried whole: {carried}; "
            f"crossed margins resampled: {crossed}; "
            f"strata held fixed: {self.strata or 'none'}; "
            f"coarser levels held fixed (no variance propagated): {fixed}"
        )


@dataclass(eq=False)
class NestedDesign:
    """A compiled nesting specification that can be resampled repeatedly.

    Build once, draw many times. Construction validates the nesting and resolves
    the sampling unit; each draw then costs a few multinomials.

    Parameters
    ----------
    levels : Mapping[str, array_like]
        Ordered **outermost first**, one label per observation per level, e.g.
        ``{"speaker": spk_id, "utterance": utt_id}``. Labels must be globally
        unique within their level; a label appearing under two parents means the
        design is crossed rather than nested and raises.
    unit : str
        **Required, no default.** Name of the primary sampling unit; must be one
        of ``levels``. Levels above it are held fixed, it is resampled, and
        levels below it are carried along whole. See the module docstring for why
        this is not optional.
    resample_within : bool
        Also redraw the levels below ``unit`` within each drawn unit. Off by
        default: it adds a spurious within-cluster variance term (up to sqrt(2)
        inflation) and is not what the paper does. Turn it on only if the inner
        level is a fresh draw from a per-cluster superpopulation.
    strata : array_like, optional
        Per-observation stratum label. Strata are held fixed and units are
        resampled independently within each, preserving stratum sizes.
    crossed : Mapping[str, array_like], optional
        Per-observation labels for factors crossed with (not nested in) the unit.
        Each is resampled on its own margin and weights multiply across margins.
    strata_name : str, optional
        Label for ``strata`` used in reporting.
    acknowledge_pseudoreplication : bool
        Suppress :class:`PseudoReplicationWarning` when ``unit`` is nested inside
        a coarser level.
    """

    levels: Mapping[str, np.ndarray]
    unit: str
    strata: np.ndarray | None = None
    crossed: Mapping[str, np.ndarray] | None = None
    strata_name: str | None = None
    resample_within: bool = False
    acknowledge_pseudoreplication: bool = False

    n_obs: int = field(init=False)
    fixed_levels: tuple[str, ...] = field(init=False)
    resampled_levels: tuple[str, ...] = field(init=False)
    carried_levels: tuple[str, ...] = field(init=False)
    crossed_names: tuple[str, ...] = field(init=False)
    n_units: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.levels, Mapping) or not self.levels:
            raise TypeError("levels must be a non-empty ordered mapping, outermost level first")
        names = tuple(self.levels)
        if not isinstance(self.unit, str):
            raise TypeError(
                "unit must be a string naming one of the levels; it has no default because "
                "the correct sampling unit depends on the question being asked"
            )
        if self.unit not in names:
            raise ValueError(
                f"unit={self.unit!r} is not one of the declared levels {names}. "
                "State the sampling unit explicitly (e.g. 'utterance', 'speaker', 'noise_type')."
            )

        arrays = {}
        n_obs = None
        for name, labels in self.levels.items():
            codes = _dense_codes(labels, name)
            if n_obs is None:
                n_obs = codes.size
            elif codes.size != n_obs:
                raise ValueError(
                    f"level {name!r} has {codes.size} observations, expected {n_obs}"
                )
            arrays[name] = codes
        self.n_obs = int(n_obs)

        # A label reused under two parents means the design is crossed, not
        # nested. Catch it here rather than silently splitting the level into
        # (parent, child) clusters, which would change the estimand.
        for parent_name, child_name in zip(names, names[1:]):
            parent, child = arrays[parent_name], arrays[child_name]
            first_parent = np.full(int(child.max()) + 1, -1, dtype=np.intp)
            first_parent[child[::-1]] = parent[::-1]
            if not np.array_equal(first_parent[child], parent):
                raise ValueError(
                    f"level {child_name!r} is not nested inside {parent_name!r}: at least one "
                    f"{child_name} label occurs under more than one {parent_name}. If "
                    f"{parent_name!r} is crossed with {child_name!r} rather than containing it, "
                    "pass it as crossed=... or strata=... instead; if the labels are numbered "
                    "locally within each parent, make them globally unique first."
                )

        cut = names.index(self.unit)
        self.fixed_levels = names[:cut]
        self.resampled_levels = names[cut:] if self.resample_within else (self.unit,)
        self.carried_levels = () if self.resample_within else names[cut + 1 :]
        if self.fixed_levels and not self.acknowledge_pseudoreplication:
            warnings.warn(
                f"unit={self.unit!r} is nested inside {list(self.fixed_levels)}, which will be "
                f"held fixed. The interval will describe variation over {self.unit!r} only and "
                f"will be too narrow for any claim about {self.fixed_levels[0]!r}. "
                "Pass acknowledge_pseudoreplication=True if that is intended.",
                PseudoReplicationWarning,
                stacklevel=3,
            )

        # Stratum codes, held fixed. Folded into the unit key so that the same
        # utterance under two noise types is two distinct clusters.
        if self.strata is None:
            self._stratum_codes = None
        else:
            self._stratum_codes = _dense_codes(self.strata, self.strata_name or "strata")
            if self._stratum_codes.size != self.n_obs:
                raise ValueError("strata must have one label per observation")

        # Nesting paths: level d's key is the composite of levels unit..d,
        # prefixed by the stratum so strata partition the unit clusters.
        prefix = [] if self._stratum_codes is None else [self._stratum_codes]
        self._level_codes: list[np.ndarray] = []
        for depth, name in enumerate(self.resampled_levels):
            path = prefix + [arrays[n] for n in self.resampled_levels[: depth + 1]]
            self._level_codes.append(_composite_codes(path))
        self._level_sizes = [int(c.max()) + 1 for c in self._level_codes]
        self.n_units = self._level_sizes[0]

        # parent -> children, per depth
        self._children: list[tuple[np.ndarray, np.ndarray]] = []
        for depth in range(len(self._level_codes) - 1):
            parent_of_obs = self._level_codes[depth]
            child_of_obs = self._level_codes[depth + 1]
            n_child = self._level_sizes[depth + 1]
            parent_of_child = np.full(n_child, -1, dtype=np.intp)
            parent_of_child[child_of_obs] = parent_of_obs
            if not np.array_equal(parent_of_child[child_of_obs], parent_of_obs):
                raise ValueError(
                    f"level {self.resampled_levels[depth + 1]!r} is not nested inside "
                    f"{self.resampled_levels[depth]!r}: a child appears under two parents. "
                    "Use crossed= for factors that are crossed rather than nested."
                )
            self._children.append(_group_csr(parent_of_child, self._level_sizes[depth]))

        # Unit clusters grouped by stratum.
        if self._stratum_codes is None:
            self._unit_strata = [np.arange(self.n_units, dtype=np.intp)]
        else:
            root_codes = self._level_codes[0]
            stratum_of_root = np.full(self.n_units, -1, dtype=np.intp)
            stratum_of_root[root_codes] = self._stratum_codes
            n_strata = int(self._stratum_codes.max()) + 1
            order, bounds = _group_csr(stratum_of_root, n_strata)
            self._unit_strata = [
                order[bounds[s] : bounds[s + 1]] for s in range(n_strata) if bounds[s + 1] > bounds[s]
            ]

        # Crossed margins.
        self._crossed_codes: list[tuple[str, np.ndarray, int]] = []
        overlap = set(self.crossed or ()) & set(names)
        if overlap:
            raise ValueError(f"crossed factors {sorted(overlap)} are also declared as nested levels")
        for name, labels in (self.crossed or {}).items():
            codes = _dense_codes(labels, name)
            if codes.size != self.n_obs:
                raise ValueError(f"crossed factor {name!r} must have one label per observation")
            self._crossed_codes.append((name, codes, int(codes.max()) + 1))
        self.crossed_names = tuple(n for n, _, _ in self._crossed_codes)

        self._innermost_codes = self._level_codes[-1]

    # -- drawing ----------------------------------------------------------- #
    def weights(self, rng: np.random.Generator) -> np.ndarray:
        """One hierarchical resample, expressed as per-observation multiplicities.

        Returns a float array of length ``n_obs``. An observation's weight is the
        number of times its innermost cluster was drawn, multiplied across any
        crossed margins. Weights are integer-valued; only their ratios matter.
        """
        counts = np.zeros(self.n_units, dtype=np.float64)
        for members in self._unit_strata:
            m = members.size
            counts[members] = rng.multinomial(m, np.full(m, 1.0 / m))

        for depth, (order, bounds) in enumerate(self._children):
            child_counts = np.zeros(self._level_sizes[depth + 1], dtype=np.float64)
            for parent in np.flatnonzero(counts):
                kids = order[bounds[parent] : bounds[parent + 1]]
                k = kids.size
                if k == 0:
                    continue
                draws = int(counts[parent]) * k
                child_counts[kids] = rng.multinomial(draws, np.full(k, 1.0 / k))
            counts = child_counts

        w = counts[self._innermost_codes]
        for _, codes, n_cat in self._crossed_codes:
            margin = rng.multinomial(n_cat, np.full(n_cat, 1.0 / n_cat)).astype(np.float64)
            w = w * margin[codes]
        return w

    def indices(self, rng: np.random.Generator) -> np.ndarray:
        """One hierarchical resample as an explicit array of observation indices.

        Equivalent to :meth:`weights` but materialized: an observation whose
        cluster was drawn twice appears twice. Use this for statistics that
        cannot accept weights (medians, rank statistics); it allocates a full
        resampled dataset per replicate.
        """
        w = self.weights(rng)
        return np.repeat(np.arange(self.n_obs, dtype=np.intp), w.astype(np.intp))


# --------------------------------------------------------------------------- #
# Functional entry points
# --------------------------------------------------------------------------- #
def hierarchical_resample_weights(
    levels: Mapping[str, np.ndarray],
    *,
    unit: str,
    rng: np.random.Generator | int | None = None,
    strata=None,
    crossed: Mapping[str, np.ndarray] | None = None,
    strata_name: str | None = None,
    resample_within: bool = False,
    acknowledge_pseudoreplication: bool = False,
) -> np.ndarray:
    """Draw one hierarchical resample as per-observation multiplicities.

    ``unit`` is required and has no default; see the module docstring.
    """
    design = NestedDesign(
        levels=levels,
        unit=unit,
        strata=strata,
        crossed=crossed,
        strata_name=strata_name,
        resample_within=resample_within,
        acknowledge_pseudoreplication=acknowledge_pseudoreplication,
    )
    return design.weights(np.random.default_rng(rng))


def hierarchical_resample_indices(
    levels: Mapping[str, np.ndarray],
    *,
    unit: str,
    rng: np.random.Generator | int | None = None,
    strata=None,
    crossed: Mapping[str, np.ndarray] | None = None,
    strata_name: str | None = None,
    resample_within: bool = False,
    acknowledge_pseudoreplication: bool = False,
) -> np.ndarray:
    """Draw one hierarchical resample as an array of observation indices.

    ``unit`` is required and has no default; see the module docstring.
    """
    design = NestedDesign(
        levels=levels,
        unit=unit,
        strata=strata,
        crossed=crossed,
        strata_name=strata_name,
        resample_within=resample_within,
        acknowledge_pseudoreplication=acknowledge_pseudoreplication,
    )
    return design.indices(np.random.default_rng(rng))


def hierarchical_bootstrap(
    statistic: Callable[[np.ndarray, np.ndarray], np.ndarray],
    levels: Mapping[str, np.ndarray],
    *,
    unit: str,
    n_boot: int = PAPER_N_BOOT,
    ci_level: float = 0.95,
    seed: int | None = PAPER_SEED,
    strata=None,
    crossed: Mapping[str, np.ndarray] | None = None,
    strata_name: str | None = None,
    resample_within: bool = False,
    expand_indices: bool = False,
    acknowledge_pseudoreplication: bool = False,
) -> BootstrapResult:
    """Carry a user-supplied statistic through a hierarchical cluster bootstrap.

    Parameters
    ----------
    statistic : callable
        Called as ``statistic(indices, weights)`` and must return a float or an
        array (for example one value per layer), the same shape every call.
        ``indices`` selects observations and ``weights`` gives their
        multiplicities. In the default weight mode ``indices`` is
        ``arange(n_obs)`` and the resample lives entirely in ``weights``; with
        ``expand_indices=True`` the resample lives in ``indices`` and ``weights``
        is all ones. The two are mathematically identical for any statistic that
        respects weights, so a statistic written for one mode works in the other.
    levels : Mapping[str, array_like]
        Nesting specification, ordered outermost first, one label per
        observation. See :class:`NestedDesign`.
    unit : str
        **Required, no default.** The sampling unit. This determines what the
        interval is an interval *over*: the returned interval is sampling
        variation over ``unit`` conditional on the corpus, and is a statement
        about a speaker population only if ``unit`` is the speaker.
    n_boot : int
        Number of replicates. The paper uses 1000.
    ci_level : float
        Coverage of the equal-tailed percentile interval.
    seed : int or None
        Seed for ``numpy.random.default_rng``. Fixing it makes the interval
        reproducible; the paper uses 0.
    strata, crossed, strata_name, resample_within, acknowledge_pseudoreplication
        See :class:`NestedDesign`.
    expand_indices : bool
        Materialize duplicated indices instead of passing weights.

    Returns
    -------
    BootstrapResult
        Point estimate, interval, all replicates, and the design provenance.
        ``result.describe()`` states the estimand in one line.
    """
    if n_boot < 2:
        raise ValueError(f"n_boot must be at least 2, got {n_boot}")

    design = NestedDesign(
        levels=levels,
        unit=unit,
        strata=strata,
        crossed=crossed,
        strata_name=strata_name,
        resample_within=resample_within,
        acknowledge_pseudoreplication=acknowledge_pseudoreplication,
    )
    rng = np.random.default_rng(seed)
    all_idx = np.arange(design.n_obs, dtype=np.intp)

    estimate = np.asarray(statistic(all_idx, np.ones(design.n_obs)), dtype=float)

    replicates = np.empty((n_boot,) + estimate.shape, dtype=float)
    for b in range(n_boot):
        if expand_indices:
            idx = design.indices(rng)
            value = statistic(idx, np.ones(idx.size))
        else:
            w = design.weights(rng)
            value = statistic(all_idx, w)
        replicates[b] = np.asarray(value, dtype=float)

    low, high = percentile_ci(replicates, ci_level=ci_level, axis=0)
    return BootstrapResult(
        estimate=estimate,
        ci_low=np.asarray(low),
        ci_high=np.asarray(high),
        replicates=replicates,
        unit=unit,
        n_boot=int(n_boot),
        ci_level=float(ci_level),
        seed=seed,
        resampled_levels=design.resampled_levels,
        carried_levels=design.carried_levels,
        fixed_levels=design.fixed_levels,
        strata=strata_name if strata is not None else None,
        crossed=design.crossed_names,
        n_units=design.n_units,
    )
