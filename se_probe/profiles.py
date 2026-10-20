"""Per-layer robustness/sensitivity profiles and their summary statistics.

This module is the reference implementation of the paper's per-layer analysis:

1. **Mean degradation-level curve.** For each probed layer, CKA is averaged over the
   probing material (utterances x noise types, or utterances x RIRs) at each
   degradation level, giving one curve per layer (:func:`mean_level_curves`).
2. **Linearisation.** Each curve is fitted by OLS, ``CKA = alpha + beta * s``, with
   ``s`` in dB. ``beta`` is the *sensitivity* and ``alpha`` (the fit at 0 dB) the
   *robustness* (:func:`fit_curves`, :func:`fit_profile`).
3. **Endpoint and area statistics.** ``c_low`` / ``c_high`` are the empirical mean CKA
   at the most severe / mildest swept level, ``rng = c_high - c_low``, and ``auc`` is
   the normalised trapezoidal area under the curve divided by the grid span.
4. **Saturation spread.** The across-layer correlation ``r(alpha, beta)`` is partly
   forced by CKA saturating at the clean end. :func:`tradeoff_summary` reports it
   together with the saturation spread ``f = sd(c_high) / sd(alpha)``, the leading
   singular-component share ``PC1`` of the layer x level matrix, and the floor
   ``sqrt(1 - f**2)`` on ``|r|`` that the identity permits.
5. **Bootstrap intervals.** Utterance-cluster (:func:`utterance_bootstrap`) and
   two-way utterance x noise/RIR (:func:`two_way_cluster_bootstrap`) percentile
   intervals on every per-layer statistic. The two-way variant draws exactly as the
   cluster script behind the published intervals did (one generator, ``seed=0``,
   utterances then clusters, layers in sorted order); a bit-identical replay needs
   the full 824-utterance table and every hooked layer threaded through the stream
   (``stream_layers``), as ``scripts/analyze_snr_profiles.py --bootstrap`` does.

All functions are NumPy/pandas only and operate on the long-format result tables
(columns ``layer``, ``clean_idx``, ``snr`` or ``target_c50``, ``CKA``, ...).
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from se_probe.layers import (
    MUSE_STAGES,
    muse_stage_of,
    probed_layers,
    short_labels,
)

__all__ = [
    "PAPER_SEED_HIERARCHICAL",
    "PAPER_SEED_UTTERANCE",
    "PAPER_SEED_FT_PROFILE",
    "PAPER_SEED_CROSS_TASK",
    "PAPER_N_BOOT",
    "mean_level_curves",
    "fit_curves",
    "fit_profile",
    "build_cube",
    "utterance_bootstrap",
    "two_way_cluster_bootstrap",
    "pc_shares",
    "saturation_spread",
    "identity_r",
    "tradeoff_summary",
    "profile_agreement",
    "layer_resample_ci",
    "noise_set_independence",
    "random_init_summary",
    "emergence_convergence",
    "finetune_shift",
    "bottleneck_gap",
]

_trapz = getattr(np, "trapezoid", None) or np.trapz

#: Seed of the hierarchical utterance x noise bootstrap behind the published CIs.
PAPER_SEED_HIERARCHICAL = 0
#: Seed of the utterance bootstrap on the reverberation axis (AUC, c_low, alpha, beta).
PAPER_SEED_UTTERANCE = 20260802
#: Seed of the utterance bootstrap behind the fine-tuned C50 profile figure.
PAPER_SEED_FT_PROFILE = 20260806
#: Seed of the layer-resample intervals of the cross-task agreement.
PAPER_SEED_CROSS_TASK = 20260810
PAPER_N_BOOT = 1000


# --------------------------------------------------------------------------- curves
def _level_col(df: pd.DataFrame, level_col: Optional[str]) -> str:
    if level_col is not None:
        return level_col
    for c in ("snr", "target_c50"):
        if c in df.columns:
            return c
    raise KeyError("no degradation-level column ('snr' or 'target_c50') found")


def mean_level_curves(
    df: pd.DataFrame,
    model: Optional[str] = None,
    layers: Optional[Sequence[str]] = None,
    level_col: Optional[str] = None,
    unit_col: Optional[str] = "clean_idx",
    utterance_first: bool = True,
) -> pd.DataFrame:
    """Mean CKA per (layer, level), as a ``layers x levels`` DataFrame.

    Parameters
    ----------
    df : long-format table with ``layer``, ``CKA`` and a level column.
    model : if given, restrict to that model's probed layers in depth order.
    layers : explicit layer list (overrides ``model``). Default: all layers present.
    level_col : ``"snr"`` or ``"target_c50"``; autodetected when ``None``.
    unit_col, utterance_first : when ``True`` and ``unit_col`` is present, an
        utterance's rows at a level (its RIRs or noise types) are averaged first so
        every utterance weighs equally. On a balanced design this equals the plain
        row mean the noise-axis tables use.
    """
    level_col = _level_col(df, level_col)
    if layers is None:
        if model is None:
            raise ValueError("mean_level_curves needs `model` (probed layers in depth order) "
                             "or an explicit `layers` list; alphabetical order is not depth order")
        layers = probed_layers(model, df["layer"].unique())
    sub = df[df["layer"].isin(list(layers))]
    if utterance_first and unit_col is not None and unit_col in sub.columns:
        per_unit = sub.groupby(["layer", unit_col, level_col], observed=True)["CKA"].mean()
        mc = per_unit.groupby(level=["layer", level_col]).mean()
    else:
        mc = sub.groupby(["layer", level_col], observed=True)["CKA"].mean()
    curves = mc.unstack(level_col).reindex(list(layers))
    curves.columns = curves.columns.astype(float)
    return curves.sort_index(axis=1)


def fit_curves(curves, levels=None) -> Dict[str, np.ndarray]:
    """OLS, endpoint and area statistics of per-layer mean curves.

    ``curves`` is ``[n_layers, n_levels]`` (array or DataFrame with level columns).
    Returns a dict of arrays: ``alpha, beta, r2, c_low, c_high, rng, auc``.
    """
    if isinstance(curves, pd.DataFrame):
        if levels is None:
            levels = curves.columns.to_numpy(dtype=float)
        curves = curves.to_numpy(dtype=float)
    curves = np.asarray(curves, dtype=float)
    x = np.asarray(levels, dtype=float)
    if curves.shape[-1] != x.size:
        raise ValueError(f"curves have {curves.shape[-1]} levels, got {x.size} level values")
    order = np.argsort(x)
    x, curves = x[order], curves[..., order]
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    ym = curves.mean(axis=-1)
    beta = ((curves - ym[..., None]) * (x - xm)).sum(axis=-1) / sxx
    alpha = ym - beta * xm
    pred = alpha[..., None] + beta[..., None] * x
    ss_res = ((curves - pred) ** 2).sum(axis=-1)
    ss_tot = ((curves - ym[..., None]) ** 2).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    c_low = curves[..., 0]
    c_high = curves[..., -1]
    auc = _trapz(curves, x, axis=-1) / (x[-1] - x[0])
    return dict(alpha=alpha, beta=beta, r2=r2, c_low=c_low, c_high=c_high,
                rng=c_high - c_low, auc=auc)


def fit_profile(
    df: pd.DataFrame,
    model: str,
    level_col: Optional[str] = None,
    layers: Optional[Sequence[str]] = None,
    **curve_kwargs,
) -> pd.DataFrame:
    """Per-layer profile table (``layer, depth, label, alpha, beta, r2, c_low, c_high,
    rng, auc``) for one model, in depth order."""
    curves = mean_level_curves(df, model=model, layers=layers, level_col=level_col, **curve_kwargs)
    f = fit_curves(curves)
    layers = list(curves.index)
    out = pd.DataFrame({"layer": layers, "depth": np.arange(len(layers)),
                        "label": short_labels(model, layers), **f})
    return out


# ----------------------------------------------------------------------------- cube
def build_cube(
    df: pd.DataFrame,
    layers: Sequence[str],
    level_col: Optional[str] = None,
    unit_col: str = "clean_idx",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``[n_layer, n_unit, n_level]`` cube of per-unit mean CKA (collapsing the
    unit's RIRs / noise types), plus the sorted unit labels and level grid."""
    level_col = _level_col(df, level_col)
    sub = df[df["layer"].isin(list(layers))]
    g = sub.groupby(["layer", unit_col, level_col], observed=True)["CKA"].mean()
    units = np.sort(sub[unit_col].unique())
    levels = np.sort(sub[level_col].unique().astype(float))
    li = {L: i for i, L in enumerate(layers)}
    ui = {u: i for i, u in enumerate(units)}
    vi = {v: i for i, v in enumerate(levels)}
    cube = np.full((len(layers), len(units), len(levels)), np.nan)
    idx = g.index
    cube[[li[v] for v in idx.get_level_values(0)],
         [ui[v] for v in idx.get_level_values(1)],
         [vi[float(v)] for v in idx.get_level_values(2)]] = g.to_numpy()
    return cube, units, levels


def utterance_bootstrap(
    cube: np.ndarray,
    levels: np.ndarray,
    n_boot: int = PAPER_N_BOOT,
    seed: int = PAPER_SEED_UTTERANCE,
    stats_keys: Sequence[str] = ("alpha", "beta", "auc", "c_low", "c_high"),
    ci: float = 0.95,
) -> Dict[str, np.ndarray]:
    """Percentile intervals over whole-utterance resamples of a ``build_cube`` cube.

    Returns ``{key: estimate, key_lo, key_hi, key_se, ...}`` plus ``replicates``
    (``{key: [n_boot, n_layer]}``). Each replicate draws ``n_unit`` utterances with
    replacement, averages the resampled cube over units and recomputes the
    statistics on the resulting mean curves.
    """
    rng = np.random.default_rng(seed)
    n_u = cube.shape[1]
    point = fit_curves(np.nanmean(cube, axis=1), levels)
    draws = {k: np.empty((n_boot, cube.shape[0])) for k in stats_keys}
    for b in range(n_boot):
        sel = rng.integers(0, n_u, n_u)
        f = fit_curves(np.nanmean(cube[:, sel, :], axis=1), levels)
        for k in stats_keys:
            draws[k][b] = f[k]
    tail = 100.0 * (1.0 - ci) / 2.0
    out: Dict[str, np.ndarray] = {}
    for k in stats_keys:
        lo, hi = np.percentile(draws[k], [tail, 100.0 - tail], axis=0)
        out[k] = point[k]
        out[f"{k}_lo"], out[f"{k}_hi"] = lo, hi
        out[f"{k}_se"] = draws[k].std(axis=0, ddof=1)
    out["replicates"] = draws
    return out


def two_way_cluster_bootstrap(
    df: pd.DataFrame,
    layers: Sequence[str],
    level_col: Optional[str] = None,
    cluster_col: str = "noise_name",
    unit_col: str = "clean_idx",
    n_boot: int = PAPER_N_BOOT,
    seed: int = PAPER_SEED_HIERARCHICAL,
    rng: Optional[np.random.Generator] = None,
    ci: float = 0.95,
    stream_layers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Hierarchical utterance x cluster bootstrap of ``(alpha, beta)`` per layer.

    Each replicate draws the utterances and the clusters (noise types on the noise
    axis, RIRs on the reverberation axis) with replacement, forms the mean level
    curve of the resampled ``utterance x cluster`` grid and refits the line. Cells
    of the grid that the design does not contain (an utterance meets 5 of the 88
    RIRs) are simply absent from the weighted mean, in the replicates exactly as in
    the point estimate; cells with several rows (duplicate shards, a cluster column
    coarser than the row grain) are averaged first.

    The draws follow the cluster script behind the paper's intervals: one generator
    (``seed=0``, or ``rng`` to continue a stream across models), utterances drawn
    with ``rng.choice`` before clusters, layers visited in ``sorted()`` order. That
    script threaded *every hooked layer* through the stream, so a bit-identical
    replay of a published interval needs ``stream_layers`` = all hooked layers of
    the model (``df["layer"].unique()``); the rows returned are still only
    ``layers``, in the order given (pass them in depth order), with a ``depth``
    column. Without ``stream_layers`` the intervals are statistically equivalent
    but not bit-identical to the published ones.
    """
    level_col = _level_col(df, level_col)
    rng = np.random.default_rng(seed) if rng is None else rng
    keep = list(layers)
    visit = sorted(set(keep) | set(stream_layers or []))
    sub = df[df["layer"].isin(visit)]
    utts = np.asarray(sorted(sub[unit_col].unique()))
    clusters = np.asarray(sorted(sub[cluster_col].unique()))
    levels = np.sort(sub[level_col].unique().astype(float))
    ui = {u: i for i, u in enumerate(utts)}
    ci_ = {c: i for i, c in enumerate(clusters)}
    vi = {v: i for i, v in enumerate(levels)}
    out: Dict[str, dict] = {}
    tail = 100.0 * (1.0 - ci) / 2.0
    for layer in visit:
        d = (sub[sub["layer"] == layer]
             .assign(_lvl=lambda t: t[level_col].astype(float))
             .groupby([cluster_col, unit_col, "_lvl"], observed=True)["CKA"].mean().reset_index())
        cube = np.full((len(clusters), len(utts), len(levels)), np.nan)
        cube[d[cluster_col].map(ci_).to_numpy(), d[unit_col].map(ui).to_numpy(),
             d["_lvl"].map(vi).to_numpy()] = d["CKA"].to_numpy()
        present = np.isfinite(cube)
        filled = np.where(present, cube, 0.0)
        obs = np.nanmean(cube, axis=(0, 1))
        pt = fit_curves(obs[None, :], levels)
        A = np.empty(n_boot)
        B = np.empty(n_boot)
        for r in range(n_boot):
            u_samp = rng.choice(utts, size=len(utts), replace=True)
            n_samp = rng.choice(clusters, size=len(clusters), replace=True)
            wu = np.bincount(np.searchsorted(utts, u_samp), minlength=len(utts)) / len(utts)
            wn = np.bincount(np.searchsorted(clusters, n_samp), minlength=len(clusters)) / len(clusters)
            num = np.einsum("c,u,cul->l", wn, wu, filled)
            den = np.einsum("c,u,cul->l", wn, wu, present)
            mc = num / den
            f = fit_curves(mc[None, :], levels)
            A[r], B[r] = f["alpha"][0], f["beta"][0]
        if layer not in keep:
            continue
        alo, ahi = np.percentile(A, [tail, 100 - tail])
        blo, bhi = np.percentile(B, [tail, 100 - tail])
        out[layer] = dict(layer=layer, depth=keep.index(layer), alpha=float(pt["alpha"][0]),
                          beta=float(pt["beta"][0]), alpha_ci_lo=alo, alpha_ci_hi=ahi,
                          beta_ci_lo=blo, beta_ci_hi=bhi, c_low=float(obs[0]),
                          c_high=float(obs[-1]), n_boot=n_boot, seed=seed)
    return pd.DataFrame([out[layer] for layer in keep])


# --------------------------------------------------------------------- summaries
def pc_shares(curves) -> Tuple[float, float]:
    """Variance shares (%) of the first two singular components of the layer x level
    matrix centred by the mean **over layers** at each level (``M - M.mean(axis=0)``).

    Centering each layer's own curve instead is a different matrix (PC1 98.7 rather
    than 76.3 for Demucs under noise); the across-layer centering is the paper's.
    """
    M = curves.to_numpy(dtype=float) if isinstance(curves, pd.DataFrame) else np.asarray(curves, float)
    c = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(c, compute_uv=False)
    v = s ** 2
    v = v / v.sum() * 100.0
    return float(v[0]), float(v[1]) if v.size > 1 else float("nan")


def saturation_spread(alpha, c_high) -> float:
    """``f = sd(c_high) / sd(alpha)`` (sample standard deviations)."""
    return float(np.std(np.asarray(c_high, float), ddof=1) / np.std(np.asarray(alpha, float), ddof=1))


def identity_r(r_c: float, f: float) -> float:
    """The across-layer ``r(alpha, beta)`` forced by perfectly linear curves:
    ``(r_c f - 1) / sqrt(f**2 - 2 r_c f + 1)`` with ``r_c = r(alpha, c_high)``."""
    return float((r_c * f - 1.0) / np.sqrt(f * f - 2.0 * r_c * f + 1.0))


def _spearman(a, b) -> float:
    return float(stats.spearmanr(a, b).statistic)


def tradeoff_summary(curves: pd.DataFrame, fits: Optional[Mapping[str, np.ndarray]] = None,
                     depth: Optional[Sequence[int]] = None) -> Dict[str, float]:
    """The paper's Table I row plus the depth-trend statistics for one model/axis.

    Parameters
    ----------
    curves : ``layers x levels`` mean-curve DataFrame (:func:`mean_level_curves`).
    fits : output of :func:`fit_curves` on ``curves`` (recomputed when ``None``).
    depth : depth index per row (default ``0..n-1``, i.e. ``curves`` is in depth order).
    """
    f = fits if fits is not None else fit_curves(curves)
    levels = curves.columns.to_numpy(dtype=float)
    layers = list(curves.index)
    d = np.asarray(depth if depth is not None else np.arange(len(layers)), float)
    a, b = f["alpha"], f["beta"]
    pc1, pc2 = pc_shares(curves)
    spread = saturation_spread(a, f["c_high"])
    r_c = float(np.corrcoef(a, f["c_high"])[0, 1])
    s_high = float(levels.max())
    return dict(
        n_layers=len(layers), n_levels=int(levels.size),
        r_alpha_beta=float(np.corrcoef(a, b)[0, 1]),
        rho_alpha_beta=_spearman(a, b),
        sat_freedom=spread, r_alpha_chigh=r_c,
        identity_r=float(np.corrcoef(b, (f["c_high"] - a) / s_high)[0, 1]),
        predicted_r=identity_r(r_c, spread),
        floor_abs_r=float(np.sqrt(max(0.0, 1.0 - spread ** 2))),
        pc1=pc1, pc2=pc2,
        r_auc_alpha=float(np.corrcoef(f["auc"], a)[0, 1]),
        r_auc_clow=float(np.corrcoef(f["auc"], f["c_low"])[0, 1]),
        r2_min=float(np.nanmin(f["r2"])), r2_median=float(np.nanmedian(f["r2"])),
        c_low_min=float(f["c_low"].min()), c_low_max=float(f["c_low"].max()),
        c_high_min=float(f["c_high"].min()), c_high_max=float(f["c_high"].max()),
        auc_mean=float(f["auc"].mean()),
        peak_auc_layer=layers[int(np.argmax(f["auc"]))],
        peak_beta_layer=layers[int(np.argmax(b))],
        rho_depth_beta=_spearman(d, b), rho_depth_alpha=_spearman(d, a),
        rho_depth_range=_spearman(d, f["rng"]), rho_depth_auc=_spearman(d, f["auc"]),
    )


# ------------------------------------------------------------ profile agreement
def profile_agreement(a, b) -> Dict[str, float]:
    """Pearson and Spearman agreement of two per-layer profiles (matched layers)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    pr = stats.pearsonr(a[m], b[m])
    sr = stats.spearmanr(a[m], b[m])
    return dict(pearson_r=float(pr.statistic), pearson_p=float(pr.pvalue),
                spearman_rho=float(sr.statistic), spearman_p=float(sr.pvalue), n=int(m.sum()))


def layer_resample_ci(a, b, n_boot: int = 20000, seed: int = PAPER_SEED_CROSS_TASK,
                      ci: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """Percentile intervals on the profile agreement by resampling *layers*.

    Layers are drawn with replacement and the paired values travel together; draws
    with fewer than three distinct values on either side are discarded. Treating the
    layer as the sampling unit is the right unit for a question about profile shape.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = a.size
    rng = np.random.default_rng(seed)
    rs, rhos = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        x, y = a[idx], b[idx]
        if np.unique(x).size < 3 or np.unique(y).size < 3:
            continue
        rs.append(np.corrcoef(x, y)[0, 1])
        rhos.append(stats.spearmanr(x, y).statistic)
    tail = 100.0 * (1.0 - ci) / 2.0
    return dict(pearson=tuple(np.percentile(rs, [tail, 100 - tail])),
                spearman=tuple(np.percentile(rhos, [tail, 100 - tail])),
                n_valid=len(rs))


def noise_set_independence(per_noise_beta: pd.DataFrame, set_a: Iterable[str],
                           set_b: Iterable[str]) -> Dict[str, float]:
    """Agreement of the per-layer slope profile fitted on two disjoint noise sets.

    ``per_noise_beta`` is ``layers x noises`` (one slope per layer per noise). The
    profile of a set is the mean of its per-noise slopes, which equals fitting the
    mean curve over that set.
    """
    bh = per_noise_beta[list(set_a)].mean(axis=1).to_numpy()
    bi = per_noise_beta[list(set_b)].mean(axis=1).to_numpy()
    out = profile_agreement(bh, bi)
    out.update(mean_beta_a=float(bh.mean()), mean_beta_b=float(bi.mean()),
               peak_layer_a=str(per_noise_beta.index[int(np.argmax(bh))]),
               peak_layer_b=str(per_noise_beta.index[int(np.argmax(bi))]))
    return out


# ----------------------------------------------------------- random init / epochs
def random_init_summary(trained_beta, seed_betas: Mapping[str, Sequence[float]],
                        cka_min: Optional[float] = None) -> Dict[str, object]:
    """Random-initialisation control statistics (paper Sec. "Origin of the Profile").

    ``trained_beta`` is the trained model's per-layer slope in depth order and
    ``seed_betas`` maps each seed to its per-layer slope on the same grid.
    """
    tb = np.asarray(trained_beta, float)
    depth = np.arange(tb.size)
    per_seed = {}
    for s, b in seed_betas.items():
        b = np.asarray(b, float)
        sr = stats.spearmanr(depth, b)
        pr = stats.pearsonr(depth, b)
        per_seed[str(s)] = dict(max_beta=float(b.max()), peak_depth=int(np.argmax(b)),
                                rho_depth_beta=float(sr.statistic), rho_p=float(sr.pvalue),
                                r_depth_beta=float(pr.statistic), r_p=float(pr.pvalue))
    largest = max(v["max_beta"] for v in per_seed.values())
    mean_b = np.mean([np.asarray(b, float) for b in seed_betas.values()], axis=0)
    return dict(
        trained_peak_beta=float(tb.max()), trained_peak_depth=int(np.argmax(tb)),
        factor_trained_over_largest_seed=float(tb.max() / largest),
        per_seed=per_seed, cka_min=cka_min,
        profile_corr_untrained_vs_trained=profile_agreement(mean_b, tb),
    )


def emergence_convergence(profiles: Mapping[object, pd.DataFrame], final_key,
                          pretrained_key=None,
                          sensitive_stages=("decoder_level2", "decoder_level1", "mag_refinement"),
                          reference_stage="encoder_level1") -> Dict[str, object]:
    """Convergence of per-epoch MUSE profiles to the final checkpoint and the
    concentration of the fine-tuning change in the sensitive stages.

    ``profiles`` maps an epoch key to a :func:`fit_profile` table (same layers, depth
    order). Returns a convergence table (Pearson r of ``beta`` and ``alpha`` with the
    final profile per epoch) and, when ``pretrained_key`` is given, the ratio of the
    mean absolute change in ``beta`` / ``alpha`` between ``sensitive_stages`` and
    ``reference_stage``.
    """
    final = profiles[final_key].set_index("layer")
    rows = []
    for k, p in profiles.items():
        p = p.set_index("layer").loc[final.index]
        rows.append(dict(epoch=k,
                         beta_r_vs_final=float(np.corrcoef(p["beta"], final["beta"])[0, 1]),
                         alpha_r_vs_final=float(np.corrcoef(p["alpha"], final["alpha"])[0, 1]),
                         rms_dbeta_vs_final=float(np.sqrt(((p["beta"] - final["beta"]) ** 2).mean()))))
    out: Dict[str, object] = {"convergence": pd.DataFrame(rows)}
    if pretrained_key is not None:
        pre = profiles[pretrained_key].set_index("layer").loc[final.index]
        stage = np.array([muse_stage_of(L) for L in final.index])
        db = np.abs(pre["beta"].to_numpy() - final["beta"].to_numpy())
        da = np.abs(pre["alpha"].to_numpy() - final["alpha"].to_numpy())
        sens = np.isin(stage, list(sensitive_stages))
        ref = stage == reference_stage
        out["concentration"] = dict(
            ratio_dbeta_sensitive_over_reference=float(db[sens].mean() / db[ref].mean()),
            ratio_dalpha_sensitive_over_reference=float(da[sens].mean() / da[ref].mean()),
            mean_abs_dbeta_sensitive=float(db[sens].mean()), mean_abs_dbeta_reference=float(db[ref].mean()),
            mean_abs_dalpha_sensitive=float(da[sens].mean()), mean_abs_dalpha_reference=float(da[ref].mean()),
            pretrained_beta_r_vs_final=float(np.corrcoef(pre["beta"], final["beta"])[0, 1]),
            pretrained_alpha_r_vs_final=float(np.corrcoef(pre["alpha"], final["alpha"])[0, 1]),
        )
    return out


def finetune_shift(pre: pd.DataFrame, post: pd.DataFrame, keys=("beta", "alpha", "auc")) -> Dict[str, float]:
    """Before/after fine-tuning agreement of per-layer profiles on matched layers."""
    pre = pre.set_index("layer")
    post = post.set_index("layer").loc[pre.index]
    out: Dict[str, float] = {"n_layers": int(len(pre))}
    for k in keys:
        if k not in pre.columns or k not in post.columns:
            continue
        ag = profile_agreement(pre[k], post[k])
        d = post[k].to_numpy() - pre[k].to_numpy()
        out[f"{k}_r"] = ag["pearson_r"]
        out[f"{k}_rho"] = ag["spearman_rho"]
        out[f"{k}_mean_abs_delta"] = float(np.abs(d).mean())
        out[f"{k}_max_abs_delta"] = float(np.abs(d).max())
        out[f"{k}_max_delta_layer"] = str(pre.index[int(np.abs(d).argmax())])
    return out


def bottleneck_gap(replicates: np.ndarray, point: np.ndarray, layers: Sequence[str],
                   target: str = "lstm", ci: float = 0.95) -> Dict[str, object]:
    """Gap between ``target``'s statistic and the runner-up layer, with its bootstrap
    interval, ``P(gap > 0)`` and ``P(arg-max == target)`` over ``replicates``
    (``[n_boot, n_layer]``, e.g. ``utterance_bootstrap(...)['replicates']['auc']``)."""
    li = list(layers).index(target)
    others = [i for i in range(len(layers)) if i != li]
    runner = others[int(np.argmax(np.asarray(point)[others]))]
    gap = replicates[:, li] - replicates[:, runner]   # same estimand as the point gap
    tail = 100.0 * (1.0 - ci) / 2.0
    return dict(
        target=target, target_value=float(point[li]), runner_up=str(layers[runner]),
        runner_value=float(point[runner]), gap=float(point[li] - point[runner]),
        gap_lo=float(np.percentile(gap, tail)), gap_hi=float(np.percentile(gap, 100 - tail)),
        p_gap_gt0=float((gap > 0).mean()),
        p_peak_is_target=float((replicates.argmax(axis=1) == li).mean()),
    )


def stage_of_layers(layers: Sequence[str]) -> np.ndarray:
    """MUSE stage name per layer (helper for stage-level aggregation)."""
    return np.array([muse_stage_of(L) for L in layers])


MUSE_STAGE_ORDER = list(MUSE_STAGES)
