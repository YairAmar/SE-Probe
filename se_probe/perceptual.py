"""Quality-association analysis: layer-wise CKA against output-quality improvement.

For each utterance and metric ``M`` the improvement is ``dM = M(enhanced) - M(degraded)``.
Both CKA and ``dM`` are driven by the degradation level, so the association is
computed on **within-group centred** quantities: CKA is centred per
``(layer, level)`` and ``dM`` per ``(layer, level)`` as well (a metric does not
depend on the layer, so this equals centring per level). The statistic is a per-layer
Pearson correlation of the centred quantities.

The sampling unit matters. Utterances of one speaker are not independent, and the
VoiceBank-DEMAND test split holds two speakers, so :func:`speaker_level_analysis`
reports, per layer:

* the pooled utterance-level correlation with an **utterance-cluster bootstrap**
  interval (the pseudo-replicated interval);
* one correlation **per speaker**, recentred within the speaker; and
* the **speaker-level** mean with a Student-*t* interval over speakers.

Only an association on which the speakers agree in sign is reported as shared.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t as tdist

from se_probe.layers import SPEAKER_RANGES, probed_layers, short_labels

__all__ = [
    "METRICS",
    "HEADLINE_METRICS",
    "PAPER_SEED_SPEAKER",
    "PAPER_N_BOOT_SPEAKER",
    "PESQ_MAX",
    "add_improvements",
    "center_within",
    "per_layer_correlations",
    "cluster_bootstrap_pearson_ci",
    "t_ci",
    "speaker_of",
    "speaker_level_analysis",
    "sisdr_sign_convention",
    "ceiling_free_gain",
]

#: metric label -> (enhanced column, degraded column)
METRICS: Dict[str, Tuple[str, str]] = {
    "PESQ": ("enhanced_pesq", "noisy_pesq"),
    "STOI": ("enhanced_stoi", "noisy_stoi"),
    "SI-SDR": ("enhanced_sisdr", "noisy_sisdr"),
    "DNSMOS-OVRL": ("enhanced_dnsmos_ovrl", "noisy_dnsmos_ovrl"),
    "DNSMOS-SIG": ("enhanced_dnsmos_sig", "noisy_dnsmos_sig"),
    "DNSMOS-BAK": ("enhanced_dnsmos_bak", "noisy_dnsmos_bak"),
}
HEADLINE_METRICS = ("PESQ", "STOI", "SI-SDR")
PAPER_SEED_SPEAKER = 20260803
PAPER_N_BOOT_SPEAKER = 2000
#: Wide-band PESQ upper bound used by the ceiling-free gain.
PESQ_MAX = 4.5


def available_metrics(df: pd.DataFrame, metrics: Optional[Iterable[str]] = None) -> list:
    names = list(metrics) if metrics is not None else list(METRICS)
    return [m for m in names if METRICS[m][0] in df.columns and METRICS[m][1] in df.columns
            and df[METRICS[m][0]].notna().any()]


def add_improvements(df: pd.DataFrame, metrics: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Add ``D_<metric>`` improvement columns (enhanced minus degraded)."""
    df = df.copy()
    for name in available_metrics(df, metrics):
        e, n = METRICS[name]
        df[f"D_{name}"] = df[e] - df[n]
    return df


def center_within(df: pd.DataFrame, cols: Sequence[str], by: Sequence[str] = ("layer", "snr"),
                  suffix: str = "_c") -> pd.DataFrame:
    """Subtract the within-group mean of each column in ``cols`` (group = ``by``)."""
    df = df.copy()
    g = df.groupby(list(by), observed=True)
    for c in cols:
        df[c + suffix] = df[c] - g[c].transform("mean")
    return df


def _level_col(df):
    return "snr" if "snr" in df.columns else "target_c50"


def per_layer_correlations(df: pd.DataFrame, layers: Sequence[str],
                           metrics: Optional[Iterable[str]] = None,
                           model: Optional[str] = None,
                           unit_col: Optional[str] = None,
                           n_boot: int = 0, seed: int = PAPER_SEED_SPEAKER) -> pd.DataFrame:
    """Per-layer Pearson (and Spearman) correlation between centred CKA and centred
    metric improvement. With ``unit_col`` and ``n_boot > 0`` an utterance-cluster
    bootstrap interval is attached."""
    level = _level_col(df)
    metrics = available_metrics(df, metrics)
    df = add_improvements(df[df["layer"].isin(list(layers))], metrics)
    df = center_within(df, ["CKA"] + [f"D_{m}" for m in metrics], by=("layer", level))
    labels = short_labels(model, layers) if model else list(layers)
    rows = []
    for name in metrics:
        col = f"D_{name}_c"
        for li, layer in enumerate(layers):
            g = df.loc[df["layer"] == layer, ["CKA_c", col] + ([unit_col] if unit_col else [])]
            g = g.replace([np.inf, -np.inf], np.nan).dropna()
            x, y = g["CKA_c"].to_numpy(), g[col].to_numpy()
            ok = len(g) >= 3 and x.std() > 0 and y.std() > 0
            row = dict(metric=name, layer_idx=li, layer=layer, label=labels[li],
                       pearson=float(pearsonr(x, y)[0]) if ok else np.nan,
                       spearman=float(spearmanr(x, y)[0]) if ok else np.nan, n=len(g))
            if ok and unit_col and n_boot > 0:
                lo, hi = cluster_bootstrap_pearson_ci(x, y, g[unit_col].to_numpy(), n_boot, seed)
                row.update(utt_lo=lo, utt_hi=hi, n_utt=int(g[unit_col].nunique()))
            rows.append(row)
    return pd.DataFrame(rows)


def _rho(sx, sy, sxx, syy, sxy, n):
    num = sxy - sx * sy / n
    den = np.sqrt(np.maximum(sxx - sx * sx / n, 0) * np.maximum(syy - sy * sy / n, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def cluster_bootstrap_pearson_ci(x, y, units, n_boot: int = PAPER_N_BOOT_SPEAKER,
                                 seed: int = PAPER_SEED_SPEAKER, ci: float = 0.95) -> Tuple[float, float]:
    """Percentile interval on Pearson *r* from resampling whole clusters (utterances).

    Uses per-cluster sufficient sums so each replicate is a reweighting, which makes
    2,000 replicates over 160k rows cheap.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    uid, inv = np.unique(np.asarray(units), return_inverse=True)
    K = uid.size
    S = np.zeros((K, 6))
    np.add.at(S, inv, np.column_stack([x, y, x * x, y * y, x * y, np.ones_like(x)]))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for s in range(0, n_boot, 200):
        m = min(200, n_boot - s)
        T = S[rng.integers(0, K, size=(m, K))].sum(axis=1)
        out[s:s + m] = _rho(T[:, 0], T[:, 1], T[:, 2], T[:, 3], T[:, 4], T[:, 5])
    out = out[np.isfinite(out)]
    if out.size < 100:
        return (np.nan, np.nan)
    tail = 100.0 * (1.0 - ci) / 2.0
    return float(np.percentile(out, tail)), float(np.percentile(out, 100 - tail))


def t_ci(values, ci: float = 0.95) -> Tuple[float, float, float, float, int]:
    """``(mean, sd, lo, hi, n)`` Student-*t* interval over a handful of per-unit values."""
    v = np.asarray([q for q in np.asarray(values, float) if np.isfinite(q)], float)
    n = v.size
    if n < 2:
        return (float(v[0]) if n else np.nan), np.nan, np.nan, np.nan, n
    m, sd = float(v.mean()), float(v.std(ddof=1))
    h = tdist.ppf(0.5 + ci / 2.0, n - 1) * sd / np.sqrt(n)
    return m, sd, m - h, m + h, n


def speaker_of(clean_idx, sweep: str = "824") -> np.ndarray:
    """Speaker label per ``clean_idx`` from the known index ranges of a sweep."""
    ci = np.asarray(clean_idx)
    out = np.full(ci.shape, None, dtype=object)
    for spk, (lo, hi) in SPEAKER_RANGES[sweep].items():
        out[(ci >= lo) & (ci <= hi)] = spk
    if (out == None).any():  # noqa: E711
        raise ValueError("some clean_idx values fall outside the known speaker ranges")
    return out


def speaker_level_analysis(df: pd.DataFrame, model: str, sweep: str = "824",
                           metrics: Optional[Iterable[str]] = None,
                           layers: Optional[Sequence[str]] = None,
                           speaker_col: Optional[str] = None,
                           n_boot: int = PAPER_N_BOOT_SPEAKER,
                           seed: int = PAPER_SEED_SPEAKER) -> Dict[str, pd.DataFrame]:
    """Utterance-level, per-speaker and speaker-level association tables.

    Returns ``{"utterance": ..., "per_speaker": ..., "speaker": ...}``. The
    speaker-level table carries ``mean_r, sd, spk_lo, spk_hi, n_spk, excludes_zero``
    from :func:`t_ci` over the per-speaker correlations (each speaker recentred on
    its own rows).
    """
    layers = list(layers) if layers is not None else probed_layers(model, df["layer"].unique())
    metrics = available_metrics(df, metrics)
    d = df[df["layer"].isin(layers)].copy()
    d["speaker"] = d[speaker_col] if speaker_col else speaker_of(d["clean_idx"].to_numpy(), sweep)
    labels = short_labels(model, layers)

    ut = per_layer_correlations(d, layers, metrics, model=model, unit_col="clean_idx",
                                n_boot=n_boot, seed=seed)
    ut.insert(0, "sweep", sweep)
    ut.insert(1, "model", model)
    ut = ut.rename(columns={"pearson": "r_utt"})

    ps = []
    for spk, sub in d.groupby("speaker"):
        t = per_layer_correlations(sub, layers, metrics, model=model)
        t.insert(0, "speaker", spk)
        ps.append(t)
    ps = pd.concat(ps, ignore_index=True).rename(columns={"pearson": "r"})
    ps.insert(0, "sweep", sweep)
    ps.insert(1, "model", model)

    rows = []
    for (me, li), g in ps.groupby(["metric", "layer_idx"]):
        m, sd, lo, hi, n = t_ci(g["r"].to_numpy())
        rows.append(dict(scope=sweep, model=model, metric=me, layer_idx=li, label=labels[li],
                         mean_r=m, sd=sd, spk_lo=lo, spk_hi=hi, n_spk=n,
                         sign_unanimous=bool((g["r"] > 0).all() or (g["r"] < 0).all()),
                         per_spk="|".join(f"{s}={v:+.4f}" for s, v in zip(g["speaker"], g["r"]))))
    sl = pd.DataFrame(rows)
    sl["excludes_zero"] = (sl["spk_lo"] * sl["spk_hi"]) > 0
    return {"utterance": ut, "per_speaker": ps, "speaker": sl}


def sisdr_sign_convention(df: pd.DataFrame) -> Dict[str, object]:
    """Detect whether a table stores ``-SI-SDR`` (an early helper negated it).

    The degraded input's SI-SDR must rise with SNR; a negative correlation between
    ``noisy_sisdr`` and ``snr`` means the stored values are negated.
    """
    d = df[["snr", "noisy_sisdr"]].dropna()
    if not len(d):
        return dict(corr=np.nan, needs_negation=False)
    c = float(np.corrcoef(d["snr"], d["noisy_sisdr"])[0, 1])
    return dict(corr=c, needs_negation=bool(c < 0))


def ceiling_free_gain(enhanced_pesq, noisy_pesq, pesq_max: float = PESQ_MAX) -> np.ndarray:
    """Fraction of the PESQ headroom captured, ``(enh - noisy) / (pesq_max - noisy)``."""
    e = np.asarray(enhanced_pesq, float)
    n = np.asarray(noisy_pesq, float)
    return (e - n) / (pesq_max - n)


def speaker_summary_from_published(per_speaker: pd.DataFrame, scope: str = "pooled4") -> pd.DataFrame:
    """Speaker-level ``t`` intervals from a per-speaker correlation table
    (``model, metric, layer_idx, label, speaker, r``), e.g. one pooled over the two
    sweeps for the four available speakers."""
    rows = []
    for (mo, me, li), g in per_speaker.groupby(["model", "metric", "layer_idx"]):
        m, sd, lo, hi, n = t_ci(g["r"].to_numpy())
        rows.append(dict(scope=scope, model=mo, metric=me, layer_idx=li, label=g["label"].iloc[0],
                         mean_r=m, sd=sd, spk_lo=lo, spk_hi=hi, n_spk=n,
                         sign_unanimous=bool((g["r"] > 0).all() or (g["r"] < 0).all())))
    out = pd.DataFrame(rows)
    out["excludes_zero"] = (out["spk_lo"] * out["spk_hi"]) > 0
    return out


__all__.append("speaker_summary_from_published")
__all__.append("available_metrics")
