#!/usr/bin/env python
"""Per-layer profiles, Table I statistics and the noise-set check on the noise axis
(paper Secs. II-D, III-A and III-D).

Input: the chunk directory written by ``run_snr_grid.py`` (one parquet per
model/noise/SNR) or an aggregated parquet with ``model_name, noise_name, snr,
clean_idx, layer, CKA``. For each model present:

* ``fits_<model>_snr.csv``: layer, depth, label, alpha, beta, r2, c_low, c_high, rng, auc
  (OLS on the mean level curve over utterances and the noises selected);
* ``summary_snr.csv``: one ``tradeoff_summary`` row per model (r and rho of
  (alpha, beta), saturation spread f, identity r, PC1/PC2, depth trends);
* ``per_noise_beta_<model>.csv``: layers x noises slope table and
  ``noise_set_independence_<model>.csv``: the test-5 vs train-8 vs neither-5
  contrasts of Sec. III-A;
* with ``--bootstrap``: ``hierarchical_bootstrap_snr.csv``, the two-way utterance x
  noise percentile intervals (one generator with ``--seed`` threaded through the
  models in the order muse, mpsenet, demucs and the layers in sorted order, which
  reproduces the published intervals).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _tables import read_table, reduce_chunks  # noqa: E402

from se_probe.layers import (  # noqa: E402
    DEMAND_NEITHER_SPLIT,
    VOICEBANK_DEMAND_TEST_NOISES,
    VOICEBANK_DEMAND_TRAIN_NOISES,
    probed_layers,
    short_labels,
)
from se_probe.profiles import (  # noqa: E402
    PAPER_N_BOOT,
    PAPER_SEED_HIERARCHICAL,
    fit_curves,
    noise_set_independence,
    tradeoff_summary,
    two_way_cluster_bootstrap,
)

CANON = {"muse": "muse", "mp-senet": "mpsenet", "mpsenet": "mpsenet", "demucs": "demucs"}
MODEL_ORDER = ["muse", "mpsenet", "demucs"]


def canon(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map(lambda x: CANON.get(x, x))


def per_noise_curves(cell_means: pd.DataFrame, layers) -> dict:
    """``{noise: layers x snr mean-curve DataFrame}`` from per-(noise, snr, layer) means."""
    out = {}
    for noise, g in cell_means.groupby("noise_name"):
        c = g.pivot(index="layer", columns="snr", values="CKA").reindex(layers)
        c.columns = c.columns.astype(float)
        out[noise] = c.sort_index(axis=1)
    return out


def analyze_model(cell_means: pd.DataFrame, model: str, noises=None) -> dict:
    """Fits, summary, per-noise slopes and the noise-set contrasts for one model."""
    layers = probed_layers(model, cell_means["layer"].unique())
    curves_by_noise = per_noise_curves(cell_means[cell_means["layer"].isin(layers)], layers)
    use = list(noises) if noises else sorted(curves_by_noise)
    missing = [n for n in use if n not in curves_by_noise]
    if missing:
        raise SystemExit(f"{model}: noises {missing} absent from the input")
    curves = sum(curves_by_noise[n] for n in use) / len(use)
    if curves.isna().any().any():
        raise SystemExit(f"{model}: incomplete (layer, snr) grid")
    f = fit_curves(curves)
    fits = pd.DataFrame({"layer": layers, "depth": np.arange(len(layers)),
                         "label": short_labels(model, layers), **f})
    summary = {"model": model, "noises": ",".join(use), **tradeoff_summary(curves, f)}
    per_noise_beta = pd.DataFrame({n: fit_curves(curves_by_noise[n])["beta"] for n in sorted(curves_by_noise)},
                                  index=pd.Index(layers, name="layer"))
    contrasts = []
    sets = {"test5": VOICEBANK_DEMAND_TEST_NOISES, "train8": VOICEBANK_DEMAND_TRAIN_NOISES,
            "neither5": DEMAND_NEITHER_SPLIT}
    for a, b in (("test5", "train8"), ("test5", "neither5"), ("train8", "neither5")):
        sa = [n for n in sets[a] if n in per_noise_beta.columns]
        sb = [n for n in sets[b] if n in per_noise_beta.columns]
        if len(sa) < 2 or len(sb) < 2:
            continue
        r = noise_set_independence(per_noise_beta, sa, sb)
        contrasts.append({"contrast": f"{a}_vs_{b}", "n_a": len(sa), "n_b": len(sb), **r})
    return {"layers": layers, "curves": curves, "fits": fits, "summary": summary,
            "per_noise_beta": per_noise_beta, "contrasts": pd.DataFrame(contrasts)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path, help="chunk directory or aggregated parquet")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--noises", nargs="+", default=None, help="noises to pool (default: all present)")
    p.add_argument("--bootstrap", action="store_true", help="two-way utterance x noise bootstrap CIs")
    p.add_argument("--n-boot", type=int, default=PAPER_N_BOOT)
    p.add_argument("--seed", type=int, default=PAPER_SEED_HIERARCHICAL)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cell = reduce_chunks(a.src, ["model_name", "noise_name", "snr", "layer"])
    cell["model_name"] = canon(cell["model_name"])
    models = a.models or [m for m in MODEL_ORDER if m in set(cell["model_name"])]
    rows = []
    for m in models:
        res = analyze_model(cell[cell["model_name"] == m], m, a.noises)
        res["fits"].to_csv(a.out_dir / f"fits_{m}_snr.csv", index=False)
        res["per_noise_beta"].to_csv(a.out_dir / f"per_noise_beta_{m}.csv")
        res["contrasts"].to_csv(a.out_dir / f"noise_set_independence_{m}.csv", index=False)
        rows.append(res["summary"])
        s = res["summary"]
        print(f"{m}: n={s['n_layers']} r(alpha,beta)={s['r_alpha_beta']:+.3f} rho={s['rho_alpha_beta']:+.3f} "
              f"f={s['sat_freedom']:.3f} PC1={s['pc1']:.2f} minR2={s['r2_min']:.3f} peak beta {s['peak_beta_layer']}")
        if len(res["contrasts"]):
            print(res["contrasts"][["contrast", "pearson_r", "spearman_rho", "mean_beta_a", "mean_beta_b"]].to_string(index=False))
    pd.DataFrame(rows).to_csv(a.out_dir / "summary_snr.csv", index=False)

    if a.bootstrap:
        rng = np.random.default_rng(a.seed)
        boot = []
        for m in models:
            df = read_table(a.src, columns=["model_name", "noise_name", "snr", "clean_idx", "layer", "CKA"])
            df["model_name"] = canon(df["model_name"])
            df = df[df["model_name"] == m]
            if a.noises:
                df = df[df["noise_name"].isin(a.noises)]
            layers = probed_layers(m, df["layer"].unique())
            t = two_way_cluster_bootstrap(df, layers, level_col="snr", cluster_col="noise_name",
                                          n_boot=a.n_boot, seed=a.seed, rng=rng)
            t.insert(0, "model_name", m)
            boot.append(t)
            print(f"{m}: bootstrap done ({len(t)} layers)")
        pd.concat(boot, ignore_index=True).to_csv(a.out_dir / "hierarchical_bootstrap_snr.csv", index=False)
    print("wrote", a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
