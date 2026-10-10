#!/usr/bin/env python
"""Random-initialisation control statistics (paper Sec. III-E).

Input: the chunk directory of ``run_random_init_grid.py`` (all seeds) and the trained
MUSE SNR chunks (``run_snr_grid.py``) or a trained fits table. Per seed, the per-layer
slope ``beta`` on the mean level curve over utterances and the selected noises, the
minimum per-(layer, level) mean CKA, ``rho(depth, beta)``, the factor between the
trained peak slope and the largest untrained one, and the untrained-vs-trained profile
correlation (``se_probe.profiles.random_init_summary``). Writes
``random_init_per_layer.csv`` (depth, beta per seed, mean, sd, cka_min) and
``random_init_summary.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _tables import reduce_chunks  # noqa: E402

from se_probe.layers import VOICEBANK_DEMAND_TEST_NOISES, probed_layers, short_labels  # noqa: E402
from se_probe.profiles import fit_curves, random_init_summary  # noqa: E402


def curves_from_cells(cells: pd.DataFrame, layers) -> pd.DataFrame:
    c = cells.groupby(["layer", "snr"])["CKA"].mean().unstack("snr").reindex(layers)
    c.columns = c.columns.astype(float)
    return c.sort_index(axis=1)


def analyze(rand_cells: pd.DataFrame, trained_curves: pd.DataFrame, layers) -> dict:
    """``rand_cells``: per-(seed, noise_name, snr, layer) mean CKA; ``trained_curves``:
    layers x snr mean-curve of the trained model on the same grid."""
    seed_betas, per_seed_curves = {}, {}
    for seed, g in rand_cells.groupby("seed"):
        c = curves_from_cells(g, layers)
        per_seed_curves[seed] = c
        seed_betas[str(int(seed))] = fit_curves(c)["beta"]
    all_curves = curves_from_cells(rand_cells, layers)
    trained_beta = fit_curves(trained_curves)["beta"]
    summary = random_init_summary(trained_beta, seed_betas, cka_min=float(all_curves.to_numpy().min()))
    i, j = np.unravel_index(int(np.argmin(all_curves.to_numpy())), all_curves.shape)
    summary["cka_min_layer"] = short_labels("muse", [layers[i]])[0]
    summary["cka_min_snr"] = float(all_curves.columns[j])
    tab = pd.DataFrame({"depth": np.arange(len(layers)), "layer": layers, "label": short_labels("muse", layers),
                        **{f"beta_s{k}": v for k, v in seed_betas.items()}, "beta_trained": trained_beta})
    B = np.stack(list(seed_betas.values()), axis=1)
    tab["beta_mean"] = B.mean(axis=1)
    tab["beta_sd"] = B.std(axis=1, ddof=1) if B.shape[1] > 1 else np.nan
    tab["cka_min"] = all_curves.min(axis=1).to_numpy()
    tab["cka_mean"] = all_curves.mean(axis=1).to_numpy()
    return {"per_layer": tab, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--random-src", required=True, type=Path, help="random-init chunk dir (all seeds)")
    p.add_argument("--trained-src", required=True, type=Path, help="trained MUSE SNR chunk dir / parquet")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--noises", nargs="+", default=VOICEBANK_DEMAND_TEST_NOISES)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rand = reduce_chunks(a.random_src, ["seed", "noise_name", "snr", "layer"])
    rand = rand[rand["noise_name"].isin(a.noises)]
    layers = probed_layers("muse", rand["layer"].unique())
    rand = rand[rand["layer"].isin(layers)]
    tr = reduce_chunks(a.trained_src, ["noise_name", "snr", "layer"])
    tr = tr[tr["noise_name"].isin(a.noises) & tr["layer"].isin(layers)]
    res = analyze(rand, curves_from_cells(tr, layers), layers)
    res["per_layer"].to_csv(a.out_dir / "random_init_per_layer.csv", index=False)
    with open(a.out_dir / "random_init_summary.json", "w") as f:
        json.dump(res["summary"], f, indent=2, default=float)
    s = res["summary"]
    print(f"min CKA {s['cka_min']:.4f} at {s['cka_min_layer']} / {s['cka_min_snr']:+g} dB; trained peak beta "
          f"{s['trained_peak_beta']:.5f}; factor over largest seed {s['factor_trained_over_largest_seed']:.1f}x")
    for k, v in s["per_seed"].items():
        print(f"  seed {k}: max beta {v['max_beta']:.5f} rho(depth,beta) {v['rho_depth_beta']:+.3f} (p={v['rho_p']:.2g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
