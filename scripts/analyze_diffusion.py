#!/usr/bin/env python
"""Diffusion-map statistics of Sec. III-G ("A Geometric View") from stored psi tables.

* per-block view (``diffusion_maps_per_layer_t0.5*.parquet``): for each representative
  block, the Spearman ordering of the distance from the 30 dB reference against SNR
  (the paper reports |rho| = 1.00 in every block), its strict monotonicity, the arc
  length of the centroid trajectory in the first two diffusion coordinates and the
  decoder/encoder arc-length ratio (3.03x in the paper);
* architecture view (``diffusion_maps_architecture_t5*.parquet``): per SNR, the
  between-group and within-group mean distances of the encoder and decoder layers on
  the per-panel min-max normalised matrix, plus the Spearman trend of the
  between-group distance with SNR.

Writes ``diffusion_per_block.csv``, ``diffusion_across_layers.csv`` and
``diffusion_summary.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from se_probe.diffusion_analysis import (  # noqa: E402
    BLOCK_NAMES,
    REPRESENTATIVE_LAYERS,
    arc_length_ratio,
    architecture_distance_matrices,
    block_statistics,
    block_trajectory_from_psi,
    group_distances,
)
from se_probe.muse.consts import ARCHITECTURE_LAYERS  # noqa: E402

PANELS = [-10, -5, 0, 10, 20, 30]


def per_block_table(psi: pd.DataFrame, layers=REPRESENTATIVE_LAYERS, names=BLOCK_NAMES) -> pd.DataFrame:
    rows = []
    for L, name in zip(layers, names):
        if L not in set(psi["layer"]):
            continue
        s, M = block_trajectory_from_psi(psi, L)
        rows.append({"block": name, "layer": L, **block_statistics(s, M)})
    return pd.DataFrame(rows)


def across_layers_table(arch: pd.DataFrame, layers=ARCHITECTURE_LAYERS) -> pd.DataFrame:
    D = architecture_distance_matrices(arch, layers)
    enc = [i for i, name in enumerate(layers) if "encoder_" in name]
    dec = [i for i, name in enumerate(layers) if "decoder_" in name]
    rows = []
    for snr, m in D.items():
        g = group_distances(m, enc, dec, normalize=True)
        rows.append({"snr": snr, "norm_between": g["between"], "norm_within_enc": g["within_a"],
                     "norm_within_dec": g["within_b"], "between_over_within": g["between_over_within"],
                     "raw_between": float(m[np.ix_(enc, dec)].mean())})
    return pd.DataFrame(rows).sort_values("snr").reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-layer", type=Path, required=True, help="per-layer psi parquet (t = 0.5)")
    p.add_argument("--arch", type=Path, default=None, help="architecture psi parquet (t = 5)")
    p.add_argument("--out-dir", required=True, type=Path)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pb = per_block_table(pd.read_parquet(a.per_layer))
    pb.to_csv(a.out_dir / "diffusion_per_block.csv", index=False)
    ratio = arc_length_ratio(pb)
    summary = {"per_block": {"min_abs_rho": float(pb["abs_rho"].min()), "all_monotone": bool(pb["monotone_strict"].all()), **ratio}}
    print(pb[["block", "n_components", "abs_rho", "arc_len_2d", "max_dist_from_ref"]].to_string(index=False))
    print(f"arc length: encoder {ratio['encoder_mean']:.4f}, decoder {ratio['decoder_mean']:.4f}, ratio {ratio['ratio']:.2f}x")
    if a.arch is not None:
        al = across_layers_table(pd.read_parquet(a.arch))
        al.to_csv(a.out_dir / "diffusion_across_layers.csv", index=False)
        panels = al[al["snr"].isin(PANELS)]
        summary["across_layers"] = {
            "spearman_between_vs_snr_all": float(stats.spearmanr(al["snr"], al["norm_between"]).statistic),
            "monotone_decreasing_over_panels": bool(np.all(np.diff(panels["norm_between"].to_numpy()) < 0)),
            "between_over_within_m10": float(al.loc[al["snr"] == -10, "between_over_within"].iloc[0]),
            "between_over_within_p30": float(al.loc[al["snr"] == 30, "between_over_within"].iloc[0]),
        }
        print(panels.to_string(index=False))
    with open(a.out_dir / "diffusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
