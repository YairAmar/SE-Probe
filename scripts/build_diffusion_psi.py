#!/usr/bin/env python
"""Diffusion-map embeddings of the centroids (paper Sec. II-D "Diffusion Maps").

Two views, two different orders of operation (the order is what reproduces the
published figures):

* **per-layer view** (``t = 0.5``): for each representative layer ALL ``(noise, snr)``
  centroids are embedded jointly; averaging over noises happens afterwards in psi
  space (``se_probe.diffusion_analysis.block_trajectory_from_psi``). Output schema:
  ``layer, noise_name, snr, n_components, eigenvalues, psi_*``.
* **architecture view** (``t = 5``): centroids are first averaged over the test noises
  per (layer, snr), then the 16 encoder/decoder ``ARCHITECTURE_LAYERS`` are embedded
  per SNR. Output schema: ``snr, layer, layer_idx, n_components, eigenvalues, psi_*``.

Gaussian affinity with the median pairwise squared distance as bandwidth and a 99 %
cumulative eigenvalue-energy cutoff (``se_probe.diffusion_maps.diffusion_map_torch``).
``--control`` re-embeds a second centroid directory (e.g. the earlier 780-utterance
sweep) and reports the per-component |r| agreement against stored psi tables.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from se_probe.centroids import (  # noqa: E402
    average_centroids,
    build_diffusion_result_df,
    decode_centroids,
    load_centroids,
)
from se_probe.diffusion_analysis import (  # noqa: E402
    CUTOFF,
    REPRESENTATIVE_LAYERS,
    T_ARCHITECTURE,
    T_PER_LAYER,
    psi_matrix,
)
from se_probe.diffusion_maps import diffusion_map_torch  # noqa: E402
from se_probe.layers import VOICEBANK_DEMAND_TEST_NOISES  # noqa: E402
from se_probe.muse.consts import ARCHITECTURE_LAYERS  # noqa: E402


def per_layer_view(centroids: pd.DataFrame, layers, t: float = T_PER_LAYER, cutoff: float = CUTOFF, device="cpu") -> pd.DataFrame:
    out = []
    for layer in layers:
        ld = centroids[centroids["layer"] == layer].sort_values(["noise_name", "snr"]).reset_index(drop=True)
        if ld.empty:
            raise SystemExit(f"layer {layer} absent from the centroid table")
        psi, eigs = diffusion_map_torch(decode_centroids(ld["centroid"]), cutoff=cutoff,
                                        diffusion_time=t, device=device, return_eigs=True)
        out.append(build_diffusion_result_df(psi, eigs, ld, ["layer", "noise_name", "snr"]))
    return pd.concat(out, ignore_index=True)


def architecture_view(centroids: pd.DataFrame, noises, layers=ARCHITECTURE_LAYERS, t: float = T_ARCHITECTURE,
                      cutoff: float = CUTOFF, device="cpu") -> pd.DataFrame:
    avg = average_centroids(centroids, group_by=["layer", "snr"], noise_types=list(noises))
    if not (avg["n_averaged"] == len(noises)).all():
        raise SystemExit("architecture view: some (layer, snr) cells are not averaged over every noise")
    out = []
    for snr in sorted(avg["snr"].unique()):
        d = avg[(avg["snr"] == snr) & avg["layer"].isin(layers)].copy()
        d["layer"] = pd.Categorical(d["layer"], categories=list(layers), ordered=True)
        d = d.sort_values("layer").reset_index(drop=True)
        if len(d) != len(layers):
            raise SystemExit(f"architecture view SNR {snr}: {len(d)} layers, expected {len(layers)}")
        psi, eigs = diffusion_map_torch(decode_centroids(d["centroid"]), cutoff=cutoff,
                                        diffusion_time=t, device=device, return_eigs=True)
        d["layer"] = d["layer"].astype(str)
        d["layer_idx"] = range(len(d))
        out.append(build_diffusion_result_df(psi, eigs, d, ["snr", "layer", "layer_idx"]))
    return pd.concat(out, ignore_index=True)


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum per-component |r| between two embeddings (sign-free), shared components."""
    k = min(a.shape[1], b.shape[1])
    rs = [abs(np.corrcoef(a[:, j], b[:, j])[0, 1]) for j in range(k)
          if np.std(a[:, j]) > 0 and np.std(b[:, j]) > 0]
    return float(min(rs)) if rs else float("nan")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--centroids-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--noises", nargs="+", default=VOICEBANK_DEMAND_TEST_NOISES)
    p.add_argument("--layers", nargs="+", default=REPRESENTATIVE_LAYERS, help="layers of the per-layer view")
    p.add_argument("--t-per-layer", type=float, default=T_PER_LAYER)
    p.add_argument("--t-arch", type=float, default=T_ARCHITECTURE)
    p.add_argument("--tag", default="", help="suffix of the output file names")
    p.add_argument("--control", type=Path, default=None, help="reference centroid dir to re-embed")
    p.add_argument("--control-per-layer", type=Path, default=None, help="stored per-layer psi table to compare with")
    p.add_argument("--control-arch", type=Path, default=None, help="stored architecture psi table to compare with")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.control is not None:
        c = load_centroids(str(a.control), pattern="*.parquet")
        c = c[c["noise_name"].isin(a.noises)]
        if a.control_per_layer:
            mine = per_layer_view(c, a.layers, a.t_per_layer)
            stored = pd.read_parquet(a.control_per_layer)
            for L in a.layers:
                x = psi_matrix(stored[stored["layer"] == L].sort_values(["noise_name", "snr"]))
                y = psi_matrix(mine[mine["layer"] == L].sort_values(["noise_name", "snr"]))
                print(f"[control] per-layer {L.split('.')[1]}: min |r| = {agreement(x, y):.6f}")
        if a.control_arch:
            mine = architecture_view(c, a.noises, t=a.t_arch)
            stored = pd.read_parquet(a.control_arch)
            x = psi_matrix(stored.sort_values(["snr", "layer"]))
            y = psi_matrix(mine.sort_values(["snr", "layer"]))
            print(f"[control] architecture: min |r| = {agreement(x, y):.6f}")

    cen = load_centroids(str(a.centroids_dir), pattern="*.parquet")
    cen = cen[cen["noise_name"].isin(a.noises)]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pl = per_layer_view(cen, a.layers, a.t_per_layer)
    p = a.out_dir / f"diffusion_maps_per_layer_t{a.t_per_layer:g}{a.tag}.parquet"
    pl.to_parquet(p, index=False)
    print(f"per-layer view: {len(pl)} rows -> {p}")
    ar = architecture_view(cen, a.noises, t=a.t_arch)
    p = a.out_dir / f"diffusion_maps_architecture_t{a.t_arch:g}{a.tag}.parquet"
    ar.to_parquet(p, index=False)
    print(f"architecture view: {len(ar)} rows -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
