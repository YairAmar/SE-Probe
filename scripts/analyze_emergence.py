#!/usr/bin/env python
"""Emergence of the profile over the MUSE dereverberation fine-tune (paper Sec. III-E).

Input: the per-epoch chunk tree of ``run_emergence_grid.py`` (``e<N>/snr/*.parquet``)
and the pretrained MUSE SNR chunks. Per checkpoint the per-layer ``(alpha, beta)``
profile is fitted on the mean level curve; ``emergence_convergence.csv`` gives the
Pearson correlation of every epoch's ``beta`` and ``alpha`` profile with the final
epoch's, and ``emergence_summary.json`` the concentration of the pretrained-to-final
change in the decoder and refinement stages relative to the first encoder block
(``se_probe.profiles.emergence_convergence``).
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
from se_probe.profiles import emergence_convergence, fit_curves  # noqa: E402


def profile_from_cells(cells: pd.DataFrame, layers) -> pd.DataFrame:
    c = cells.groupby(["layer", "snr"])["CKA"].mean().unstack("snr").reindex(layers)
    c.columns = c.columns.astype(float)
    f = fit_curves(c.sort_index(axis=1))
    return pd.DataFrame({"layer": layers, "depth": np.arange(len(layers)), "label": short_labels("muse", layers), **f})


def analyze(profiles: dict, final_key, pretrained_key="pretrained") -> dict:
    return emergence_convergence(profiles, final_key, pretrained_key=pretrained_key)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emergence-root", required=True, type=Path, help="dir holding e<N>/snr/*.parquet")
    p.add_argument("--pretrained-src", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--noises", nargs="+", default=VOICEBANK_DEMAND_TEST_NOISES)
    p.add_argument("--final-epoch", type=int, default=None, help="default: the largest epoch found")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    epochs = sorted(int(d.name[1:]) for d in a.emergence_root.glob("e*") if d.is_dir() and d.name[1:].isdigit())
    if not epochs:
        raise SystemExit(f"no e<N> directories under {a.emergence_root}")
    pre = reduce_chunks(a.pretrained_src, ["noise_name", "snr", "layer"])
    pre = pre[pre["noise_name"].isin(a.noises)]
    layers = probed_layers("muse", pre["layer"].unique())
    profiles = {"pretrained": profile_from_cells(pre[pre["layer"].isin(layers)], layers)}
    for e in epochs:
        cells = reduce_chunks(a.emergence_root / f"e{e}", ["noise_name", "snr", "layer"])
        cells = cells[cells["noise_name"].isin(a.noises) & cells["layer"].isin(layers)]
        profiles[e] = profile_from_cells(cells, layers)
        profiles[e].to_csv(a.out_dir / f"profile_snr_e{e}.csv", index=False)
    profiles["pretrained"].to_csv(a.out_dir / "profile_snr_pretrained.csv", index=False)
    final = a.final_epoch or epochs[-1]
    res = analyze(profiles, final)
    res["convergence"].to_csv(a.out_dir / "emergence_convergence.csv", index=False)
    with open(a.out_dir / "emergence_summary.json", "w") as f:
        json.dump({"epochs": epochs, "final_epoch": final, "concentration": res["concentration"]}, f, indent=2)
    print(res["convergence"].to_string(index=False))
    print(json.dumps(res["concentration"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
