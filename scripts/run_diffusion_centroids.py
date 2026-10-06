#!/usr/bin/env python
"""Centroid representations for the diffusion-map analysis (paper Sec. II-D
"Diffusion Maps", Eq. centroid).

For each of the 24 MUSE ``.norm1`` layers and each (noise, SNR) condition the pooled
first-segment activations of the 824 test utterances are averaged into one centroid
vector (``C x F`` flattened). Grid: 41 integer SNRs x the five VoiceBank-DEMAND test
noises. One parquet per (noise, snr) with columns ``layer, snr, noise_name, n_utts,
dim, centroid`` (float32 bytes), the schema ``se_probe.centroids`` reads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from _clean_sources import load_voicebank_test, resolve_clean_source  # noqa: E402
from _grid import Heartbeat, chunk_loop, load_extractor  # noqa: E402

from se_probe.data_generation import add_noise_at_snr, load_demand_noise  # noqa: E402
from se_probe.device import device_info, get_device  # noqa: E402
from se_probe.layers import SNR_GRID, VOICEBANK_DEMAND_TEST_NOISES  # noqa: E402
from se_probe.muse.consts import NORM1_LAYERS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean", default=None)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--noises", nargs="+", default=VOICEBANK_DEMAND_TEST_NOISES)
    p.add_argument("--snrs", nargs="+", type=int, default=[int(s) for s in SNR_GRID])
    p.add_argument("--n-utts", type=int, default=824)
    p.add_argument("--layers", nargs="+", default=NORM1_LAYERS)
    p.add_argument("--log", type=Path, default=None)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    hb = Heartbeat("centroids", a.log)
    device = get_device()
    hb(f"START {device_info(device)} n_utts={a.n_utts} layers={len(a.layers)}")

    def chunk_path(noise, snr):
        return a.out_dir / f"centroids__{noise}__snr{snr:+03d}.parquet"

    conds = [(n, s) for n in a.noises for s in a.snrs]
    if all(chunk_path(*c).exists() for c in conds):
        hb("all chunks present, nothing to do")
        return 0
    _, clean_all = load_voicebank_test(resolve_clean_source(a.clean), a.n_utts)
    ext = load_extractor("muse", device, checkpoint=a.checkpoint)
    _ = ext(clean_all[0])

    def make_rows(noise_name, snr):
        noise = load_demand_noise(noise_name)
        sums = {L: None for L in a.layers}
        n = 0
        for c in clean_all:
            acts, _ = ext(add_noise_at_snr(c, noise, snr).astype("float32"))
            for L in a.layers:
                v = acts[L].detach().cpu().numpy().reshape(-1).astype("float64")
                sums[L] = v if sums[L] is None else sums[L] + v
            n += 1
        rows = []
        for L in a.layers:
            mean = (sums[L] / n).astype("float32")
            rows.append({"layer": L, "snr": int(snr), "noise_name": noise_name, "n_utts": n,
                         "dim": int(mean.size), "centroid": mean.tobytes()})
        return rows

    k = chunk_loop(conds, chunk_path, make_rows, hb)
    hb(f"DONE: wrote {k} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
