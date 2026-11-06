#!/usr/bin/env python
"""Random-initialisation control (paper Sec. "Origin of the Profile in Training").

The identical probing pipeline applied to an untrained MUSE instantiated from the
same configuration with randomly initialised weights, over three seeds (0, 1, 2):
824 utterances x 41 integer SNRs x the five VoiceBank-DEMAND test noises
(TBUS, SCAFE, DLIVING, OOFFICE, SPSQUARE), 24 ``.norm1`` probed layers.

One parquet per (seed, noise, snr) with columns ``clean_idx, snr, layer, CKA,
model_name, noise_name, seed``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _clean_sources import load_voicebank_test, resolve_clean_source  # noqa: E402
from _grid import Heartbeat, chunk_loop, cka_rows, extract_stack, load_extractor  # noqa: E402

from se_probe.data_generation import add_noise_at_snr, load_demand_noise  # noqa: E402
from se_probe.device import device_info, get_device  # noqa: E402
from se_probe.layers import SNR_GRID, VOICEBANK_DEMAND_TEST_NOISES  # noqa: E402
from se_probe.muse.consts import NORM1_LAYERS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True, help="random-init seed (paper: 0, 1, 2)")
    p.add_argument("--clean", default=None)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--noises", nargs="+", default=VOICEBANK_DEMAND_TEST_NOISES)
    p.add_argument("--snrs", nargs="+", type=int, default=[int(s) for s in SNR_GRID])
    p.add_argument("--n-utts", type=int, default=824)
    p.add_argument("--utt-chunk", type=int, default=150)
    p.add_argument("--log", type=Path, default=None)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    hb = Heartbeat(f"randominit/seed{a.seed}", a.log)
    device = get_device()
    hb(f"START {device_info(device)} seed={a.seed} n_utts={a.n_utts}")

    def chunk_path(noise, snr):
        return a.out_dir / f"randominit_seed{a.seed}__{noise}__snr{snr:+03d}.parquet"

    conds = [(n, s) for n in a.noises for s in a.snrs]
    if all(chunk_path(*c).exists() for c in conds):
        hb("all chunks present, nothing to do")
        return 0
    _, clean_all = load_voicebank_test(resolve_clean_source(a.clean), a.n_utts)
    ext = load_extractor("muse", device, random_seed=a.seed)
    _ = ext(clean_all[0])
    batches = [(i, clean_all[i:i + a.utt_chunk]) for i in range(0, len(clean_all), a.utt_chunk)]
    model_name = f"MUSE_random_seed{a.seed}"

    def make_rows(noise_name, snr):
        noise = load_demand_noise(noise_name)
        rows = []
        for off, batch in batches:
            clean_stack = extract_stack(ext, batch, NORM1_LAYERS)
            noisy_stack = extract_stack(ext, [add_noise_at_snr(c, noise, snr).astype("float32") for c in batch], NORM1_LAYERS)
            rows += cka_rows(clean_stack, noisy_stack, device,
                             {"snr": int(snr), "model_name": model_name, "noise_name": noise_name, "seed": a.seed}, offset=off)
        return rows

    n = chunk_loop(conds, chunk_path, make_rows, hb)
    hb(f"DONE seed={a.seed}: wrote {n} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
