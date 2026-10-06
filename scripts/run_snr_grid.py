#!/usr/bin/env python
"""Additive-noise probing sweep (paper Sec. II-C, "Experimental Setup").

Grid: 824 VoiceBank-DEMAND test utterances x 41 integer SNRs (-10..30 dB) x 18 DEMAND
noise recordings x one model (MUSE pretrained, MP-SENet DNS or Demucs DNS64), CKA
between the clean-reference and noisy activations at every hooked layer.

One parquet per (model, noise, snr) with columns ``clean_idx, snr, layer, CKA,
model_name, noise_name``; existing chunks are skipped. Pass ``--checkpoint`` to probe
a fine-tuned generator with the same grid (the emergence checkpoints and the
dereverberation fine-tunes), ``--model-name`` to tag its rows.

Example (one noise, smoke test)::

    python scripts/run_snr_grid.py --model muse --clean /data/vb_demand/clean_test \
        --out-dir results_df/snr_chunks --noises TBUS --snrs -10 0 30 --n-utts 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from _clean_sources import load_voicebank_test, resolve_clean_source  # noqa: E402
from _grid import (  # noqa: E402
    DEMUCS_LEN,
    MODEL_NAMES,
    Heartbeat,
    chunk_loop,
    cka_rows,
    extract_stack,
    load_extractor,
    pad_or_trim,
)

from se_probe.data_generation import add_noise_at_snr, load_demand_noise  # noqa: E402
from se_probe.device import device_info, get_device  # noqa: E402
from se_probe.layers import DEMAND_NOISES_18, SNR_GRID  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["muse", "mpsenet", "demucs"])
    p.add_argument("--clean", default=None, help="clean wav dir or 'hf' (default $SEPROBE_CLEAN_TEST_DIR)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--noises", nargs="+", default=DEMAND_NOISES_18)
    p.add_argument("--snrs", nargs="+", type=int, default=[int(s) for s in SNR_GRID])
    p.add_argument("--n-utts", type=int, default=824)
    p.add_argument("--checkpoint", default=None, help="fine-tuned generator to probe instead of the pretrained one")
    p.add_argument("--model-name", default=None, help="model_name column value (default: MUSE / MP-SENet / Demucs)")
    p.add_argument("--utt-chunk", type=int, default=150, help="utterances per extraction batch (RAM bound)")
    p.add_argument("--demucs-len", type=int, default=DEMUCS_LEN)
    p.add_argument("--log", type=Path, default=None, help="heartbeat log file")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    hb = Heartbeat(f"snr/{a.model}", a.log)
    device = get_device()
    model_name = a.model_name or MODEL_NAMES[a.model]
    hb(f"START {device_info(device)} model={a.model} n_utts={a.n_utts} snrs={len(a.snrs)} noises={len(a.noises)} out={a.out_dir}")

    def chunk_path(noise, snr):
        return a.out_dir / f"{a.model}__{noise}__snr{snr:+03d}.parquet"

    conds = [(n, s) for n in a.noises for s in a.snrs]
    if all(chunk_path(*c).exists() for c in conds):
        hb("all chunks present, nothing to do")
        return 0

    _, clean_all = load_voicebank_test(resolve_clean_source(a.clean), a.n_utts)
    if a.model == "demucs":
        clean_all = [pad_or_trim(x, a.demucs_len) for x in clean_all]
    hb(f"clean loaded: {len(clean_all)} utterances")

    ext = load_extractor(a.model, device, checkpoint=a.checkpoint)
    _ = ext(clean_all[0])
    batches = [(i, clean_all[i:i + a.utt_chunk]) for i in range(0, len(clean_all), a.utt_chunk)]
    noise_cache = {}

    def make_rows(noise_name, snr):
        if noise_name not in noise_cache:
            noise_cache.clear()
            noise_cache[noise_name] = load_demand_noise(noise_name)
        noise = noise_cache[noise_name]
        rows = []
        for off, batch in batches:
            clean_stack = extract_stack(ext, batch)
            noisy = [add_noise_at_snr(c, noise, snr).astype("float32") for c in batch]
            noisy_stack = extract_stack(ext, noisy)
            rows += cka_rows(clean_stack, noisy_stack, device,
                             {"snr": int(snr), "model_name": model_name, "noise_name": noise_name}, offset=off)
            del clean_stack, noisy_stack, noisy
        return rows

    n = chunk_loop(conds, chunk_path, make_rows, hb)
    hb(f"DONE {a.model}: wrote {n} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
