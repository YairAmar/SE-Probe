#!/usr/bin/env python
"""Reverberation probing sweep (paper Sec. II-C, C50 axis).

Grid: 824 VoiceBank-DEMAND test utterances x 13 nominal target C50 levels (-5..25 dB,
2.5 dB steps) x five RIRs per utterance drawn at random (``default_rng(42)``) from the
88-RIR six-room AIR test pool (lecture 32, meeting 24, office 18, corridor 6,
bathroom 4, kitchen 4). The late tail of each RIR is rescaled to the target C50 with
the first 50 ms held fixed; the reference activations use the same RIR at C50 = 50 dB.

Six arms: ``{muse,mpsenet,demucs}_{pre,ft}`` (pretrained denoising checkpoint vs the
dereverberation fine-tune given by ``--checkpoint``). Rows: ``clean_idx, utt_id,
rir_name, target_c50, achieved_c50, drr, layer, CKA, model_name``. Utterance shards
(``--utt-start/--utt-stop``) write ``shard_<arm>_<start>_<stop>.parquet``.
``verify_pool`` asserts the produced table really spans the published pool.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _clean_sources import load_voicebank_test, resolve_clean_source  # noqa: E402
from _grid import (  # noqa: E402
    DEMUCS_LEN,
    Heartbeat,
    cka_rows,
    extract_stack,
    load_extractor,
    pad_or_trim,
)

from se_probe.consts import AIR_TRAIN_ROOMS, DEFAULT_TARGET_C50S, REFERENCE_C50  # noqa: E402
from se_probe.data_generation import convolve_audio  # noqa: E402
from se_probe.device import device_info, get_device  # noqa: E402
from se_probe.io import load_air_test_rirs  # noqa: E402
from se_probe.layers import AIR_TEST_POOL_ROOM_COUNTS, AIR_TEST_POOL_SIZE  # noqa: E402
from se_probe.metrics import c50 as compute_c50  # noqa: E402
from se_probe.metrics import drr as compute_drr  # noqa: E402

ARMS = ["muse_pre", "muse_ft", "mpsenet_pre", "mpsenet_ft", "demucs_pre", "demucs_ft"]
RIR_SEED = 42
N_RIRS_PER_UTT = 5


def room_of(rir_name: str) -> str:
    hits = [r for r in AIR_TEST_POOL_ROOM_COUNTS if r in rir_name]
    if len(hits) != 1:
        raise AssertionError(f"RIR {rir_name!r} matches {hits} test rooms, expected exactly one")
    return hits[0]


def verify_pool(df: pd.DataFrame, n_utts: int = 824) -> int:
    """Hard gate on a produced table: 88 RIRs, six rooms with the published counts, no
    training-split room, five RIRs per utterance, and per-utterance (not fixed)
    draws. Returns the number of distinct 5-RIR combinations."""
    rirs = sorted(df["rir_name"].unique())
    if len(rirs) != AIR_TEST_POOL_SIZE:
        raise AssertionError(f"{len(rirs)} distinct RIRs, expected {AIR_TEST_POOL_SIZE}")
    for bad in AIR_TRAIN_ROOMS:
        hit = [r for r in rirs if bad in r]
        if hit:
            raise AssertionError(f"training-split room {bad} present: {hit[:3]}")
    rooms = {}
    for r in rirs:
        rooms[room_of(r)] = rooms.get(room_of(r), 0) + 1
    if rooms != AIR_TEST_POOL_ROOM_COUNTS:
        raise AssertionError(f"room counts {rooms} != published pool {AIR_TEST_POOL_ROOM_COUNTS}")
    per_utt = df.groupby("clean_idx")["rir_name"].nunique()
    if set(per_utt.unique()) != {N_RIRS_PER_UTT}:
        raise AssertionError(f"RIRs per utterance are {sorted(set(per_utt.unique()))}, expected {N_RIRS_PER_UTT}")
    if df["clean_idx"].nunique() != n_utts:
        raise AssertionError(f"{df['clean_idx'].nunique()} utterances, expected {n_utts}")
    combos = df.groupby("clean_idx")["rir_name"].apply(lambda s: tuple(sorted(set(s)))).nunique()
    if combos < 100:
        raise AssertionError(f"only {combos} distinct 5-RIR combinations; sampling looks like a fixed slice")
    return int(combos)


def draw_rirs(n_utts: int, n_pool: int, seed: int = RIR_SEED, k: int = N_RIRS_PER_UTT) -> np.ndarray:
    """``(n_utts, k)`` RIR indices, ``k`` without replacement per utterance, seeded."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(n_pool, size=k, replace=False) for _ in range(n_utts)])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--checkpoint", default=None, help="fine-tuned generator (required for *_ft arms)")
    p.add_argument("--clean", default=None)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--c50s", nargs="+", type=float, default=list(DEFAULT_TARGET_C50S))
    p.add_argument("--reference-c50", type=float, default=REFERENCE_C50)
    p.add_argument("--n-utts", type=int, default=824)
    p.add_argument("--utt-start", type=int, default=0)
    p.add_argument("--utt-stop", type=int, default=None)
    p.add_argument("--rir-seed", type=int, default=RIR_SEED)
    p.add_argument("--demucs-len", type=int, default=DEMUCS_LEN)
    p.add_argument("--verify", action="store_true", help="run verify_pool on the written shard (needs all 824 utterances)")
    p.add_argument("--log", type=Path, default=None)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    model = a.arm.split("_")[0]
    if a.arm.endswith("_ft") and not a.checkpoint:
        raise SystemExit("--checkpoint is required for a fine-tuned arm")
    hb = Heartbeat(f"reverb/{a.arm}", a.log)
    device = get_device()
    stop = a.utt_stop if a.utt_stop is not None else a.n_utts
    out = a.out_dir / f"shard_{a.arm}_{a.utt_start:04d}_{stop:04d}.parquet"
    if out.exists():
        hb(f"{out} exists, nothing to do")
        return 0
    hb(f"START {device_info(device)} arm={a.arm} utts [{a.utt_start}, {stop})")

    ids, clean_all = load_voicebank_test(resolve_clean_source(a.clean), a.n_utts)
    rirs, rir_names = load_air_test_rirs()
    if len(rirs) != AIR_TEST_POOL_SIZE:
        hb(f"WARNING: {len(rirs)} RIRs found in $SEPROBE_AIR_RIR_DIR, the paper pool has {AIR_TEST_POOL_SIZE}")
    draws = draw_rirs(a.n_utts, len(rirs), a.rir_seed)  # full-sweep draw so shards agree

    ext = load_extractor(model, device, checkpoint=a.checkpoint, reverb=True)
    rows = []
    for u in range(a.utt_start, stop):
        clean = clean_all[u]
        if model == "demucs":
            clean = pad_or_trim(clean, a.demucs_len)
        for ri in draws[u]:
            rir = rirs[ri]
            ref_audio, _ = convolve_audio(clean, rir, a.reference_c50)
            ref_stack = extract_stack(ext, [ref_audio.astype("float32")])
            for tc in a.c50s:
                rev, rir_mod = convolve_audio(clean, rir, tc)
                rev_stack = extract_stack(ext, [rev.astype("float32")])
                rows += cka_rows(ref_stack, rev_stack, device, {
                    "utt_id": ids[u], "rir_name": rir_names[ri], "target_c50": float(tc),
                    "achieved_c50": float(compute_c50(rir_mod)), "drr": float(compute_drr(rir_mod)),
                    "model_name": a.arm}, offset=u)
        if (u - a.utt_start + 1) % 25 == 0:
            hb(f"{u - a.utt_start + 1}/{stop - a.utt_start} utterances")
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    hb(f"wrote {out} ({len(df)} rows)")
    if a.verify:
        hb(f"pool OK: {verify_pool(df, a.n_utts)} distinct 5-RIR combinations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
