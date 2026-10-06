#!/usr/bin/env python
"""Emergence of the profile along the dereverberation fine-tune (paper Sec. "Origin
of the Profile in Training").

Probes the intermediate MUSE checkpoints (epochs 1, 3, 5, 7, 9, 10, 12, 24, 36, 48 in
the paper) on the noise axis: 824 utterances x 41 integer SNRs x the five
VoiceBank-DEMAND test noises, 24 ``.norm1`` layers. Each epoch reuses the SNR-grid
code path of ``run_snr_grid.py`` with ``--checkpoint``.

Checkpoints are given either explicitly (``--checkpoints 1=path/g_x 3=path/g_y``) or as a
root holding ``epoch_<N>/g_<step>`` files; within an epoch directory the file with
the LARGEST step number is used (a plain name sort once picked the wrong file).
"""
from __future__ import annotations

import argparse
import re
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

PAPER_EPOCHS = [1, 3, 5, 7, 9, 10, 12, 24, 36, 48]


def latest_generator(epoch_dir: Path) -> Path:
    """``g_<step>`` with the largest numeric step in ``epoch_dir``."""
    cands = [(int(m.group(1)), p) for p in epoch_dir.glob("g_*")
             if (m := re.match(r"g_(\d+)$", p.name))]
    if not cands:
        raise FileNotFoundError(f"no g_<step> file in {epoch_dir}")
    return max(cands)[1]


def resolve_checkpoints(a) -> dict:
    if a.checkpoints:
        out = {}
        for item in a.checkpoints:
            e, p = item.split("=", 1)
            out[int(e)] = Path(p)
        return out
    if not a.checkpoint_root:
        raise SystemExit("give --checkpoints epoch=path ... or --checkpoint-root")
    return {e: latest_generator(a.checkpoint_root / f"epoch_{e}") for e in a.epochs}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="*", help="epoch=path pairs")
    p.add_argument("--checkpoint-root", type=Path, default=None, help="dir with epoch_<N>/g_<step>")
    p.add_argument("--epochs", nargs="+", type=int, default=PAPER_EPOCHS)
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
    ckpts = resolve_checkpoints(a)
    hb = Heartbeat("emergence", a.log)
    device = get_device()
    hb(f"START {device_info(device)} epochs={sorted(ckpts)}")
    _, clean_all = load_voicebank_test(resolve_clean_source(a.clean), a.n_utts)
    batches = [(i, clean_all[i:i + a.utt_chunk]) for i in range(0, len(clean_all), a.utt_chunk)]
    total = 0
    for epoch, ckpt in sorted(ckpts.items()):
        out_dir = a.out_dir / f"e{epoch}" / "snr"

        def chunk_path(noise, snr, _e=epoch, _d=out_dir):
            return _d / f"e{_e}__{noise}__snr{snr:+03d}.parquet"

        conds = [(n, s) for n in a.noises for s in a.snrs]
        if all(chunk_path(*c).exists() for c in conds):
            hb(f"epoch {epoch}: all chunks present")
            continue
        hb(f"epoch {epoch}: checkpoint {ckpt}")
        ext = load_extractor("muse", device, checkpoint=str(ckpt))
        _ = ext(clean_all[0])

        def make_rows(noise_name, snr, _e=epoch, _ext=ext):
            noise = load_demand_noise(noise_name)
            rows = []
            for off, batch in batches:
                cs = extract_stack(_ext, batch, NORM1_LAYERS)
                ns = extract_stack(_ext, [add_noise_at_snr(c, noise, snr).astype("float32") for c in batch], NORM1_LAYERS)
                rows += cka_rows(cs, ns, device, {"snr": int(snr), "model_name": f"MUSE_FT_e{_e}",
                                                  "noise_name": noise_name, "epoch": int(_e)}, offset=off)
            return rows

        total += chunk_loop(conds, chunk_path, make_rows, hb)
        del ext
    hb(f"DONE: wrote {total} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
