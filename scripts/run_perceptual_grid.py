#!/usr/bin/env python
"""Per-mixture output-quality metrics for the quality-association analysis (paper
Sec. II-D "Quality-Association Analysis" and Sec. III-B).

Grid: 824 VoiceBank-DEMAND test utterances x 41 integer SNRs x the five test noises
(TBUS, SCAFE, DLIVING, OOFFICE, SPSQUARE) x {MUSE, MP-SENet, Demucs}. For every mixture
the model enhances the noisy input and PESQ (wb), STOI, SI-SDR and the three DNSMOS
scores are computed for both the noisy input and the enhanced output.

This is the repaired v2 pipeline: MP-SENet is built through the canonical loader,
DNSMOS fails loudly instead of silently writing NaN, the DNSMOS keys are the real
``sig_mos/bak_mos/ovrl_mos``, and the SI-SDR helper is self-tested for its sign at
start-up. One parquet per (model, noise, snr) under ``<out-dir>/<model>/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from _clean_sources import load_voicebank_test, resolve_clean_source  # noqa: E402
from _grid import MODEL_NAMES, Heartbeat, chunk_loop  # noqa: E402

from se_probe.data_generation import add_noise_at_snr, load_demand_noise  # noqa: E402
from se_probe.device import device_info, get_device  # noqa: E402
from se_probe.layers import SNR_GRID, VOICEBANK_DEMAND_TEST_NOISES  # noqa: E402
from se_probe.metrics import sisdr  # noqa: E402

METRIC_COLS = [f"{p}_{m}" for p in ("noisy", "enhanced")
               for m in ("pesq", "stoi", "sisdr", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovrl")]


def assert_sisdr_sign() -> None:
    """Abort unless ``se_probe.metrics.sisdr`` returns +SI-SDR."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000).astype(np.float32)
    n = rng.standard_normal(16000).astype(np.float32)
    if not sisdr(x, x) > 50:
        raise RuntimeError("SI-SDR sign check FAILED: sisdr(x, x) is not a large positive number")
    for snr_db in (-10, 0, 10, 20):
        y = x + n * (np.linalg.norm(x) / np.linalg.norm(n)) * 10 ** (-snr_db / 20)
        if abs(sisdr(x, y) - snr_db) > 1.5:
            raise RuntimeError(f"SI-SDR calibration FAILED at {snr_db} dB: got {sisdr(x, y):.3f}")


def load_enhancer(model_key: str, device, checkpoint=None):
    """``enhance(np.float32 @16 kHz) -> np.float32`` for one model."""
    if model_key == "muse":
        from se_probe.muse.model import load_muse_model
        model = load_muse_model(device=device, checkpoint_path=checkpoint)
        model.eval()

        def enh(a):
            with torch.no_grad():
                x = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device).unsqueeze(0)
                y = model(x)
                y = y[0] if isinstance(y, (tuple, list)) else y
            return y.squeeze().detach().cpu().numpy().astype("float32")[:len(a)]
        return enh
    if model_key == "mpsenet":
        from se_probe.mpsenet.model import load_mpsenet_model
        wrapper = load_mpsenet_model(device=device, checkpoint_path=checkpoint)
        model = wrapper._model
        model.eval()

        def enh(a):
            with torch.no_grad():
                out = model(np.asarray(a, dtype=np.float32))
            y = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(out, (tuple, list)) and len(out) > 1 and int(out[1]) != 16000:
                raise RuntimeError(f"MP-SENet returned sr={out[1]}, expected 16000")
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().numpy()
            return np.asarray(y, dtype="float32").squeeze()[:len(a)]
        return enh
    if model_key == "demucs":
        from se_probe.demucs.model import load_demucs_model
        wrapper = load_demucs_model(device=device, checkpoint_path=checkpoint)
        model = wrapper._model
        model.eval()

        def enh(a):
            with torch.no_grad():
                x = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device).unsqueeze(0).unsqueeze(0)
                y = model(x)
            return y.squeeze().detach().cpu().numpy().astype("float32")[:len(a)]
        return enh
    raise ValueError(model_key)


def compute_all_metrics(clean, degraded, dnsmos_eval, prefix: str, strict_dnsmos: bool) -> dict:
    import pesq as pesq_lib
    from pystoi import stoi as stoi_fn

    n = min(len(clean), len(degraded))
    r = np.asarray(clean[:n], dtype=np.float32).squeeze()
    t = np.asarray(degraded[:n], dtype=np.float32).squeeze()
    out = {}
    try:
        out[f"{prefix}_pesq"] = float(pesq_lib.pesq(16000, r, t, "wb"))
    except Exception:
        out[f"{prefix}_pesq"] = float("nan")
    try:
        out[f"{prefix}_stoi"] = float(stoi_fn(r, t, 16000, extended=False))
    except Exception:
        out[f"{prefix}_stoi"] = float("nan")
    try:
        out[f"{prefix}_sisdr"] = float(sisdr(r, t))
    except Exception:
        out[f"{prefix}_sisdr"] = float("nan")
    if dnsmos_eval is None:
        for k in ("sig", "bak", "ovrl"):
            out[f"{prefix}_dnsmos_{k}"] = float("nan")
        return out
    try:
        dm = dnsmos_eval(t)
        out[f"{prefix}_dnsmos_sig"] = float(dm["sig_mos"])
        out[f"{prefix}_dnsmos_bak"] = float(dm["bak_mos"])
        out[f"{prefix}_dnsmos_ovrl"] = float(dm["ovrl_mos"])
    except Exception:
        if strict_dnsmos:
            raise
        for k in ("sig", "bak", "ovrl"):
            out[f"{prefix}_dnsmos_{k}"] = float("nan")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["muse", "mpsenet", "demucs"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--clean", default=None)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--noises", nargs="+", default=VOICEBANK_DEMAND_TEST_NOISES)
    p.add_argument("--snrs", nargs="+", type=int, default=[int(s) for s in SNR_GRID])
    p.add_argument("--n-utts", type=int, default=824)
    p.add_argument("--no-dnsmos", action="store_true", help="skip DNSMOS (columns written as NaN)")
    p.add_argument("--log", type=Path, default=None)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    hb = Heartbeat(f"perceptual/{a.model}", a.log)
    device = get_device()
    hb(f"START {device_info(device)} model={a.model} n_utts={a.n_utts}")
    assert_sisdr_sign()
    hb("sisdr sign check OK")
    out_dir = a.out_dir / a.model

    def chunk_path(noise, snr):
        return out_dir / f"perc_{a.model}__{noise}__snr{snr:+03d}.parquet"

    conds = [(n, s) for n in a.noises for s in a.snrs]
    if all(chunk_path(*c).exists() for c in conds):
        hb("all chunks present, nothing to do")
        return 0
    _, clean_all = load_voicebank_test(resolve_clean_source(a.clean), a.n_utts)
    enh = load_enhancer(a.model, device, a.checkpoint)
    dnsmos = None
    if not a.no_dnsmos:
        from se_probe.metrics import (
            GPUDNSMOSEvaluator,  # fails loudly if onnx2torch/speechmos missing
        )
        dnsmos = GPUDNSMOSEvaluator(sample_rate=16000, device=device)
        _ = dnsmos(clean_all[0])
    model_name = MODEL_NAMES[a.model]

    def make_rows(noise_name, snr):
        noise = load_demand_noise(noise_name)
        rows = []
        for u, clean in enumerate(clean_all):
            noisy = add_noise_at_snr(clean, noise, snr).astype("float32")
            enhanced = enh(noisy)
            L = min(len(clean), len(noisy), len(enhanced))
            row = {"clean_idx": u, "snr": int(snr), "noise_name": noise_name, "model_name": model_name}
            row.update(compute_all_metrics(clean[:L], noisy[:L], dnsmos, "noisy", not a.no_dnsmos))
            row.update(compute_all_metrics(clean[:L], enhanced[:L], dnsmos, "enhanced", not a.no_dnsmos))
            rows.append(row)
        return rows

    n = chunk_loop(conds, chunk_path, make_rows, hb)
    hb(f"DONE {a.model}: wrote {n} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
