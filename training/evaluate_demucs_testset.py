"""Held-out test-set evaluation for the fine-tuned Demucs (DNS64) dereverb model.

Methodologically identical to evaluate_mpsenet_testset.py -- same test set, same
metrics (PESQ wb, STOI, ESTOI, SI-SDR), same enhanced-vs-reverberant-input
reporting, same JSON/CSV schema -- so the two can go in one table.

The ONLY thing that differs is the inference core, because Demucs is a
waveform-domain model:
  * MP-SENet: STFT -> model -> iSTFT, in segment_size chunks.
  * Demucs:   full-length waveform straight through; Demucs.forward pads to
              valid_length and crops internally, so no chunking. This mirrors
              validate() in train_demucs.py exactly.

Normalization follows datasets/reverb_waveform_dataset.py: scale the input by
sqrt(len / sum(reverb^2)), then undo that scale on the output so metrics are
computed against the unmodified clean reference (as in the MP-SENet harness).
STOI/ESTOI/SI-SDR are scale-invariant and PESQ level-aligns internally, so this
is equivalent to the training-time convention.

DNSMOS is deliberately not computed: speechmos.dnsmos is broken in this env.
"""

import os
import csv
import json
import time
import argparse
import multiprocessing as mp

import numpy as np
import torch
import librosa

from pesq import pesq
from pystoi import stoi

SR = 16000


def si_sdr(ref, est):
    """Scale-Invariant Signal-to-Distortion Ratio (dB)."""
    ref = ref - ref.mean()
    est = est - est.mean()
    s_target = np.dot(ref, est) / (np.dot(ref, ref) + 1e-8) * ref
    e_noise = est - s_target
    return 10 * np.log10(np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-8))


def load_demucs_model(checkpoint_path, device):
    """Build dns64() and load our {'generator': state_dict} checkpoint strictly.

    Matches train_demucs.py: `model = dns64()`, saved as
    torch.save({"generator": model.state_dict()}, ...).
    """
    from denoiser.pretrained import dns64

    model = dns64().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "generator" not in ckpt:
        raise ValueError(
            f"Expected a 'generator' key in {checkpoint_path}; got {list(ckpt.keys())[:10]}"
        )
    # strict=True: a silent architecture mismatch must fail loudly, not produce
    # garbage from a partially-initialised model.
    model.load_state_dict(ckpt["generator"], strict=True)
    model.eval()
    return model


def process_audio(reverb_wav, model, device):
    """Full-length waveform inference, mirroring train_demucs.py::validate."""
    x = torch.FloatTensor(reverb_wav).to(device)
    norm_factor = torch.sqrt(len(x) / torch.sum(x ** 2.0)).to(device)
    x = (x * norm_factor).unsqueeze(0)          # [1, T]
    enhanced = model(x).squeeze()               # Demucs pads/crops internally
    enhanced = enhanced / norm_factor           # undo scaling -> compare vs raw clean
    return enhanced.cpu().numpy()


def _safe(fn, default=np.nan):
    try:
        v = fn()
        return float(v) if np.isfinite(v) else default
    except Exception:
        return default


def metrics_worker(payload):
    """Compute all metrics for one utterance. Runs in a pool worker."""
    idx, meta, clean, enhanced, noisy = payload
    out = dict(meta)
    out["pesq"] = _safe(lambda: pesq(SR, clean, enhanced, "wb"))
    out["stoi"] = _safe(lambda: stoi(clean, enhanced, SR, extended=False))
    out["estoi"] = _safe(lambda: stoi(clean, enhanced, SR, extended=True))
    out["si_sdr"] = _safe(lambda: si_sdr(clean, enhanced))
    out["pesq_input"] = _safe(lambda: pesq(SR, clean, noisy, "wb"))
    out["stoi_input"] = _safe(lambda: stoi(clean, noisy, SR, extended=False))
    out["estoi_input"] = _safe(lambda: stoi(clean, noisy, SR, extended=True))
    out["si_sdr_input"] = _safe(lambda: si_sdr(clean, noisy))
    return out


def utterance_stream(rows, wavs_dir, clean_dir, model, device):
    """Yield (idx, meta, clean, enhanced, noisy), doing GPU inference on the fly."""
    t0 = time.time()
    for i, row in enumerate(rows):
        noisy, _ = librosa.load(os.path.join(wavs_dir, row["filepath"]), sr=SR)
        clean, _ = librosa.load(os.path.join(clean_dir, row["utterance_id"] + ".wav"), sr=SR)

        with torch.no_grad():
            enhanced = process_audio(noisy, model, device)

        n = min(len(clean), len(enhanced), len(noisy))
        meta = {
            "utterance_id": row["utterance_id"],
            "rir_id": row["rir_id"],
            "rt60": row.get("rt60", ""),
            "drr": row.get("drr", ""),
            "c50": row.get("c50", ""),
        }
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  inference {i+1}/{len(rows)}  ({el:.0f}s, {(i+1)/el:.1f} utt/s)", flush=True)
        yield (i, meta, clean[:n].astype(np.float64),
               enhanced[:n].astype(np.float64), noisy[:n].astype(np.float64))


def summarize(results):
    def agg(key):
        v = np.array([r[key] for r in results], dtype=float)
        v = v[np.isfinite(v)]
        return {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": int(v.size)}

    keys = ["pesq", "stoi", "estoi", "si_sdr",
            "pesq_input", "stoi_input", "estoi_input", "si_sdr_input"]
    s = {k: agg(k) for k in keys}
    s["delta"] = {m: s[m]["mean"] - s[m + "_input"]["mean"]
                  for m in ["pesq", "stoi", "estoi", "si_sdr"]}
    return s


def rt60_breakdown(results):
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 5.0)]
    out = []
    for lo, hi in bins:
        rs = [r for r in results if r["rt60"] not in ("", None) and lo <= float(r["rt60"]) < hi]
        if not rs:
            continue
        b = {"rt60_lo": lo, "rt60_hi": hi, "n": len(rs)}
        for k in ["pesq", "stoi", "estoi", "si_sdr",
                  "pesq_input", "stoi_input", "estoi_input", "si_sdr_input"]:
            v = np.array([r[k] for r in rs], dtype=float)
            v = v[np.isfinite(v)]
            b[k] = float(np.mean(v))
        out.append(b)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reverb_dir", required=True,
                        help="Test set dir (must be the SAME one used for MP-SENet)")
    parser.add_argument("--output", required=True, help="Output path prefix (no extension)")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    model = load_demucs_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded dns64 + checkpoint {args.checkpoint} ({n_params/1e6:.2f}M params)", flush=True)

    wavs_dir = os.path.join(args.reverb_dir, "wavs")
    clean_dir = os.path.join(args.reverb_dir, "clean_refs")
    meta_path = os.path.join(args.reverb_dir, "metadata.csv")
    with open(meta_path) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    # Fingerprint the test set so identity with the MP-SENet run is provable.
    import hashlib
    with open(meta_path, "rb") as f:
        meta_md5 = hashlib.md5(f.read()).hexdigest()
    print(f"Test set: {len(rows)} reverberant utts, "
          f"{len(set(r['utterance_id'] for r in rows))} clean utts, "
          f"{len(set(r['rir_id'] for r in rows))} RIRs", flush=True)
    print(f"metadata.csv md5: {meta_md5}", flush=True)

    t0 = time.time()
    stream = utterance_stream(rows, wavs_dir, clean_dir, model, device)
    with mp.Pool(args.workers) as pool:
        results = list(pool.imap(metrics_worker, stream, chunksize=4))
    print(f"Done in {time.time() - t0:.0f}s", flush=True)

    summary = summarize(results)
    breakdown = rt60_breakdown(results)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    csv_path = args.output + ".csv"
    fieldnames = ["utterance_id", "rir_id", "rt60", "drr", "c50",
                  "pesq", "stoi", "estoi", "si_sdr",
                  "pesq_input", "stoi_input", "estoi_input", "si_sdr_input"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    payload = {
        "model": "demucs_dns64",
        "checkpoint": os.path.abspath(args.checkpoint),
        "inference": ("full-length waveform through dns64 (Demucs pads to valid_length "
                      "internally); input scaled by sqrt(len/sum(x^2)), output rescaled back"),
        "test_set": {
            "dir": os.path.abspath(args.reverb_dir),
            "metadata_md5": meta_md5,
            "n_utterances": len(rows),
            "n_unique_clean": len(set(r["utterance_id"] for r in rows)),
            "n_unique_rirs": len(set(r["rir_id"] for r in rows)),
            "clean_source": "data/VB_DEMAND_16K/clean_test (VoiceBank+DEMAND/test.txt)",
            "rir_split": "test split of data/rir_split.json (202 test / 798 train RIRs)",
        },
        "summary": summary,
        "rt60_breakdown": breakdown,
        "notes": "DNSMOS omitted: speechmos.dnsmos is broken in this environment (returns nan).",
    }
    with open(args.output + ".json", "w") as f:
        json.dump(payload, f, indent=2)

    s = summary
    print("\n=== HELD-OUT TEST SET RESULTS (Demucs) ===")
    print(f"{'metric':<10} {'reverb input':>14} {'enhanced':>12} {'delta':>10}")
    for m, fmt in [("pesq", "{:.3f}"), ("stoi", "{:.4f}"), ("estoi", "{:.4f}"), ("si_sdr", "{:+.2f}")]:
        print(f"{m:<10} {fmt.format(s[m+'_input']['mean']):>14} "
              f"{fmt.format(s[m]['mean']):>12} {fmt.format(s['delta'][m]):>10}")
    print("\n--- Per-RT60 breakdown (enhanced / input) ---")
    for b in breakdown:
        print(f"  RT60=[{b['rt60_lo']:.1f},{b['rt60_hi']:.1f})s n={b['n']:<5} "
              f"PESQ {b['pesq']:.3f}/{b['pesq_input']:.3f}  "
              f"STOI {b['stoi']:.4f}/{b['stoi_input']:.4f}  "
              f"SI-SDR {b['si_sdr']:+.2f}/{b['si_sdr_input']:+.2f} dB")
    print(f"\nSaved: {csv_path}\n       {args.output}.json")


if __name__ == "__main__":
    main()
