"""Held-out test-set evaluation for the fine-tuned MP-SENet dereverb model.

Evaluates PESQ (wb), STOI, ESTOI and SI-SDR for BOTH the enhanced output and the
reverberant input (baseline), on test_sets/reverb (VB+DEMAND clean_test x
test-split RIRs).

DNSMOS is deliberately not computed: speechmos.dnsmos is broken in this env.

GPU inference runs in the main process; the (CPU-bound) metric computation is
farmed out to a worker pool so the 4120-utterance set finishes in minutes.

Usage:
    python evaluate_mpsenet_testset.py \
      --checkpoint checkpoints_mpsenet/all_v2/epoch_50/g_00295909 \
      --config checkpoints_mpsenet/all_v2/config.json \
      --output results/mpsenet_ft_test_eval
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

from env import AttrDict
from datasets.dataset import mag_pha_stft, mag_pha_istft
from models.mpsenet_generator import MPNet

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


def load_mpsenet_model(checkpoint_path, h, device):
    model = MPNet(
        dense_channel=h.dense_channel,
        n_fft=h.n_fft,
        sigmoid_beta=h.beta,
        num_tsblocks=h.num_tsblocks,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'generator' in ckpt:
        model.load_state_dict(ckpt['generator'])
    elif 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        gen_sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
        model.load_state_dict(gen_sd)
    else:
        raise ValueError(f"Unrecognized checkpoint format: {list(ckpt.keys())[:10]}")
    model.eval()
    return model


def process_audio(noisy_wav, model, h, device):
    """Segment-wise full-length inference (identical to evaluate_mpsenet.py)."""
    segment_size = h.segment_size
    n_fft, hop_size, win_size = h.n_fft, h.hop_size, h.win_size
    compress_factor = h.compress_factor

    noisy_wav = torch.FloatTensor(noisy_wav).to(device)
    norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
    noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
    orig_size = noisy_wav.size(1)

    if noisy_wav.size(1) >= segment_size:
        last_segment_size = noisy_wav.size(1) % segment_size
        if last_segment_size > 0:
            last_segment = noisy_wav[:, -segment_size:]
            noisy_wav_main = noisy_wav[:, :-last_segment_size]
            segments = list(torch.split(noisy_wav_main, segment_size, dim=1))
            segments.append(last_segment)
            reshapelast = 1
        else:
            segments = list(torch.split(noisy_wav, segment_size, dim=1))
            reshapelast = 0
            last_segment_size = 0
    else:
        padded_zeros = torch.zeros(1, segment_size - noisy_wav.size(1)).to(device)
        noisy_wav = torch.cat((noisy_wav, padded_zeros), dim=1)
        segments = [noisy_wav]
        reshapelast = 0
        last_segment_size = 0

    processed_segments = []
    for i, segment in enumerate(segments):
        noisy_amp, noisy_pha, _ = mag_pha_stft(segment, n_fft, hop_size, win_size, compress_factor)
        amp_g, pha_g, _ = model(noisy_amp.to(device), noisy_pha.to(device))
        audio_g = mag_pha_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
        audio_g = audio_g / norm_factor
        audio_g = audio_g.squeeze()
        if reshapelast == 1 and i == len(segments) - 2:
            audio_g = audio_g[:-(segment_size - last_segment_size)]
        processed_segments.append(audio_g)

    processed_audio = torch.cat(processed_segments, dim=-1)
    return processed_audio[:orig_size].cpu().numpy()


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


def utterance_stream(rows, wavs_dir, clean_dir, model, h, device):
    """Yield (idx, meta, clean, enhanced, noisy) doing GPU inference on the fly."""
    t0 = time.time()
    for i, row in enumerate(rows):
        noisy, _ = librosa.load(os.path.join(wavs_dir, row["filepath"]), sr=SR)
        clean, _ = librosa.load(os.path.join(clean_dir, row["utterance_id"] + ".wav"), sr=SR)

        with torch.no_grad():
            enhanced = process_audio(noisy, model, h, device)

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
    """Mean/std over utterances, ignoring nan."""
    def agg(key):
        v = np.array([r[key] for r in results], dtype=float)
        v = v[np.isfinite(v)]
        return {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": int(v.size)}

    keys = ["pesq", "stoi", "estoi", "si_sdr",
            "pesq_input", "stoi_input", "estoi_input", "si_sdr_input"]
    s = {k: agg(k) for k in keys}
    s["delta"] = {
        m: s[m]["mean"] - s[m + "_input"]["mean"]
        for m in ["pesq", "stoi", "estoi", "si_sdr"]
    }
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--reverb_dir", default="test_sets/reverb")
    parser.add_argument("--output", required=True, help="Output path prefix (no extension)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Debug: only N utterances")
    args = parser.parse_args()

    with open(args.config) as f:
        h = AttrDict(json.load(f))

    print(f"Config: n_fft={h.n_fft} hop={h.hop_size} win={h.win_size} "
          f"compress={h.compress_factor} dense_channel={h.dense_channel} "
          f"num_tsblocks={h.num_tsblocks} beta={h.beta} segment={h.segment_size}", flush=True)
    assert (h.n_fft, h.hop_size, h.win_size) == (400, 100, 400), \
        f"STFT config mismatch: got n_fft={h.n_fft}, hop={h.hop_size}, win={h.win_size}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    model = load_mpsenet_model(args.checkpoint, h, device)
    print(f"Loaded checkpoint {args.checkpoint}", flush=True)

    wavs_dir = os.path.join(args.reverb_dir, "wavs")
    clean_dir = os.path.join(args.reverb_dir, "clean_refs")
    with open(os.path.join(args.reverb_dir, "metadata.csv")) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    print(f"Test set: {len(rows)} reverberant utterances, "
          f"{len(set(r['utterance_id'] for r in rows))} unique clean utts, "
          f"{len(set(r['rir_id'] for r in rows))} unique RIRs", flush=True)

    t0 = time.time()
    stream = utterance_stream(rows, wavs_dir, clean_dir, model, h, device)
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
        "checkpoint": os.path.abspath(args.checkpoint),
        "config": os.path.abspath(args.config),
        "config_stft": {"n_fft": h.n_fft, "hop_size": h.hop_size, "win_size": h.win_size,
                        "compress_factor": h.compress_factor, "dense_channel": h.dense_channel,
                        "num_tsblocks": h.num_tsblocks, "beta": h.beta,
                        "segment_size": h.segment_size},
        "test_set": {
            "dir": os.path.abspath(args.reverb_dir),
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
    json_path = args.output + ".json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    s = summary
    print("\n=== HELD-OUT TEST SET RESULTS ===")
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
    print(f"\nSaved: {csv_path}\n       {json_path}")


if __name__ == "__main__":
    main()
