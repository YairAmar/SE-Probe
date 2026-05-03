"""Generate reverb and noise test sets for evaluation.

Reverb test: 824 utterances x 5 random RIRs each -> test_sets/reverb/wavs/
  Uses all test-split RIR-Mega RIRs with random assignment (seeded).
  RIRs are loaded from the raw dataset dir, onset-clipped on the fly.
Noise test: symlinks from VB_DEMAND_16K/noisy_test -> test_sets/noise/wavs/
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VB_CLEAN_TEST = os.path.join(BASE_DIR, "data", "VB_DEMAND_16K", "clean_test")
VB_NOISY_TEST = os.path.join(BASE_DIR, "data", "VB_DEMAND_16K", "noisy_test")
RIR_SPLIT = os.path.join(BASE_DIR, "data", "rir_split.json")
RIR_METADATA = os.path.join(BASE_DIR, "data", "rir_metadata.csv")
TEST_INDEX = os.path.join(BASE_DIR, "VoiceBank+DEMAND", "test.txt")

REVERB_OUT = os.path.join(BASE_DIR, "test_sets", "reverb")
NOISE_OUT = os.path.join(BASE_DIR, "test_sets", "noise")

SAMPLE_RATE = 16000
SEED = 42
RIRS_PER_UTTERANCE = 5


def load_rir(rir_dir, wav_path, channel=0):
    """Load a single RIR: read WAV, extract channel, onset-clip at peak."""
    full_path = os.path.join(rir_dir, wav_path)
    rir, _ = sf.read(full_path, dtype="float32")
    if rir.ndim > 1:
        ch = min(channel, rir.shape[1] - 1)
        rir = rir[:, ch]
    peak_idx = np.argmax(np.abs(rir))
    return rir[peak_idx:]


def main():
    parser = argparse.ArgumentParser(description="Generate reverb and noise test sets")
    parser.add_argument("--rir-dir", default="/home/yairamr/work/data/rirs/rirmega_small",
                        help="Root directory of raw RIR dataset")
    args = parser.parse_args()

    rng = np.random.RandomState(SEED)

    # Read test utterance IDs
    with open(TEST_INDEX, "r") as f:
        test_ids = [line.split("|")[0] for line in f.read().strip().split("\n") if line.strip()]
    print(f"Test utterances: {len(test_ids)}")

    # Load RIR metadata (filename -> wav_path + acoustic params)
    rir_meta = {}
    with open(RIR_METADATA, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rir_meta[row["filename"]] = row

    # Load test-split RIR filenames
    with open(RIR_SPLIT, "r") as f:
        rir_split = json.load(f)
    test_rir_fns = sorted([fn for fn, s in rir_split.items() if s == "test"])
    print(f"Test-split RIRs: {len(test_rir_fns)}")

    # Load all test RIRs into memory (onset-clipped, channel 0)
    rirs = {}
    for i, fn in enumerate(test_rir_fns):
        meta = rir_meta.get(fn)
        if meta is None:
            print(f"  WARNING: {fn} not in metadata, skipping")
            continue
        wav_path = meta["wav_path"]
        rirs[fn] = load_rir(args.rir_dir, wav_path)
        if (i + 1) % 2000 == 0:
            print(f"  Loaded {i + 1}/{len(test_rir_fns)} test RIRs")
    # Update list to only include successfully loaded RIRs
    test_rir_fns = [fn for fn in test_rir_fns if fn in rirs]
    print(f"Loaded {len(rirs)} test RIRs into memory")

    print(f"Expected total: {len(test_ids)} x {RIRS_PER_UTTERANCE} = "
          f"{len(test_ids) * RIRS_PER_UTTERANCE} files")

    # --- Generate reverb test set ---
    reverb_wavs_dir = os.path.join(REVERB_OUT, "wavs")
    reverb_clean_dir = os.path.join(REVERB_OUT, "clean_refs")
    os.makedirs(reverb_wavs_dir, exist_ok=True)
    os.makedirs(reverb_clean_dir, exist_ok=True)

    reverb_rows = []
    for i, utt_id in enumerate(test_ids):
        clean_path = os.path.join(VB_CLEAN_TEST, f"{utt_id}.wav")
        clean, sr = sf.read(clean_path, dtype="float64")

        # Symlink clean ref
        ref_link = os.path.join(reverb_clean_dir, f"{utt_id}.wav")
        if not os.path.exists(ref_link):
            os.symlink(os.path.abspath(clean_path), ref_link)

        # Randomly pick RIRS_PER_UTTERANCE RIRs (without replacement)
        chosen = rng.choice(test_rir_fns, size=RIRS_PER_UTTERANCE, replace=False)

        for rir_fn in chosen:
            rir = rirs[rir_fn]

            reverb = fftconvolve(clean, rir, mode="full")[:len(clean)]

            out_name = f"{utt_id}_{rir_fn}.wav"
            sf.write(os.path.join(reverb_wavs_dir, out_name), reverb, sr)

            meta = rir_meta.get(rir_fn, {})
            reverb_rows.append({
                "utterance_id": utt_id,
                "rir_id": rir_fn,
                "rir_filename": meta.get("wav_path", ""),
                "rt60": meta.get("rt60", ""),
                "drr": meta.get("drr", ""),
                "c50": meta.get("c50", ""),
                "filepath": out_name,
            })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_ids)} utterances "
                  f"({len(reverb_rows)} files so far)")

    # Write reverb metadata
    reverb_meta_path = os.path.join(REVERB_OUT, "metadata.csv")
    fieldnames = ["utterance_id", "rir_id", "rir_filename", "rt60", "drr", "c50", "filepath"]
    with open(reverb_meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reverb_rows)
    print(f"Reverb test set: {len(reverb_rows)} files -> {reverb_wavs_dir}")

    # --- Generate noise test set (symlinks) ---
    noise_wavs_dir = os.path.join(NOISE_OUT, "wavs")
    noise_clean_dir = os.path.join(NOISE_OUT, "clean_refs")
    os.makedirs(noise_wavs_dir, exist_ok=True)
    os.makedirs(noise_clean_dir, exist_ok=True)

    noise_rows = []
    for utt_id in test_ids:
        noisy_path = os.path.join(VB_NOISY_TEST, f"{utt_id}.wav")
        clean_path = os.path.join(VB_CLEAN_TEST, f"{utt_id}.wav")

        noisy_link = os.path.join(noise_wavs_dir, f"{utt_id}.wav")
        if not os.path.exists(noisy_link):
            os.symlink(os.path.abspath(noisy_path), noisy_link)

        clean_link = os.path.join(noise_clean_dir, f"{utt_id}.wav")
        if not os.path.exists(clean_link):
            os.symlink(os.path.abspath(clean_path), clean_link)

        noise_rows.append({
            "utterance_id": utt_id,
            "filepath": f"{utt_id}.wav",
        })

    noise_meta_path = os.path.join(NOISE_OUT, "metadata.csv")
    with open(noise_meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utterance_id", "filepath"])
        writer.writeheader()
        writer.writerows(noise_rows)
    print(f"Noise test set: {len(noise_rows)} files -> {noise_wavs_dir}")


if __name__ == "__main__":
    main()
