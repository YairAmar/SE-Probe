"""Compute acoustic metadata (RT60, DRR, C50, C80) for RIR-Mega RIRs.

Walks source directories, reads each multi-channel WAV, extracts a single
channel, onset-clips at the absolute peak, and computes acoustic parameters
on the clipped signal.  Does NOT write any clipped files — onset clipping
is done on-the-fly when loading RIRs in the dataset classes.

Supports RIR-Mega v1.2.1 layout:
  src_dir/rir_output_50k/audio/*.wav   (linear-array, 8-channel)
  src_dir/rir_output_8k_circ/audio/*.wav (circular-array, 8-channel)

Also reads the manifest CSVs (if present) for family/id metadata.

Output: rir_metadata.csv with columns:
  filename, wav_path, family, rt60, drr, c50, c80
  where wav_path is relative to --src-dir.
"""

import os
import argparse
import csv
import numpy as np
import soundfile as sf


def compute_rt60_schroeder(rir, sr):
    """Estimate RT60 using Schroeder backward integration."""
    energy = rir ** 2
    schroeder = np.cumsum(energy[::-1])[::-1]
    schroeder_db = 10 * np.log10(schroeder / (schroeder[0] + 1e-12) + 1e-12)

    idx_5 = np.argmax(schroeder_db <= -5)
    idx_25 = np.argmax(schroeder_db <= -25)

    if idx_5 == 0 or idx_25 == 0 or idx_25 <= idx_5:
        idx_15 = np.argmax(schroeder_db <= -15)
        if idx_15 > idx_5 > 0:
            slope = (schroeder_db[idx_15] - schroeder_db[idx_5]) / ((idx_15 - idx_5) / sr)
            if abs(slope) < 1e-6:
                return -1.0
            return -60.0 / slope
        return -1.0

    slope = (schroeder_db[idx_25] - schroeder_db[idx_5]) / ((idx_25 - idx_5) / sr)
    if abs(slope) < 1e-6:
        return -1.0
    rt60 = -60.0 / slope
    return rt60 if rt60 > 0 else -1.0


def compute_drr(rir, sr, direct_ms=2.5):
    """Direct-to-reverberant ratio in dB."""
    direct_samples = max(1, int(sr * direct_ms / 1000))
    direct_energy = np.sum(rir[:direct_samples] ** 2)
    reverb_energy = np.sum(rir[direct_samples:] ** 2)
    if reverb_energy < 1e-12:
        return 100.0
    return 10 * np.log10(direct_energy / reverb_energy + 1e-12)


def compute_clarity(rir, sr, t_ms):
    """Clarity index C_t (e.g., C50, C80) in dB."""
    t_samples = int(sr * t_ms / 1000)
    early_energy = np.sum(rir[:t_samples] ** 2)
    late_energy = np.sum(rir[t_samples:] ** 2)
    if late_energy < 1e-12:
        return 100.0
    return 10 * np.log10(early_energy / late_energy + 1e-12)


def load_manifest_lookup(src_dir):
    """Load manifest CSVs to get family info per wav_path.

    Returns dict: wav_path (relative to src_dir) -> family string.
    """
    lookup = {}
    manifest_dir = os.path.join(src_dir, "manifests")
    if not os.path.isdir(manifest_dir):
        return lookup
    for csv_file in sorted(os.listdir(manifest_dir)):
        if not csv_file.endswith(".csv"):
            continue
        path = os.path.join(manifest_dir, csv_file)
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                wp = row.get("wav_path", "").strip()
                fam = row.get("family", "").strip()
                if wp:
                    lookup[wp] = fam
    return lookup


def collect_wav_files(src_dir):
    """Walk src_dir for WAV files under known subdirectories.

    Returns list of (absolute_path, wav_path_relative_to_src_dir, family_guess).
    """
    results = []
    for entry in sorted(os.listdir(src_dir)):
        subdir = os.path.join(src_dir, entry)
        if not os.path.isdir(subdir):
            continue
        # Skip non-audio directories like manifests/, checksums/
        if entry in ("manifests", "checksums"):
            continue
        family_guess = "circular" if "circ" in entry.lower() else "linear"
        for root, _, files in os.walk(subdir):
            for fname in sorted(files):
                if fname.lower().endswith(".wav"):
                    abs_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(abs_path, src_dir)
                    results.append((abs_path, rel_path, family_guess))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute acoustic metadata for RIR-Mega RIRs (no file copying)")
    parser.add_argument("--src-dir", default="/home/yairamr/work/data/rirs/rirmega_small",
                        help="Root dir containing rir_output_50k/, rir_output_8k_circ/, manifests/")
    parser.add_argument("--meta-out", default=None,
                        help="Output CSV path (default: data/rir_metadata.csv)")
    parser.add_argument("--channel", type=int, default=0,
                        help="Channel to extract from multi-channel recordings (default: 0)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta_path = args.meta_out or os.path.join(base_dir, "data", "rir_metadata.csv")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    print(f"Source: {args.src_dir}")
    print(f"Metadata output: {meta_path}")
    print(f"Channel: {args.channel}")

    # Load manifest for per-RIR family info
    manifest_lookup = load_manifest_lookup(args.src_dir)
    print(f"Manifest entries: {len(manifest_lookup)}")

    wav_files = collect_wav_files(args.src_dir)
    print(f"Found {len(wav_files)} WAV files")

    rows = []
    family_counts = {}

    for i, (abs_path, rel_path, family_guess) in enumerate(wav_files):
        rir, sr = sf.read(abs_path, dtype="float64")

        # Extract single channel
        if rir.ndim > 1:
            ch = min(args.channel, rir.shape[1] - 1)
            rir = rir[:, ch]

        # Onset clip at absolute peak (for metadata computation only)
        peak_idx = np.argmax(np.abs(rir))
        rir_clipped = rir[peak_idx:]

        # Compute metadata on clipped RIR
        rt60 = compute_rt60_schroeder(rir_clipped, sr)
        drr = compute_drr(rir_clipped, sr)
        c50 = compute_clarity(rir_clipped, sr, 50)
        c80 = compute_clarity(rir_clipped, sr, 80)

        # Use manifest family if available, else guess from dirname
        family = manifest_lookup.get(rel_path, family_guess)

        # filename = flat identifier for split_rirs.py and dataset loading
        # Use the relative wav_path so datasets can reconstruct the full path
        stem = os.path.splitext(os.path.basename(abs_path))[0]
        family_prefix = "circ" if "circ" in rel_path.lower() else "lin"
        filename = f"{family_prefix}_{stem}"

        rows.append({
            "filename": filename,
            "wav_path": rel_path,
            "family": family,
            "rt60": f"{rt60:.4f}",
            "drr": f"{drr:.2f}",
            "c50": f"{c50:.2f}",
            "c80": f"{c80:.2f}",
        })

        family_counts[family] = family_counts.get(family, 0) + 1

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(wav_files)} files")

    # Write metadata CSV
    fieldnames = ["filename", "wav_path", "family", "rt60", "drr", "c50", "c80"]
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Stats
    rt60_vals = [float(r["rt60"]) for r in rows if float(r["rt60"]) > 0]
    print(f"\nRT60 stats: mean={np.mean(rt60_vals):.3f}, "
          f"median={np.median(rt60_vals):.3f}, "
          f"min={np.min(rt60_vals):.3f}, max={np.max(rt60_vals):.3f}")
    invalid_count = sum(1 for r in rows if float(r["rt60"]) < 0)
    if invalid_count:
        print(f"  Invalid RT60 count: {invalid_count}")
    for fam, cnt in sorted(family_counts.items()):
        print(f"  {fam}: {cnt} RIRs")
    print(f"Saved metadata ({len(rows)} entries) to {meta_path}")


if __name__ == "__main__":
    main()
