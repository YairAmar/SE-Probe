"""Split RIRs into train/test.

If rir_metadata.csv has a 'split' column (from official manifests):
  - Merge official "valid" into "train"
  - Keep "test" as "test"
Otherwise: fall back to 80/20 stratified split by RT60 bins.

Input:  data/rir_metadata.csv
Output: data/rir_split.json  -> {filename: "train"|"test"}
"""

import os
import csv
import json
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_PATH = os.path.join(BASE_DIR, "data", "rir_metadata.csv")
SPLIT_PATH = os.path.join(BASE_DIR, "data", "rir_split.json")

RT60_BINS = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, float("inf")),
]

TRAIN_RATIO = 0.8
SEED = 42


def get_rt60_bin(rt60):
    for i, (lo, hi) in enumerate(RT60_BINS):
        if lo <= rt60 < hi:
            return i
    return len(RT60_BINS) - 1


def split_from_official(rows):
    """Use the official 'split' column. Merge 'valid' into 'train'."""
    split = {}
    counts = {}
    for row in rows:
        official = row["split"].strip().lower()
        # Merge valid/validation into train (val = random holdout at runtime)
        if official in ("valid", "validation", "val"):
            label = "train"
        elif official == "test":
            label = "test"
        else:
            label = "train"
        split[row["filename"]] = label
        counts[official] = counts.get(official, 0) + 1

    print("Official split distribution:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    return split


def split_stratified(rows):
    """Fall back to 80/20 stratified split by RT60 bins."""
    random.seed(SEED)

    bins = {}
    invalid_rt60 = []
    for row in rows:
        rt60 = float(row["rt60"])
        if rt60 < 0:
            invalid_rt60.append(row["filename"])
            continue
        b = get_rt60_bin(rt60)
        bins.setdefault(b, []).append(row["filename"])

    split = {}

    for b in sorted(bins.keys()):
        filenames = bins[b]
        random.shuffle(filenames)
        n_train = max(1, int(len(filenames) * TRAIN_RATIO))
        for fn in filenames[:n_train]:
            split[fn] = "train"
        for fn in filenames[n_train:]:
            split[fn] = "test"
        lo, hi = RT60_BINS[b]
        hi_str = f"{hi:.1f}" if hi != float("inf") else "inf"
        print(f"RT60 bin [{lo:.1f}, {hi_str}): {len(filenames)} total, "
              f"{n_train} train, {len(filenames) - n_train} test")

    # Invalid RT60 RIRs go to train
    for fn in invalid_rt60:
        split[fn] = "train"
    if invalid_rt60:
        print(f"Invalid RT60 (assigned to train): {len(invalid_rt60)}")

    return split


def main():
    with open(META_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    has_split_col = "split" in rows[0] if rows else False
    has_family_col = "family" in rows[0] if rows else False

    if has_split_col:
        # Check that split column has actual values
        non_empty = sum(1 for r in rows if r.get("split", "").strip())
        if non_empty > 0:
            print(f"Found official 'split' column ({non_empty}/{len(rows)} non-empty)")
            split = split_from_official(rows)
        else:
            print("Split column exists but is empty; falling back to stratified split")
            split = split_stratified(rows)
    else:
        print("No 'split' column found; using stratified 80/20 split")
        split = split_stratified(rows)

    train_count = sum(1 for v in split.values() if v == "train")
    test_count = sum(1 for v in split.values() if v == "test")
    print(f"\nTotal: {len(split)} RIRs -> {train_count} train, {test_count} test")

    # Per-family stats
    if has_family_col:
        family_stats = {}
        for row in rows:
            fam = row.get("family", "unknown")
            label = split.get(row["filename"], "unknown")
            family_stats.setdefault(fam, {"train": 0, "test": 0})
            if label in family_stats[fam]:
                family_stats[fam][label] += 1
        for fam in sorted(family_stats):
            s = family_stats[fam]
            print(f"  {fam}: {s['train']} train, {s['test']} test")

    with open(SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Saved to {SPLIT_PATH}")


if __name__ == "__main__":
    main()
