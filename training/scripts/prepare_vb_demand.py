"""Extract VoiceBank-DEMAND 16kHz from HuggingFace to local directory structure.

Output: data/VB_DEMAND_16K/{clean_train,noisy_train,clean_test,noisy_test}/
Files named like p226_001.wav matching VoiceBank+DEMAND/{training,test}.txt IDs.
"""

import os
import sys
import importlib
import soundfile as sf

# The project has a local datasets/ package that shadows HuggingFace datasets.
# Temporarily remove the project root from sys.path to import the correct one.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_path_backup = sys.path[:]
sys.path = [p for p in sys.path if os.path.abspath(p) != _project_root]
sys.modules.pop("datasets", None)
from datasets import load_dataset  # HuggingFace datasets
sys.path = _path_backup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "VB_DEMAND_16K")

SPLITS = {
    "train": {
        "index_file": os.path.join(BASE_DIR, "VoiceBank+DEMAND", "training.txt"),
        "clean_dir": os.path.join(OUT_DIR, "clean_train"),
        "noisy_dir": os.path.join(OUT_DIR, "noisy_train"),
    },
    "test": {
        "index_file": os.path.join(BASE_DIR, "VoiceBank+DEMAND", "test.txt"),
        "clean_dir": os.path.join(OUT_DIR, "clean_test"),
        "noisy_dir": os.path.join(OUT_DIR, "noisy_test"),
    },
}


def read_index(path):
    """Read pipe-separated index file, return list of IDs."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.split("|")[0] for line in f.read().strip().split("\n") if line.strip()]


def main():
    ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")

    for split_name, paths in SPLITS.items():
        ids = read_index(paths["index_file"])
        id_set = set(ids)
        os.makedirs(paths["clean_dir"], exist_ok=True)
        os.makedirs(paths["noisy_dir"], exist_ok=True)

        print(f"Processing {split_name} split: {len(ids)} expected utterances")

        # Build lookup from HF dataset by filename
        hf_split = ds[split_name]
        found = 0
        for row in hf_split:
            # HF dataset has 'id' field matching our index IDs
            uid = row["id"]
            if uid not in id_set:
                continue

            clean_audio = row["clean"]["array"]
            sr = row["clean"]["sampling_rate"]
            noisy_audio = row["noisy"]["array"]

            sf.write(os.path.join(paths["clean_dir"], f"{uid}.wav"), clean_audio, sr)
            sf.write(os.path.join(paths["noisy_dir"], f"{uid}.wav"), noisy_audio, sr)
            found += 1

        print(f"  Saved {found}/{len(ids)} utterances")
        if found != len(ids):
            # Try matching without exact id — some datasets use filename as key
            missing = id_set - {row["id"] for row in hf_split}
            if missing:
                print(f"  WARNING: {len(missing)} IDs not found in HF dataset")
                print(f"  First few missing: {list(missing)[:5]}")

    print("Done. Output directory:", OUT_DIR)


if __name__ == "__main__":
    main()
